import json
import re
import shlex
import struct
import subprocess
import traceback
from pathlib import Path

import gradio as gr
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PRESET_CHOICES = [
    "(none)",
    "flux2",
    "t5xxl",
    "hunyuan",
    "wan",
    "custom (type below)",
]

FP8_DTYPE_NAMES = {"F8_E4M3", "F8_E5M2"}
FP8_TORCH_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}

# Documented "K2" exclusion profile for Krea2 ConvRot conversions: these layers
# should stay at higher precision (per community Krea2 ConvRot releases).
KREA2_K2_KEYWORDS = ["first", "last", "tmlp", "tproj", "txtfusion", "txtmlp"]
KREA2_K2_DEFAULT_PATTERN = "diffusion_model\\.(" + "|".join(KREA2_K2_KEYWORDS) + ")\\."


def read_safetensors_header(path):
    """Read only the JSON header of a safetensors file (no tensor data)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header


def find_companion_scale(key, all_keys):
    """Look for a common per-layer scale-tensor naming convention next to a weight key."""
    base = key[: -len(".weight")] if key.endswith(".weight") else key
    candidates = [
        key + "_scale",
        key + ".scale",
        base + ".scale_weight",
        base + ".weight_scale",
        base + "_scale",
    ]
    for c in candidates:
        if c in all_keys:
            return c
    return None


def scan_fp8_bias_layers(path):
    """Return sorted layer prefixes (key without '.bias') whose bias tensor is FP8."""
    header = read_safetensors_header(path)
    layers = []
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if key.endswith(".bias") and info.get("dtype") in FP8_DTYPE_NAMES:
            layers.append(key[: -len(".bias")])
    return sorted(set(layers))


def scan_fp8_scale_metadata(path):
    """Check whether FP8 weight tensors have a companion per-layer scale tensor
    (scaled FP8) or not (naive FP8)."""
    header = read_safetensors_header(path)
    all_keys = set(header.keys())
    fp8_weight_keys = [
        k for k, v in header.items()
        if k != "__metadata__" and k.endswith(".weight") and v.get("dtype") in FP8_DTYPE_NAMES
    ]
    scaled, unscaled = [], []
    for k in fp8_weight_keys:
        companion = find_companion_scale(k, all_keys)
        (scaled if companion else unscaled).append((k, companion))
    return {
        "total_fp8_weights": len(fp8_weight_keys),
        "scaled": scaled,
        "unscaled": unscaled,
        "metadata": header.get("__metadata__", {}) or {},
    }


def scan_scale_metadata_ui(uploaded_file, manual_path):
    try:
        if manual_path and manual_path.strip():
            path = manual_path.strip()
        elif uploaded_file is not None:
            path = uploaded_file.name
        else:
            raise gr.Error("Provide a checkpoint via upload or path first.")

        path = Path(path)
        if not path.is_file():
            raise gr.Error(f"Input file not found: {path}")

        info = scan_fp8_scale_metadata(path)
        log = (
            f"FP8 weight tensors: {info['total_fp8_weights']}\n"
            f"  - with a companion scale tensor (scaled FP8): {len(info['scaled'])}\n"
            f"  - with no companion scale tensor (naive FP8): {len(info['unscaled'])}\n\n"
        )
        if info["scaled"]:
            log += "Example scaled pairs:\n" + "\n".join(
                f"  {w} -> {s}" for w, s in info["scaled"][:8]
            ) + "\n\n"
        if info["unscaled"]:
            log += "Example unscaled weights:\n" + "\n".join(
                f"  {w}" for w, _ in info["unscaled"][:8]
            ) + "\n\n"
        if info["metadata"]:
            log += f"File metadata: {info['metadata']}\n\n"

        if info["scaled"] and not info["unscaled"]:
            log += (
                "=> This looks like SCALED FP8. Use the Dequantize step below "
                "before running INT8 ConvRot conversion, or ctq will quantize "
                "the raw unscaled FP8 values directly, producing garbage output."
            )
        elif info["unscaled"] and not info["scaled"]:
            log += (
                "=> This looks like NAIVE FP8 (no companion scale found). "
                "Dequantizing to BF16 first is still recommended for best "
                "results, but the scale-mismatch risk described above doesn't apply."
            )
        else:
            log += "=> Mixed: some weights scaled, some not. Dequantizing first is recommended."
        return log
    except gr.Error:
        raise
    except Exception as e:
        return f"Error: {e}\n\n{traceback.format_exc()}"


QUANT_METADATA_KEY_SUBSTRINGS = ["quant"]
QUANT_MARKER_TENSOR_SUBSTRINGS = ["comfy_quant", "quant_config", ".qtype", ".quant_type"]


def is_quant_marker_tensor(key):
    lower = key.lower()
    return any(s in lower for s in QUANT_MARKER_TENSOR_SUBSTRINGS)


def dequantize_fp8_to_bf16(input_path, output_path, progress):
    tensors = {}
    metadata = {}
    stats_log = []
    stripped_metadata_keys = []
    dropped_marker_tensors = []

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        all_keys = set(keys)
        meta = f.metadata()
        if meta:
            for k, v in meta.items():
                if any(s in k.lower() for s in QUANT_METADATA_KEY_SUBSTRINGS):
                    stripped_metadata_keys.append(k)
                    continue
                metadata[k] = v

        # Only mark a "*_scale"/"*.scale" tensor as consumed if it is the
        # confirmed companion of an actual FP8 weight tensor. Never skip by
        # name pattern alone — real model weights (e.g. a norm layer's
        # learnable "scale"/gain parameter) can share these suffixes and
        # must be copied through untouched.
        header = read_safetensors_header(input_path)
        fp8_weight_keys = [
            k for k, v in header.items()
            if k != "__metadata__" and k.endswith(".weight") and v.get("dtype") in FP8_DTYPE_NAMES
        ]
        companion_map = {}
        consumed_keys = set()
        for wk in fp8_weight_keys:
            companion = find_companion_scale(wk, all_keys)
            if companion is not None:
                companion_map[wk] = companion
                consumed_keys.add(companion)

        n = len(keys)
        n_dequant, n_copied, n_sample_logged = 0, 0, 0

        for i, key in enumerate(keys):
            progress((i + 1) / n, desc=f"Processing {key}")
            if key in consumed_keys:
                continue
            if is_quant_marker_tensor(key):
                # Per-tensor quantization marker (e.g. a "*.comfy_quant" tag)
                # left over from the source FP8 file. The output is plain
                # BF16, so this marker must not carry over — leaving it in
                # makes ComfyUI think the layer is still quantized and it
                # will fail to find the (no-longer-present) companion
                # quantized-tensor data at load time.
                dropped_marker_tensors.append(key)
                continue

            tensor = f.get_tensor(key)
            if tensor.dtype in FP8_TORCH_DTYPES:
                companion = companion_map.get(key)
                t = tensor.to(torch.float32)
                if companion is not None:
                    scale = f.get_tensor(companion).to(torch.float32)
                    t = t * scale
                bf16 = t.to(torch.bfloat16)
                tensors[key] = bf16
                n_dequant += 1
                if n_sample_logged < 6:
                    stats_log.append(
                        f"  {key}: scale={'yes (' + companion + ')' if companion else 'no'}, "
                        f"abs-mean={bf16.float().abs().mean().item():.5f}, "
                        f"abs-max={bf16.float().abs().max().item():.5f}"
                    )
                    n_sample_logged += 1
            else:
                tensors[key] = tensor
                n_copied += 1

    metadata["dequantized_from_fp8"] = "true"
    save_file(tensors, str(output_path), metadata=metadata)

    log = (
        f"Dequantized {n_dequant} FP8 tensors to BF16 (using embedded scale where found).\n"
        f"Copied {n_copied} other tensors unchanged.\n"
        f"Dropped {len(dropped_marker_tensors)} leftover quantization marker tensor(s)"
        + (f": {', '.join(dropped_marker_tensors[:10])}" + (" ..." if len(dropped_marker_tensors) > 10 else "") if dropped_marker_tensors else "")
        + f".\nStripped {len(stripped_metadata_keys)} quantization-related metadata key(s)"
        + (f": {', '.join(stripped_metadata_keys)}" if stripped_metadata_keys else "")
        + f".\n\nSample dequantized tensor stats (sanity check — values should look like "
        f"typical near-unit-scale weights, not near-zero or huge):\n"
        + "\n".join(stats_log)
        + f"\n\nSaved to: {output_path}"
    )
    return log, str(output_path)


def run_dequantize(uploaded_file, manual_path, output_name, progress=gr.Progress()):
    try:
        if manual_path and manual_path.strip():
            input_path = manual_path.strip()
        elif uploaded_file is not None:
            input_path = uploaded_file.name
        else:
            raise gr.Error("Provide a checkpoint via upload or path first.")

        input_path = Path(input_path)
        if not input_path.is_file():
            raise gr.Error(f"Input file not found: {input_path}")

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        if output_name and output_name.strip():
            out_name = output_name.strip()
            if not out_name.endswith(".safetensors"):
                out_name += ".safetensors"
        else:
            out_name = input_path.stem + "_bf16.safetensors"

        output_path = out_dir / out_name
        log, saved_path = dequantize_fp8_to_bf16(input_path, output_path, progress)
        return log, saved_path
    except gr.Error:
        raise
    except Exception as e:
        return f"Error: {e}\n\n{traceback.format_exc()}", None


def scan_for_fp8_biases(uploaded_file, manual_path, current_pattern):
    try:
        if manual_path and manual_path.strip():
            path = manual_path.strip()
        elif uploaded_file is not None:
            path = uploaded_file.name
        else:
            raise gr.Error("Provide a checkpoint via upload or path first.")

        path = Path(path)
        if not path.is_file():
            raise gr.Error(f"Input file not found: {path}")

        fp8_bias_layers = scan_fp8_bias_layers(path)

        # Merge with the known Krea2 K2 profile so we never lose the
        # documented exclusions, and add anything the scan finds beyond it.
        already_covered = lambda name: any(kw in name for kw in KREA2_K2_KEYWORDS)
        extra_layers = [l for l in fp8_bias_layers if not already_covered(l)]

        parts = [re.escape(kw) for kw in KREA2_K2_KEYWORDS]
        parts += [re.escape(l.split(".")[-1]) for l in extra_layers]
        pattern = "diffusion_model\\.(" + "|".join(dict.fromkeys(parts)) + ")\\."

        log = (
            f"FP8 bias tensors found ({len(fp8_bias_layers)}):\n"
            + ("\n".join(f"  - {l}" for l in fp8_bias_layers) if fp8_bias_layers else "  (none)")
            + f"\n\nMerged with Krea2 K2 profile ({', '.join(KREA2_K2_KEYWORDS)}).\n"
            + f"Exclude-layers pattern:\n{pattern}"
        )
        return log, pattern
    except gr.Error:
        raise
    except Exception as e:
        return f"Error: {e}\n\n{traceback.format_exc()}", current_pattern


def build_command(input_path, output_path, use_int8, scaling_mode, block_size,
                   use_convrot, convrot_group_size, use_comfy_quant, save_metadata,
                   preset, custom_preset_flag, extra_flags, use_simple, use_low_memory,
                   exclude_layers):
    cmd = ["ctq", "-i", input_path, "-o", output_path]

    if use_int8:
        cmd.append("--int8")
        cmd += ["--scaling_mode", scaling_mode]
        if scaling_mode == "block":
            cmd += ["--block_size", str(int(block_size))]

    if use_simple:
        cmd.append("--simple")
    if use_low_memory:
        cmd.append("--low-memory")

    if use_convrot:
        cmd.append("--convrot")
        cmd += ["--convrot-group-size", str(int(convrot_group_size))]

    if use_comfy_quant:
        cmd.append("--comfy_quant")
    if save_metadata:
        cmd.append("--save-quant-metadata")

    if exclude_layers.strip():
        cmd += ["--exclude-layers", exclude_layers.strip()]

    if preset == "custom (type below)" and custom_preset_flag.strip():
        flag = custom_preset_flag.strip()
        if not flag.startswith("--"):
            flag = "--" + flag
        cmd.append(flag)
    elif preset not in ("(none)", "custom (type below)"):
        cmd.append(f"--{preset}")

    if extra_flags.strip():
        cmd += shlex.split(extra_flags.strip())

    return cmd


def run_conversion(uploaded_file, manual_path, output_name, use_int8, scaling_mode,
                    block_size, use_convrot, convrot_group_size, use_comfy_quant,
                    save_metadata, preset, custom_preset_flag, extra_flags,
                    use_simple, use_low_memory, exclude_layers, progress=gr.Progress()):
    try:
        if manual_path and manual_path.strip():
            input_path = manual_path.strip()
        elif uploaded_file is not None:
            input_path = uploaded_file.name
        else:
            raise gr.Error("Provide a checkpoint via upload or path.")

        input_path = Path(input_path)
        if not input_path.is_file():
            raise gr.Error(f"Input file not found: {input_path}")

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        if output_name and output_name.strip():
            out_name = output_name.strip()
            if not out_name.endswith(".safetensors"):
                out_name += ".safetensors"
        else:
            suffix = "_int8_convrot" if use_convrot else "_int8"
            out_name = input_path.stem + suffix + ".safetensors"

        output_path = out_dir / out_name

        cmd = build_command(
            str(input_path), str(output_path), use_int8, scaling_mode, block_size,
            use_convrot, convrot_group_size, use_comfy_quant, save_metadata,
            preset, custom_preset_flag, extra_flags, use_simple, use_low_memory,
            exclude_layers,
        )

        progress(0.05, desc="Starting ctq...")
        log_lines = ["$ " + " ".join(shlex.quote(c) for c in cmd), ""]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            log_lines.append(line.rstrip())
        process.wait()

        if process.returncode != 0:
            log_lines.append(f"\nctq exited with code {process.returncode}")
            return "\n".join(log_lines), None

        progress(1.0, desc="Done")
        log_lines.append(f"\nSaved to: {output_path}")
        return "\n".join(log_lines), str(output_path)

    except gr.Error:
        raise
    except FileNotFoundError:
        return (
            "Error: 'ctq' command not found. Make sure convert_to_quant is installed "
            "in this environment (pip install convert_to_quant) and the venv is active.",
            None,
        )
    except Exception as e:
        return f"Error: {e}\n\n{traceback.format_exc()}", None


def list_presets():
    try:
        result = subprocess.run(
            ["ctq", "--help-filters"], capture_output=True, text=True, timeout=30
        )
        return (result.stdout or "") + (result.stderr or "")
    except FileNotFoundError:
        return "Error: 'ctq' command not found. Install convert_to_quant first."
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Krea2 INT8 ConvRot Converter") as demo:
    gr.Markdown("# Krea2 Checkpoint: INT8 ConvRot Converter")
    gr.Markdown(
        "**If your source checkpoint is already FP8** (e.g. a LoRA merged into an "
        "FP8 base), start on the **Dequantize** tab first. Re-quantizing raw FP8 "
        "values straight to INT8 can produce total noise if the FP8 weights are "
        "stored with a separate per-layer scale tensor that the quantizer doesn't "
        "know about. Dequantizing to BF16 first (using that embedded scale) fixes "
        "this before ConvRot ever runs.\n\n"
        "**Note:** as of this writing, ComfyUI's native diffusion-model loader does "
        "not reliably run Krea2 ConvRot INT8 checkpoints — use a ConvRot-aware "
        "custom node (e.g. ComfyUI-INT8-Fast / its SparknightLLC fork, or "
        "WINT8-ComfyUI), and set that node's `model_type` explicitly to `krea2` "
        "rather than `auto`."
    )

    with gr.Tabs():
        with gr.Tab("1. Dequantize FP8 -> BF16 (do this first if source is FP8)"):
            with gr.Row():
                with gr.Column():
                    deq_file_input = gr.File(label="Upload checkpoint (.safetensors)", file_types=[".safetensors"])
                    deq_path_input = gr.Textbox(
                        label="Or local file path (recommended)",
                        placeholder=r"C:\models\Soliloquy.safetensors",
                    )
                    deq_output_name = gr.Textbox(
                        label="Output filename (optional)",
                        placeholder="Soliloquy_bf16.safetensors",
                    )
                    with gr.Row():
                        scan_scale_btn = gr.Button("Scan: scaled FP8 or naive FP8?")
                        dequantize_btn = gr.Button("Dequantize to BF16", variant="primary")
                with gr.Column():
                    deq_log_output = gr.Textbox(label="Log", lines=20)
                    deq_file_output = gr.File(label="Dequantized (BF16) checkpoint")

            scan_scale_btn.click(
                scan_scale_metadata_ui,
                inputs=[deq_file_input, deq_path_input],
                outputs=[deq_log_output],
            )
            dequantize_btn.click(
                run_dequantize,
                inputs=[deq_file_input, deq_path_input, deq_output_name],
                outputs=[deq_log_output, deq_file_output],
            )

        with gr.Tab("2. INT8 ConvRot Conversion"):
            gr.Markdown(
                "Wraps the `ctq` CLI from "
                "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant). "
                "Point this at your BF16 file from step 1 (or an already-unquantized "
                "checkpoint)."
            )
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload checkpoint (.safetensors)", file_types=[".safetensors"])
                    path_input = gr.Textbox(
                        label="Or local file path (recommended for large checkpoints)",
                        placeholder=r"C:\models\Soliloquy_bf16.safetensors",
                    )
                    output_name = gr.Textbox(
                        label="Output filename (optional)",
                        placeholder="krea2_int8_convrot.safetensors",
                    )

                    use_int8 = gr.Checkbox(label="INT8 quantize (--int8)", value=True)
                    scaling_mode = gr.Radio(["row", "tensor", "block"], value="row", label="Scaling mode")
                    block_size = gr.Number(label="Block size (only used if scaling mode = block)", value=128, precision=0)

                    use_simple = gr.Checkbox(
                        label="Simple mode / skip learned rounding (--simple) — much faster, slightly lower fidelity",
                        value=True,
                    )
                    use_low_memory = gr.Checkbox(label="Low memory mode (--low-memory)", value=True)

                    use_convrot = gr.Checkbox(label="Apply ConvRot (Hadamard rotation)", value=True)
                    convrot_group_size = gr.Radio(["64", "256", "1024"], value="256", label="ConvRot group size")

                    use_comfy_quant = gr.Checkbox(label="ComfyUI-compatible metadata (--comfy_quant)", value=True)
                    save_metadata = gr.Checkbox(label="Save quant metadata (--save-quant-metadata)", value=True)

                    preset = gr.Dropdown(PRESET_CHOICES, value="(none)", label="Architecture preset (layer exclusions)")
                    custom_preset_flag = gr.Textbox(
                        label="Custom preset flag (check 'List available presets' for your ctq version)",
                        placeholder="e.g. krea2, zimage",
                    )
                    exclude_layers = gr.Textbox(
                        label="Exclude layers (regex, --exclude-layers) — leave unquantized (defaults to Krea2 K2 profile)",
                        value=KREA2_K2_DEFAULT_PATTERN,
                        placeholder=r"diffusion_model\.first\.",
                    )
                    scan_fp8_bias_btn = gr.Button("Scan checkpoint for FP8 biases (merge into exclude pattern)")
                    extra_flags = gr.Textbox(label="Extra raw CLI flags (advanced)", placeholder="--calib_samples 8192")

                    with gr.Row():
                        convert_btn = gr.Button("Convert", variant="primary")
                        list_presets_btn = gr.Button("List available presets")

                with gr.Column():
                    log_output = gr.Textbox(label="ctq output log", lines=20)
                    file_output = gr.File(label="Converted checkpoint")

            convert_btn.click(
                run_conversion,
                inputs=[file_input, path_input, output_name, use_int8, scaling_mode, block_size,
                        use_convrot, convrot_group_size, use_comfy_quant, save_metadata,
                        preset, custom_preset_flag, extra_flags, use_simple, use_low_memory,
                        exclude_layers],
                outputs=[log_output, file_output],
            )
            list_presets_btn.click(list_presets, inputs=[], outputs=[log_output])
            scan_fp8_bias_btn.click(
                scan_for_fp8_biases,
                inputs=[file_input, path_input, exclude_layers],
                outputs=[log_output, exclude_layers],
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
