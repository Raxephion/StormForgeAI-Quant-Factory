import json
import re
import shlex
import struct
import subprocess
import traceback
from pathlib import Path

import gradio as gr

PRESET_CHOICES = [
    "(none)",
    "flux2",
    "t5xxl",
    "hunyuan",
    "wan",
    "custom (type below)",
]

FP8_DTYPE_NAMES = {"F8_E4M3", "F8_E5M2"}

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
        "Wraps the `ctq` CLI from "
        "[silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) "
        "to quantize a `.safetensors` checkpoint to INT8, with an optional group-wise "
        "Hadamard rotation (**ConvRot**) applied before quantization to suppress "
        "weight outliers and improve accuracy versus plain INT8.\n\n"
        "**Note:** as of this writing, ComfyUI's native diffusion-model loader does "
        "not reliably run Krea2 ConvRot INT8 checkpoints — use a ConvRot-aware "
        "custom node (e.g. ComfyUI-INT8-Fast / its SparknightLLC fork, or "
        "WINT8-ComfyUI), and set that node's `model_type` explicitly to `krea2` "
        "rather than `auto`.\n\n"
        "The exclude-layers field below defaults to the documented Krea2 'K2' "
        "profile (`first, last, tmlp, tproj, txtfusion, txtmlp` kept unquantized) "
        "— clear it only if you know your checkpoint uses different layer names."
    )

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload checkpoint (.safetensors)", file_types=[".safetensors"])
            path_input = gr.Textbox(
                label="Or local file path (recommended for large checkpoints)",
                placeholder=r"C:\models\krea2_fp8.safetensors",
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
