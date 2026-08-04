# StormForgeAI Quant Factory


A Gradio front end with two steps:
1. **Dequantize FP8 -> BF16** — required if your source checkpoint is already FP8.
2. **INT8 ConvRot Conversion** — wraps `ctq` (from silveroxides/convert_to_quant),
   quantizing a safetensors checkpoint to INT8 with an optional Hadamard rotation
   (ConvRot) applied first to reduce outliers vs. plain INT8.

## Setup
1. Run `install.bat` (creates a venv, installs PyTorch + Triton + convert_to_quant + gradio + scipy).
   Edit the PyTorch line in `install.bat` if cu128 isn't right for your GPU.
2. Run `run.bat` to launch the app in your browser.

## Is your source checkpoint already FP8? Read this first.

**If your checkpoint was ever loaded/saved through ComfyUI as FP8 (e.g. a merge
made via `UNETLoader` with `weight_dtype: fp8_e4m3fn_fast`, or downloaded as an
`*_fp8*.safetensors` file), start on the Dequantize tab — don't feed it straight
into ConvRot.**

Many FP8 checkpoints store weights as `raw_fp8_value` plus a
**separate per-layer `.weight_scale` tensor**, where the real value is
`raw_fp8_value.float() * weight_scale`. If a quantizer doesn't know about that
scale tensor and treats the raw FP8 bits as the final weight, every value it
re-quantizes is off by whatever that scale factor was — across potentially
hundreds of layers at once. This doesn't produce mildly-worse output; it
produces **total noise**, because the whole model's weight magnitudes are
systematically wrong before ConvRot ever runs.

Workflow:
1. **Dequantize tab** -> click "Scan: scaled FP8 or naive FP8?" first. It tells
   you exactly how many weight tensors have a companion scale tensor.
2. Click "Dequantize to BF16". This reconstructs real values using each
   tensor's own embedded scale (or just upcasts, for the handful of layers
   that don't have one — commonly the sensitive layers noted below).
3. Check the printed sample stats (abs-mean/abs-max) — these should look like
   ordinary weight magnitudes, not near-zero or absurdly large.
4. Take that `_bf16.safetensors` output into the **ConvRot tab** and convert
   that instead of the original FP8 file.

If your source checkpoint is already BF16/FP16/FP32 (never quantized), skip
straight to the ConvRot tab.

## ConvRot tab usage
- Prefer the local file path field over uploading. Type the real path to your
  checkpoint (e.g. `C:\models\Soliloquy_bf16.safetensors`) rather than using
  the upload button — avoids a Windows file-locking issue with Gradio's
  temp-upload handling (see Troubleshooting) and skips re-uploading a
  multi-GB file through the browser.
- Leave ConvRot on for the outlier-suppressed quantization; turn it off for plain INT8.
- ConvRot group size: 256 is the recommended default; pick a size that evenly
  divides the model's layer dimensions (64/256/1024 are supported).
- "List available presets" runs `ctq --help-filters` so you can see which
  architecture-exclusion flags your installed version supports (e.g. --flux2, --wan).
  There's no confirmed built-in Krea2 preset as of this writing.
- Leave "Simple mode" and "Low memory mode" on unless you specifically want the
  much slower SVD-based learned-rounding optimization.
- **Exclude-layers field defaults to the Krea2 "K2" profile**
  (`first, last, tmlp, tproj, txtfusion, txtmlp`) — these layers are documented
  as needing to stay unquantized for Krea2. Leave this as-is unless you know
  your checkpoint uses different layer names. This is a separate concern from
  the FP8 dequantization above — needed even on a clean BF16 source.
- "Scan checkpoint for FP8 biases" merges any additional FP8-bias layers it
  finds into the K2 profile rather than replacing it — safe to run before
  every conversion, even on an already-dequantized file.

## Which output should you actually use?
**BF16 will always give the best quality** — it reconstructs your source as
accurately as possible and runs at full precision from there. INT8 ConvRot
quantizes that BF16 result a second time; ConvRot minimizes the damage from
that second pass but can't fully undo it. Use INT8 ConvRot only when you need
the smaller file size / faster inference and are willing to trade some
fidelity for it.

## Loading the outputs in ComfyUI
- **BF16 output**: load like any normal checkpoint with the standard
  "Load Diffusion Model" node, `weight_dtype: default`.
- **INT8 ConvRot output**: as of this writing, ComfyUI's *native* loader does
  not reliably run this combination (INT8 + ConvRot specifically — its native
  fast paths cover plain INT8 and 4-bit ConvRot, but not 8-bit ConvRot together).
  Use a ConvRot-aware custom node instead — ComfyUI-INT8-Fast (or its
  SparknightLLC fork) or WINT8-ComfyUI — and set that node's `model_type`
  explicitly to `krea2` rather than `auto`.

## Troubleshooting

**`PermissionError` when clicking Scan (file in `AppData\Local\Temp\gradio\...`)**
Windows briefly locks Gradio's temp copy of an uploaded file. Type the real path
to your checkpoint into the local file path field instead of using the upload
button, and click Scan again.

**`ModuleNotFoundError: No module named 'scipy'`**
ConvRot needs scipy, which isn't pulled in automatically by `convert_to_quant`.
`scipy` is in `requirements.txt`; if you hit this on an older install, run
`pip install scipy` inside the venv.

**`NotImplementedError: "sum_cpu" not implemented for 'Float8_e4m3fn'`**
Upstream `ctq` bug: bias-correction calls `.mean()` on a bias tensor still
stored in FP8, and PyTorch has no CPU reduction kernel for that dtype. Fixes,
in order:
1. `pip install -U convert_to_quant` — may already be patched upstream.
2. Dequantize to BF16 first (see above) — removes FP8 tensors from the input
   entirely, so this can't trigger.
3. If you still hit it, use "Scan checkpoint for FP8 biases" to auto-merge
   the affected layer(s) into the exclude-layers pattern.

**`KeyError` on a `*.scale` tensor when loading in ComfyUI (e.g.
`'txtfusion.layerwise_blocks.0.prenorm.scale'`)**
This means a real model weight (a norm layer's learnable scale/gain
parameter, not a quantization scale) was dropped during dequantization. Make
sure you're on the current `app.py` — earlier versions had a bug that matched
any tensor ending in `.scale`/`_scale` as a quantization-scale companion, which
could delete legitimate weights sharing that naming pattern. Re-run Dequantize
with the fixed version.

**Conversion seems stuck / very slow**
Check that PyTorch actually sees your GPU:
```
venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, ctq is running on CPU. Also make sure "Simple mode" (`--simple`) is
checked — without it, `ctq` runs a much slower SVD-based learned-rounding
optimization per tensor.

