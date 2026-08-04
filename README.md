# Krea2 INT8 ConvRot Converter

A Gradio front end for `ctq` (from silveroxides/convert_to_quant), which quantizes
safetensors checkpoints to INT8 with an optional Hadamard rotation (ConvRot) applied
before quantization to reduce outliers and improve accuracy vs. plain INT8.

## Setup
1. Run `install.bat` (creates a venv, installs PyTorch + Triton + convert_to_quant + gradio + scipy).
   Edit the PyTorch line in `install.bat` if cu128 isn't right for your GPU.
2. Run `run.bat` to launch the app in your browser.

## Usage
- Prefer the local file path field over uploading. Type the real path to your
  checkpoint (e.g. `C:\models\Soliloquy.safetensors`) rather than using the upload
  button — this avoids a Windows file-locking issue with Gradio's temp-upload
  handling (see Troubleshooting) and skips re-uploading a multi-GB file through
  the browser.
- Leave ConvRot on for the outlier-suppressed quantization; turn it off for plain INT8.
- ConvRot group size: 256 is the recommended default; pick a size that evenly divides
  the model's layer dimensions (64/256/1024 are supported).
- "List available presets" runs `ctq --help-filters` so you can see which
  architecture-exclusion flags your installed version supports (e.g. --flux2, --wan).
  There's no confirmed built-in Krea2 preset as of this writing — use the custom
  preset field if one exists for your ctq version, or leave presets off.
- Leave "Simple mode" and "Low memory mode" on unless you specifically want the
  slower SVD-based learned-rounding optimization (much higher fidelity, much slower,
  especially without a fast GPU).
- Before converting, click "Scan checkpoint for FP8 biases" — it reads just the
  safetensors header (instant) and auto-fills the exclude-layers field with every
  layer whose bias is stored in FP8, working around a known `ctq` bug (see
  Troubleshooting). Do this before every new checkpoint you convert.

## Troubleshooting

**`PermissionError` when clicking Scan (file in `AppData\Local\Temp\gradio\...`)**
Windows briefly locks Gradio's temp copy of an uploaded file. Type the real path
to your checkpoint into the local file path field instead of using the upload
button, and click Scan again.

**`ModuleNotFoundError: No module named 'scipy'`**
ConvRot needs scipy, which isn't pulled in automatically by `convert_to_quant`.
`scipy` is in `requirements.txt` now; if you hit this on an older install, run
`pip install scipy` inside the venv.

**`NotImplementedError: "sum_cpu" not implemented for 'Float8_e4m3fn'`**
This is an upstream `ctq` bug: bias-correction calls `.mean()` on a bias tensor
still stored in FP8, and PyTorch has no CPU reduction kernel for that dtype. It
can hit any layer whose bias happens to be FP8, not just one specific layer.
Fixes, in order:
1. `pip install -U convert_to_quant` — may already be patched upstream.
2. Click "Scan checkpoint for FP8 biases" before converting to auto-generate an
   `--exclude-layers` pattern covering every affected layer at once, rather than
   discovering them one crash at a time. Excluded layers are left at original
   precision instead of being quantized to INT8.

**Conversion seems stuck / very slow**
Check that PyTorch actually sees your GPU:
```
venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, ctq is running on CPU. Also make sure "Simple mode" (`--simple`) is
checked — without it, `ctq` runs a much slower SVD-based learned-rounding
optimization per tensor.

## Important compatibility note
As of this writing, ComfyUI's stock "Load Diffusion Model" node loads plain INT8
Krea2 checkpoints but NOT ConvRot INT8 ones. To actually run a ConvRot-quantized
checkpoint, you need a ConvRot-aware custom node, e.g.:
- ComfyUI-INT8-Fast (BobJohnson24)
- WINT8-ComfyUI (Winnougan)

Check those repos for current compatibility before committing to ConvRot output.
