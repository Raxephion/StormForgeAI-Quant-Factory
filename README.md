# StormForgeAI Quant Factory

> A simple, local Gradio workbench for checkpoint quantization, currently focused on INT8 and ConvRot conversion of `.safetensors` diffusion models.

![Status](https://img.shields.io/badge/status-early%20development-orange)
![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Interface](https://img.shields.io/badge/interface-Gradio-ff7c00)
![License](https://img.shields.io/badge/license-see%20repository-lightgrey)

## Overview

**StormForgeAI Quant Factory** provides a clean graphical interface around the [`ctq`](https://github.com/silveroxides/convert_to_quant) command-line tool from `silveroxides/convert_to_quant`.

The project is intended to make checkpoint quantization easier and more approachable for users who do not want to assemble commands manually or build a dedicated ComfyUI workflow. It runs locally, accepts `.safetensors` checkpoints, exposes the important conversion controls, streams the conversion log, and returns the completed checkpoint through the interface.

The current release is centered on **INT8 quantization**, with optional **ConvRot Hadamard rotation** to reduce weight outliers before quantization. Krea2 is the primary tested use case, but the interface also exposes architecture presets and advanced flags supported by the installed `ctq` version.

> [!IMPORTANT]
> StormForgeAI Quant Factory is in **early and active development**. The present INT8/ConvRot workflow is the first stage of a broader quantization toolkit. Additional quantization methods, model-family support, validation tools, presets, and workflow improvements are planned as development continues.

## Current Capabilities

- Convert local `.safetensors` checkpoints to **INT8**.
- Apply optional **ConvRot** group-wise Hadamard rotation before quantization.
- Choose between **row**, **tensor**, and **block** scaling modes.
- Configure block size when using block scaling.
- Select ConvRot group sizes of **64**, **256**, or **1024**.
- Use **Simple mode** for faster conversion without learned rounding.
- Enable **Low Memory mode** for systems with limited RAM or VRAM.
- Write **ComfyUI-compatible quantization metadata**.
- Preserve additional quantization metadata in the output checkpoint.
- Use built-in architecture filters exposed by `ctq`, including Flux 2, T5-XXL, Hunyuan, and WAN where supported by the installed version.
- Supply a custom architecture-filter flag.
- Pass advanced raw CLI options directly to `ctq`.
- Define regular-expression-based layer exclusions.
- Scan a checkpoint header for FP8 bias tensors and merge affected layers into the exclusion pattern.
- Apply the included Krea2 K2 exclusion profile by default.
- View the full command and live conversion output in the interface.
- Download the completed checkpoint from the app after conversion.

## Why ConvRot?

Plain INT8 conversion can be affected by large weight outliers. ConvRot applies a group-wise Hadamard rotation before quantization, redistributing those outliers and potentially preserving more model quality than straightforward INT8 conversion.

ConvRot support at inference time depends on the model loader being used. A ConvRot-quantized checkpoint generally requires a ConvRot-aware loader or custom node rather than a standard diffusion-model loader.

## Easy Windows Setup

The repository includes two batch files to keep setup and startup straightforward.

### Requirements

- Windows 10 or Windows 11
- Python available from the command line
- An NVIDIA GPU is recommended
- Sufficient free storage for the source and converted checkpoints

### Install

1. Download or clone the repository.
2. Open the project folder.
3. Run:

```bat
install.bat
```

The installer will:

- Create a local Python virtual environment in `venv`.
- Upgrade `pip`.
- Install the CUDA 12.8 build of PyTorch.
- Install Triton for Windows.
- Install Gradio, `convert_to_quant`, Safetensors, SciPy, and the remaining Python requirements.

> [!NOTE]
> The included installer uses the PyTorch CUDA 12.8 package index. Edit the PyTorch installation line in `install.bat` when a different build is required for your hardware or driver setup.

### Run

After installation, launch the app with:

```bat
run.bat
```

The batch file activates the local virtual environment and starts the Gradio interface in your browser.

## Recommended Workflow

1. Launch the app with `run.bat`.
2. Enter the full local path to the source checkpoint.
3. Click **Scan checkpoint for FP8 biases**.
4. Review the generated exclusion pattern.
5. Choose the quantization and ConvRot settings.
6. Enter an optional output filename.
7. Click **Convert**.
8. Follow the conversion output in the log panel.
9. Retrieve the completed checkpoint from the output panel or the local `outputs` folder.

For very large checkpoints, the local path field is recommended instead of browser upload. This avoids copying a multi-gigabyte model into Gradio's temporary upload directory and can prevent Windows file-locking issues.

## Default Krea2 Profile

The current interface includes a default Krea2 K2 exclusion profile that keeps the following layer groups at their original precision:

```text
first
last
tmlp
tproj
txtfusion
txtmlp
```

The FP8-bias scanner reads only the Safetensors header and merges any additional affected layer names into the exclusion expression. It does not load the full checkpoint tensor data merely to perform the scan.

Do not clear or replace the exclusion pattern unless you understand the layer layout of the checkpoint being converted.

## Output

Converted checkpoints are written to:

```text
outputs\
```

When no output name is supplied, the app generates one from the source checkpoint name. Depending on the selected mode, the filename receives an `_int8` or `_int8_convrot` suffix.

## Compatibility

StormForgeAI Quant Factory produces checkpoints through `convert_to_quant`; successful conversion does not guarantee that every inference loader can execute every output format.

In particular, ConvRot checkpoints require a ConvRot-aware implementation. Compatible ComfyUI options may include projects such as:

- `ComfyUI-INT8-Fast`
- The SparknightLLC fork of `ComfyUI-INT8-Fast`
- `WINT8-ComfyUI`

For Krea2, explicitly selecting `krea2` as the model type may be necessary when the loader provides that option. Check the relevant loader repository for its current model and format support.

## Troubleshooting

### `PermissionError` from a Gradio temporary path

Windows may temporarily lock a checkpoint copied into Gradio's upload directory. Use the **local file path** field instead of the browser upload control and run the scan or conversion again.

### `ModuleNotFoundError: No module named 'scipy'`

ConvRot requires SciPy. Re-run `install.bat`, or activate the environment and install it manually:

```bat
venv\Scripts\activate
pip install scipy
```

### `NotImplementedError: "sum_cpu" not implemented for 'Float8_e4m3fn'`

This can occur when bias correction reaches a bias tensor stored in FP8 and PyTorch attempts an unsupported CPU reduction.

Recommended steps:

1. Update `convert_to_quant`.
2. Scan the checkpoint for FP8 biases before conversion.
3. Keep the generated affected layers in the exclusion pattern.

```bat
venv\Scripts\activate
pip install -U convert_to_quant
```

Excluded layers remain at their original precision rather than being converted to INT8.

### Conversion is extremely slow

Confirm that PyTorch can see the GPU:

```bat
venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```

A result of `False` usually means conversion is running on the CPU. Also verify that **Simple mode** is enabled unless the slower learned-rounding process is intentional.

### The `ctq` command is not found

Re-run `install.bat`, or activate the virtual environment and reinstall the requirements:

```bat
venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap

Development will continue beyond the present INT8 and ConvRot workflow. Planned areas of expansion include:

- Additional quantization formats and methods.
- Broader diffusion-model architecture support.
- More tested presets and model-family profiles.
- Improved checkpoint inspection and preflight validation.
- Better compatibility guidance for inference loaders.
- Conversion summaries and output verification.
- More safeguards for unsupported tensor layouts and data types.
- Interface and usability improvements.
- Continued optimization for lower-memory systems.

The exact roadmap will evolve alongside the quantization ecosystem and the needs uncovered through real-world testing.

## Project Structure

```text
StormForgeAI-Quant-Factory/
├── app.py
├── install.bat
├── run.bat
├── requirements.txt
├── readme.md
├── outputs/
└── venv/                 # created during installation
```

## Dependencies and Credits

StormForgeAI Quant Factory currently builds on:

- [Gradio](https://www.gradio.app/) for the local user interface.
- [convert_to_quant](https://github.com/silveroxides/convert_to_quant) and its `ctq` CLI for checkpoint conversion.
- [PyTorch](https://pytorch.org/) for tensor operations and GPU acceleration.
- [Safetensors](https://github.com/huggingface/safetensors) for checkpoint header inspection.
- [SciPy](https://scipy.org/) for ConvRot-related operations.
- [Triton for Windows](https://github.com/woct0rdho/triton-windows) for supported accelerated kernels on Windows.

Please respect the licenses and terms of the upstream projects, the source checkpoint, and any model being converted.

## Development Status

This project should currently be treated as an evolving technical utility rather than a finished universal quantization solution. Back up important checkpoints, retain the original model, review conversion logs, and test generated outputs carefully before relying on them in a production workflow.

Bug reports, model-specific findings, compatibility notes, and reproducible test results are valuable as StormForgeAI Quant Factory continues to grow.

---

<p align="center"><strong>StormForgeAI Quant Factory</strong><br>Quantization made accessible, inspectable, and expandable.</p>
