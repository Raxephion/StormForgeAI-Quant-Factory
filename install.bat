@echo off
setlocal
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch (CUDA 12.8 build)...
echo If this is the wrong CUDA version for your GPU, edit this line -
echo see https://pytorch.org/get-started/locally/ for the correct command.
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo Installing Triton for Windows (needed for INT8 kernels)...
pip install -U "triton-windows<3.6"

echo Installing convert_to_quant, gradio, and remaining requirements...
pip install -r requirements.txt

echo.
echo Installation complete. Run run.bat to start the app.
pause
