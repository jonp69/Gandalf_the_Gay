@echo off
cd /d "%~dp0"

echo ================================================
echo Video Meme Compositor Setup
echo ================================================

echo [INFO] Checking Python environment...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.11
    pause
    exit /b 1
)

echo [INFO] Checking CUDA availability...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
) else (
    echo [WARNING] No NVIDIA GPU or drivers detected - will install CPU version
)

echo ================================================
echo [1/2] Creating virtual environment...
echo ================================================

if exist "venv" (
    echo [INFO] Virtual environment already exists, recreating...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo [INFO] Virtual environment activated: %VIRTUAL_ENV%
python -c "import sys; print('Python executable:', sys.executable)"

echo ================================================
echo [2/2] Installing packages...
echo ================================================

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing compatible PyTorch ecosystem with CUDA support...
python -m pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo [WARNING] CUDA 11.8 failed, trying CPU version...
    python -m pip install --no-cache-dir torch torchvision
)

echo [INFO] Verifying PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo [INFO] Installing xformers for faster inference...
python -m pip install --no-cache-dir xformers
if %errorlevel% neq 0 (
    echo [WARNING] xformers installation failed, will use slower inference
)

echo [INFO] Installing remaining packages...
python -m pip install --no-cache-dir -r requirements_simple.txt

echo ================================================
echo Setup Complete!
echo ================================================

echo [INFO] Running Video Meme Compositor...
python main.py

echo ================================================
echo Application finished
echo ================================================
pause