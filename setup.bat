@echo off
setlocal enabledelayedexpansion

echo ================================================
echo [1/4] Checking Python environment...
echo ================================================

:: Check if requirements.txt contains LLM/ML packages
set "USE_PYTHON311=false"

if not exist requirements.txt (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)

echo [INFO] Checking requirements.txt for AI/ML packages...
findstr /i "torch" requirements.txt >nul && set "USE_PYTHON311=true" && echo [INFO] Found torch - using Python 3.11
findstr /i "transformers" requirements.txt >nul && set "USE_PYTHON311=true" && echo [INFO] Found transformers - using Python 3.11
findstr /i "tensorflow" requirements.txt >nul && set "USE_PYTHON311=true" && echo [INFO] Found tensorflow - using Python 3.11
findstr /i "diffusers" requirements.txt >nul && set "USE_PYTHON311=true" && echo [INFO] Found diffusers - using Python 3.11

:: Select Python interpreter
if "%USE_PYTHON311%"=="true" (
    set "PYTHON_CMD=C:\Python311\python.exe"
    echo [INFO] Using Python 3.11 for LLM/ML packages
) else (
    set "PYTHON_CMD=python"
    echo [INFO] Using system Python
)

:: Verify Python is available
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python interpreter not found: %PYTHON_CMD%
    echo Please ensure Python is installed and accessible.
    pause
    exit /b 1
)

echo.

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" goto ACTIVATE_VENV

:CREATE_VENV
echo ================================================
echo [1/4] Creating virtual environment...
echo ================================================

%PYTHON_CMD% -m venv venv
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

echo ================================================
echo [2/4] Installing packages with CUDA support...
echo ================================================
echo [INFO] Virtual environment activated: %VIRTUAL_ENV%
echo [INFO] Python executable: 
python -c "import sys; print(sys.executable)"
echo.
echo [INFO] Checking CUDA availability...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
) else (
    echo [WARNING] No NVIDIA GPU or drivers detected - will install CPU version
)
echo.

call :INSTALL_DEPENDENCIES

goto AFTER_VENV

:ACTIVATE_VENV
echo ================================================
echo [1/4] Activating existing virtual environment...
echo ================================================

call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo ================================================
echo [2/4] Verifying installed packages...
echo ================================================

python -c "import cv2, PIL, numpy, torch, transformers" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Some packages missing, reinstalling dependency stack...
    call :INSTALL_DEPENDENCIES
) else (
    echo [INFO] All key packages verified successfully
    call :ENSURE_TORCH_STACK
)

goto AFTER_VENV

:AFTER_VENV

echo.
echo ================================================
echo [3/4] Running Video Meme Compositor...
echo ================================================
echo [INFO] Virtual environment is ready!
echo [INFO] Python interpreter: %PYTHON_CMD%
echo [INFO] All dependencies installed successfully.
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Press any key to run the application...
pause >nul

python main.py
set "APP_EXIT_CODE=%errorlevel%"

echo.
echo ================================================
echo [4/4] Application closed!
echo ================================================
if %APP_EXIT_CODE% neq 0 (
    echo [INFO] Application exited with code: %APP_EXIT_CODE%
) else (
    echo [INFO] Application completed successfully.
)
echo.
echo Press any key to exit...
pause >nul

deactivate

goto :EOF

:INSTALL_DEPENDENCIES
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

call :INSTALL_TORCH
call :INSTALL_XFORMERS

echo [INFO] Installing remaining packages (forcing no cache for clean install)...
python -m pip install --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages from requirements.txt
    pause
    exit /b 1
)

call :INSTALL_TORCH
call :INSTALL_XFORMERS
exit /b 0

:ENSURE_TORCH_STACK
call :INSTALL_TORCH
call :INSTALL_XFORMERS
exit /b 0

:INSTALL_TORCH
echo [INFO] Installing compatible PyTorch ecosystem with CUDA 12.1 support...
python -m pip install --no-cache-dir --upgrade --force-reinstall torch==2.7.0+cu121 torchvision==0.22.0+cu121 --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo WARNING: CUDA 12.1 installation failed, trying CUDA 11.8...
    python -m pip install --no-cache-dir --upgrade --force-reinstall torch==2.7.0+cu118 torchvision==0.22.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    if %errorlevel% neq 0 (
        echo WARNING: CUDA installation failed, falling back to CPU version
        python -m pip install --no-cache-dir --upgrade --force-reinstall torch==2.7.0 torchvision==0.22.0
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install PyTorch
            pause
            exit /b 1
        )
    )
)

echo [INFO] Verifying PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
exit /b 0

:INSTALL_XFORMERS
echo [INFO] Installing xformers compatible with torch 2.5.1...
python -m pip install --no-cache-dir xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo WARNING: xformers installation failed, will use slower inference
    exit /b 0
)
exit /b 0