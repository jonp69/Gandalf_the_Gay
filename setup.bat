@echo off
setlocal enabledelayedexpansion

echo ================================================
echo [1/4] Checking Python environment...
echo ================================================

:: Check if requirements.txt contains LLM/ML packages
set "USE_PYTHON311=false"
set "LLM_PACKAGES=transformers torch tensorflow langchain openai llama-cpp-python sentence-transformers diffusers accelerate bitsandbytes"

if exist requirements.txt (
    for /f "tokens=*" %%i in (requirements.txt) do (
        set "line=%%i"
        :: Skip comments and empty lines
        if not "!line:~0,1!"=="#" if not "!line!"=="" (
            for %%p in (%LLM_PACKAGES%) do (
                echo !line! | findstr /i "%%p" >nul
                if !errorlevel! equ 0 (
                    set "USE_PYTHON311=true"
                    echo [INFO] Found LLM package: %%p in requirements.txt
                    goto :found_llm
                )
            )
        )
    )
    :found_llm
) else (
    echo ERROR: requirements.txt not found!
    pause
    exit /b 1
)

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
if exist "venv\" (
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
    
    :: Check if packages are installed by trying to import key ones
    python -c "import cv2, PIL, numpy, torch, transformers" 2>nul
    if %errorlevel% neq 0 (
        echo [INFO] Some packages missing, installing from requirements.txt...
        python -m pip install --upgrade pip
        if %errorlevel% neq 0 (
            echo ERROR: Failed to upgrade pip
            pause
            exit /b 1
        )
        
        pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install packages from requirements.txt
            pause
            exit /b 1
        )
    ) else (
        echo [INFO] All key packages verified successfully
    )
    
) else (
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
    echo [2/4] Installing packages from requirements.txt...
    echo ================================================
    
    python -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        echo ERROR: Failed to upgrade pip
        pause
        exit /b 1
    )
    
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install packages from requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo ================================================
echo [3/4] Running Video Meme Compositor...
echo ================================================
echo [INFO] Virtual environment is ready!
echo [INFO] Python interpreter: %PYTHON_CMD%
echo [INFO] All dependencies installed successfully.
echo.
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