@echo off
title G-TRADE GPU SETUP
color 0E
cls

echo =============================================================
echo    G-TRADE GPU SETUP: Auto-Detect + Install
echo =============================================================
echo.
echo  This script will:
echo    1. Create conda environment gtrade_gpu (Python 3.10)
echo    2. Detect your GPU and install matching CUDA/cuDNN
echo    3. Install TensorFlow with GPU support
echo    4. Install all project dependencies
echo    5. Verify GPU acceleration works
echo.
echo  Supported GPUs:
echo    - NVIDIA: RTX 20xx/30xx/40xx, GTX 16xx, A100, etc.
echo    - No GPU: falls back to CPU (slower but functional)
echo.
echo =============================================================
pause

echo.
echo [1/5] Creating conda environment gtrade_gpu (Python 3.10)...
echo -------------------------------------------------------
call conda create -n gtrade_gpu python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create environment!
    pause
    exit /b 1
)

echo.
echo [2/5] Detecting GPU...
echo -------------------------------------------------------
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    set HAS_GPU=1
) else (
    echo [INFO] No NVIDIA GPU detected. Will install CPU-only TensorFlow.
    set HAS_GPU=0
)

echo.
echo [3/5] Installing CUDA toolkit + cuDNN...
echo -------------------------------------------------------
if "%HAS_GPU%"=="1" (
    REM CUDA 11.8 works with TF 2.10-2.15 and supports RTX 20xx-40xx
    call conda install -n gtrade_gpu -c conda-forge cudatoolkit=11.8 cudnn=8.6 -y
    if errorlevel 1 (
        echo [WARN] conda-forge failed, trying CUDA 11.2 as fallback...
        call conda install -n gtrade_gpu -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
    )
) else (
    echo [SKIP] No GPU — skipping CUDA installation.
)

echo.
echo [4/5] Installing TensorFlow + project dependencies...
echo -------------------------------------------------------
if "%HAS_GPU%"=="1" (
    call conda run -n gtrade_gpu pip install "tensorflow>=2.10,<2.16" --no-cache-dir
) else (
    call conda run -n gtrade_gpu pip install "tensorflow-cpu>=2.10" --no-cache-dir
)
if errorlevel 1 (
    echo [ERROR] TensorFlow installation failed!
    pause
    exit /b 1
)

call conda run -n gtrade_gpu pip install -r requirements.txt --no-cache-dir

echo.
echo [5/5] Verifying GPU setup...
echo =============================================================
call conda run -n gtrade_gpu python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(); print(f'  TensorFlow {tf.__version__}'); print(f'  GPU devices: {len(gpus)}'); [print(f'    - {g.name}') for g in gpus]; print(f'  Status: {\"GPU READY\" if gpus else \"CPU ONLY\"}')"

echo.
echo =============================================================
echo    SETUP COMPLETE
echo =============================================================
echo.
echo  Environment: gtrade_gpu
echo  Activate:    conda activate gtrade_gpu
echo  Run:         run_gtrade.bat or gtrade_gui.bat
echo.
if "%HAS_GPU%"=="0" (
    echo  NOTE: Running on CPU. Training will be slower but functional.
    echo  To enable GPU, install an NVIDIA GPU with CUDA support.
    echo.
)
pause
