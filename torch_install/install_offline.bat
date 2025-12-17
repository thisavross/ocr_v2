@echo off
echo PyTorch CPU Offline Installation
echo ================================
echo.
echo Checking Python version...
python --version
echo.

echo Step 1: Installing dependencies...
echo Installing typing-extensions...
pip install typing_extensions-4.12.2-py3-none-any.whl

echo Installing numpy...
pip install numpy-*.whl

echo Installing Pillow...
pip install Pillow-*.whl

echo.
echo Step 2: Installing PyTorch packages...
echo Installing PyTorch...
pip install torch-2.5.1+cpu-cp310-cp310-win_amd64.whl

echo Installing torchvision...
pip install torchvision-0.20.1+cpu-cp310-cp310-win_amd64.whl

echo Installing torchaudio...
pip install torchaudio-2.5.1+cpu-cp310-cp310-win_amd64.whl

echo.
echo Step 3: Verifying installation...
python verify_install.py

echo.
echo Installation complete!
pause