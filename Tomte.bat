@echo off
rem Opens Tomte. You can also drop FBX, texture and .tomte setup files onto this file.
cd /d "%~dp0"

python -c "import PySide6, numpy, PIL" 2>nul
if not errorlevel 1 goto run

echo Tomte needs Python libraries that aren't installed yet:
echo     numpy, Pillow, PySide6 and opencv-python-headless - see requirements.txt
choice /M "Install them now with pip"
if errorlevel 2 exit /b 1
python -m pip install -r requirements.txt
if not errorlevel 1 goto run
echo.
echo Couldn't install them. Is Python installed and on your PATH?
pause
exit /b 1

:run
start "" pythonw -m uvtransfer.gui %*
