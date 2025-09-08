@echo off
REM Setup script for Windows users
echo 🚀 Setting up Female CEO ROA Analysis
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Run the setup script
python setup.py

echo.
echo Setup complete! Press any key to exit...
pause >nul
