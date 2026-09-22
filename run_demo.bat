@echo off
title AIC OCR Studio Web Demo
echo ===================================================
echo   Khoi dong AIC OCR Studio Web Demo...
echo ===================================================

:: Check for python in conda environment or system PATH
set PYTHON_CMD=python
if exist "G:\miniconda3\envs\CV_env\python.exe" (
    set PYTHON_CMD="G:\miniconda3\envs\CV_env\python.exe"
)

echo [OK] Su dung Python: %PYTHON_CMD%
echo [INFO] Mo trinh duyet tai: http://127.0.0.1:8000
echo ===================================================

%PYTHON_CMD% -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
