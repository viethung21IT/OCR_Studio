# Script khoi dong nhanh AIC OCR Studio Web Demo
$ErrorActionPreference = "Stop"

$PYTHON_CMD = "python"
if (Test-Path "G:\miniconda3\envs\CV_env\python.exe") {
    $PYTHON_CMD = "G:\miniconda3\envs\CV_env\python.exe"
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  AIC OCR Studio Web Demo" -ForegroundColor Green
Write-Host "  Python:    $PYTHON_CMD" -ForegroundColor DarkCyan
Write-Host "  Dia chi:   http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Cyan

& $PYTHON_CMD -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
