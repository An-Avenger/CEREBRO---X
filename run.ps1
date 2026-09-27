# Cerebro X Unified Startup Script
$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "      Starting Cerebro X Platform      " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Start Backend API
Write-Host "[1/3] Launching FastAPI Backend on Port 8000..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath "python" -ArgumentList "scripts/run_api.py" -PassThru -WindowStyle Minimized
Write-Host "Backend API process started (PID: $($backendProcess.Id))." -ForegroundColor Green

# Wait for backend to initialize (models take a few seconds to load)
Write-Host "Waiting 15 seconds for AI models to load into memory..." -ForegroundColor Gray
Start-Sleep -Seconds 15

# Start Frontend UI
Write-Host "[2/3] Launching Frontend UI on Port 3000..." -ForegroundColor Yellow
Set-Location -Path "frontend"
$frontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -PassThru -WindowStyle Minimized
Set-Location -Path ".."
Write-Host "Frontend server started (PID: $($frontendProcess.Id))." -ForegroundColor Green

# Save PIDs for cleanup
$backendProcess.Id | Out-File -FilePath ".backend_pid" -Encoding ASCII
$frontendProcess.Id | Out-File -FilePath ".frontend_pid" -Encoding ASCII

# Open Browser
Write-Host "[3/3] Opening Web Dashboard..." -ForegroundColor Yellow
Start-Process "http://localhost:3000"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " System is LIVE. Close this window to keep running." -ForegroundColor Cyan
Write-Host " To shut down, run: .\stop.ps1" -ForegroundColor Red
Write-Host "=========================================" -ForegroundColor Cyan
