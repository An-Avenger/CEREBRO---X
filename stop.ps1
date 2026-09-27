# Cerebro X Cleanup Script
$ErrorActionPreference = "SilentlyContinue"

Write-Host "Shutting down Cerebro X Platform..." -ForegroundColor Yellow

if (Test-Path ".backend_pid") {
    $backendPid = Get-Content ".backend_pid"
    Stop-Process -Id $backendPid -Force
    Remove-Item ".backend_pid"
    Write-Host "Backend API (PID: $backendPid) stopped." -ForegroundColor Green
} else {
    Write-Host "No backend process found to stop." -ForegroundColor Gray
}

if (Test-Path ".frontend_pid") {
    $frontendPid = Get-Content ".frontend_pid"
    Stop-Process -Id $frontendPid -Force
    Remove-Item ".frontend_pid"
    Write-Host "Frontend server (PID: $frontendPid) stopped." -ForegroundColor Green
} else {
    Write-Host "No frontend process found to stop." -ForegroundColor Gray
}

Write-Host "Shutdown complete." -ForegroundColor Cyan
