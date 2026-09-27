# reset_docker.ps1
# Script to forcefully restart Docker Desktop and WSL to fix npipe connection issues

Write-Host "Stopping Docker Desktop..."
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue

Write-Host "Shutting down WSL..."
wsl --shutdown

Write-Host "Restarting Docker service (if running as a service)..."
Restart-Service com.docker.service -ErrorAction SilentlyContinue

Write-Host "Starting Docker Desktop..."
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

Write-Host "Waiting 30 seconds for Docker Engine to start..."
Start-Sleep -Seconds 30

Write-Host "Checking Docker status..."
docker info
