# CEREBRO-X PROJECT EXECUTION GUIDE

This guide provides the exact commands and troubleshooting steps to run the Cerebro-X project on a local machine.

## Prerequisites
- Windows OS (PowerShell)
- Python 3.10+
- Node.js 18+ (with npm)
- Required Python packages (see `requirements.txt`)

## Quick Start (Automated)

The easiest way to start both the backend and frontend is using the unified PowerShell script.

1. Open PowerShell as Administrator (or standard user if execution policies allow).
2. Navigate to the project root directory:
   ```powershell
   cd e:\PROJECTS\CEREBRO-X
   ```
3. Run the startup script:
   ```powershell
   .\run.ps1
   ```
   *This script will:*
   - Launch FastAPI on port 8000 in a minimized window.
   - Wait 15 seconds for ML models to load into memory.
   - Launch Next.js on port 3000 in a minimized window.
   - Open your default web browser to `http://localhost:3000`.

## Manual Start (If run.ps1 fails)

If the automated script fails or you want to see the live logs for debugging, run the services separately.

### 1. Start the Backend (Terminal 1)
```powershell
cd e:\PROJECTS\CEREBRO-X
python scripts/run_api.py
```
Wait until you see: `Uvicorn running on http://127.0.0.1:8000`

### 2. Start the Frontend (Terminal 2)
```powershell
cd e:\PROJECTS\CEREBRO-X\frontend
npm run dev
```
Wait until you see: `✓ Ready in XXs`

### 3. Open the Browser
Navigate to [http://localhost:3000](http://localhost:3000)

## Shutting Down

To cleanly shut down the background processes created by `run.ps1`:
```powershell
.\stop.ps1
```
This script reads the `.backend_pid` and `.frontend_pid` files and forcefully terminates the Node and Python processes.

## Localhost URLs
- **Main Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Demo Walkthrough**: [http://localhost:3000/demo](http://localhost:3000/demo)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

## Running Tests

To verify that the project and the new MRI pipeline are functioning correctly, run the PyTest suite:
```powershell
cd e:\PROJECTS\CEREBRO-X
python -m pytest tests/ -v
```

## Troubleshooting

### Issue: "Connection Refused" on localhost:3000
**Cause**: The Next.js frontend has not finished compiling yet.
**Fix**: Wait 15-30 seconds and refresh the browser.

### Issue: Port 8000 is already in use
**Cause**: A previous instance of the backend crashed or was not properly shut down.
**Fix**: Run `.\stop.ps1`. If that doesn't work, manually kill the Python process in Task Manager.

### Issue: "CNN3D Checkpoint Not Available" during MRI upload
**Cause**: The raw MRI CNN model has not been trained because the OASIS-2 `.nii` files are not bundled in the repo.
**Fix**: This is expected behavior. The pipeline successfully preprocesses the MRI, but inference is blocked. To fix, you must download the dataset and run `scripts/run_phase3_mri.py`.
