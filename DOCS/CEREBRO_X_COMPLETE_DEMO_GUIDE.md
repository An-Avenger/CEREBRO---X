# CEREBRO-X COMPLETE DEMO GUIDE

## 1. What Cerebro-X is
Cerebro-X is an AI-powered Digital Brain Twin framework designed to predict the progression of Alzheimer’s Disease using longitudinal clinical data and multimodal (MRI, EEG) features.

## 2. What problem it solves
Alzheimer's Disease progression is currently difficult to track and predict over time. Instead of relying on a single snapshot of a patient's health, Cerebro-X uses temporal sequence modeling (GRU) to learn a patient's longitudinal trajectory, creating a latent "Digital Brain Twin" representation to predict future clinical dementia ratings (CDR).

## 3. Project architecture
- **Frontend**: Next.js, React, TypeScript (Port 3000)
- **Backend**: FastAPI, Python (Port 8000)
- **Database**: SQLite (persists prediction history)
- **Models**: PyTorch-based `TemporalCerebroNet` (clinical), Bimodal fusion network, and a scaffolded `Lightweight3DCNN` (MRI).

## 4. Folder structure
- `/frontend/` - Next.js UI
- `/src/cerebro_x/` - Core Python backend and ML modules
- `/scripts/` - Execution and training scripts
- `/artifacts/` - Saved PyTorch model checkpoints
- `/data/` - Raw and processed datasets (OASIS-2)
- `/tests/` - PyTest suite

## 5. How frontend works
The Next.js frontend runs as a Single Page Application (SPA), providing an interactive dashboard. It communicates asynchronously with the FastAPI backend using standard REST HTTP requests and visualizes data using React state.

## 6. How backend works
The FastAPI backend serves as the prediction engine. It loads the PyTorch models (`.pt` checkpoints) into memory at startup. It exposes endpoints (e.g., `/predict/clinical`, `/mri/upload`) that accept patient data, run it through the ML models, and return predictions and explanations.

## 7. How database works
A local SQLite database (`cerebro_x.db`) stores patient prediction history. When a prediction is made, the inputs and the model's output are logged via SQLAlchemy so they can be reviewed later on the Prediction Log page.

## 8. How the ML models work
- **TemporalCerebroNet**: Processes a sequence of patient visits. A GRU converts the 19 clinical features per visit into a 64-dimensional hidden state (`Z_t`). A linear classifier predicts the next-visit CDR.
- **Bimodal Model**: Concatenates the clinical `Z_t` with 4 scalar MRI features (nWBV, eTIV, ASF, nWBV_delta) before prediction.
- **CNN3D (Scaffolded)**: Takes a 64x64x64 raw NIfTI voxel input and extracts a 64-D embedding.

## 9. Dataset information
- **OASIS-2 Longitudinal**: 150 subjects, 373 total visits.
- **Data prep**: Filtered to 223 leakage-safe "current → next" visit pairs.
- **Target**: Next-visit Clinical Dementia Rating (CDR) mapped to 4 classes (0, 0.5, 1, 2).

## 10. Exact project startup instructions
Open a PowerShell terminal in the project root:
```powershell
.\run.ps1
```
*(If the script fails, you can run them manually in two terminals: `python scripts/run_api.py` and `cd frontend; npm run dev`)*

## 11. Exact localhost URLs
- **Dashboard**: [http://localhost:3000](http://localhost:3000)
- **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Demo Mode**: [http://localhost:3000/demo](http://localhost:3000/demo)

## 12. Exact button-by-button demo sequence
1. Open `http://localhost:3000`
2. Click **Demo Mode** in the sidebar.
3. Click **Next Step** to read the overview.
4. Click **Predict CDR** in the sidebar.
5. Click **Load Sample Patient**.
6. Click **Predict Next-Visit CDR (Clinical)**.
7. Click **Predict Next-Visit CDR (Bimodal)**.
8. Click **Brain Twin** in the sidebar.
9. Click **Extract Z_t State**.
10. Click **Explainability** in the sidebar.
11. Click **Experiments** in the sidebar.
12. Click **Prediction Log** in the sidebar.

## 13. What to click on each page
*(Covered in section 12)*

## 14. What output to expect
- **Clinical Predict**: A popup showing the predicted CDR class (e.g., "Very Mild (CDR 0.5)").
- **Brain Twin**: A visualization of the 64-dimensional latent vector and patient trajectory.
- **Experiments**: A table of models with 75% accuracy for Clinical GRU and 71.4% for Bimodal.

## 15. What each output means
- **Predicted CDR**: The model's estimation of the patient's dementia severity at their *next* visit.
- **Z_t Vector**: The numerical representation of the patient's brain health learned by the neural network.

## 16. What to say during a project presentation
See the **Presentation Scripts** section below.

## 17. Common errors and fixes
- **"Connection Refused" on port 3000**: Next.js hasn't finished compiling. Wait 15 seconds and refresh.
- **"CNN3D Checkpoint Not Available"**: This is expected. The raw MRI training requires OASIS-2 NIfTI files which are not bundled locally.
- **Port 8000 in use**: Run `.\stop.ps1` to kill orphaned python processes.

## 18. API endpoint documentation
View live at `http://localhost:8000/docs`. Key endpoints:
- `POST /predict/clinical`: Runs TemporalCerebroNet.
- `POST /mri/upload`: Parses NIfTI and returns voxel-based scalars.
- `POST /explain/mri`: Runs 3D Grad-CAM (if checkpoint exists).

## 19. Model architecture explanation
The core is an RNN (GRU). At each time step (visit), it takes the clinical features and updates its internal state. The final hidden state is passed through fully connected layers to output probabilities for the 4 CDR classes.

## 20. Digital Brain Twin explanation
The "Twin" in Cerebro-X is a mathematical concept. It is not a 3D video game brain. It is the GRU's hidden state array that tracks how the patient's cognitive profile degrades or remains stable over time.

## 21. Z_t explanation
`Z_t` is the 64-dimensional hidden vector output by the GRU. It stands for the Latent State (Z) at time (t).

## 22. MRI pipeline explanation
The newly added pipeline loads raw `.nii` files using `nibabel`, reorients them to canonical RAS, normalizes intensities, and resizes them to 64x64x64. This tensor is ready to be fed into the `Lightweight3DCNN`. 

## 23. Explainability explanation
- **Clinical**: Uses heuristic linear formulas to approximate feature importance.
- **MRI**: Implements 3D Grad-CAM, which hooks into the CNN's final convolutional layer to generate heatmaps showing which brain regions influenced the prediction. (Currently disabled pending trained weights).

## 24. Current limitations
- **No 3D CNN weights**: Requires local OASIS-2 MRI training data.
- **Proxy nWBV**: The current NIfTI upload uses a simple intensity threshold to estimate brain volume, which is not equivalent to FreeSurfer's exact masking.
- **Static Twin UI**: The frontend Z_t representation is visually static.

## 25. Future work
- Train the `Lightweight3DCNN` on OASIS-2 NIfTI volumes.
- Integrate the 64-D CNN embedding directly into the Bimodal GRU fusion layer (replacing the scalar proxy).
- Train the EEG fusion layer.

## 26. Honest implementation-status matrix
- **Clinical GRU Pipeline**: WORKING
- **Next-Visit Prediction**: WORKING
- **FastAPI / Next.js**: WORKING
- **Raw NIfTI Preprocessing**: WORKING
- **Raw MRI CNN Inference**: NOT DEPLOYED (Needs checkpoint)
- **Raw MRI Grad-CAM**: NOT DEPLOYED (Needs checkpoint)
- **Tri-modal (Clinical+MRI+EEG)**: PLANNED

## 27. Demo checklist
- [ ] Backend running (`/health` returns 200)
- [ ] Frontend running (`localhost:3000` loads)
- [ ] Demo Mode page renders
- [ ] Sample Patient loads successfully
- [ ] Clinical prediction succeeds
- [ ] Bimodal prediction succeeds
- [ ] MRI pipeline status is explained honestly

## 28, 29. Viva & Interview Questions
*(See `CEREBRO_X_VIVA_GUIDE.md`)*

## 30. 5-minute presentation script
"Welcome to Cerebro-X. This is an AI-powered Digital Brain Twin dashboard for predicting Alzheimer's progression. Instead of a single snapshot, we use longitudinal OASIS-2 data. On the Predict page, we load a patient's historical visits. When I click 'Predict Clinical', our Temporal GRU model processes the sequence, generates a 64-dimensional latent Brain Twin state, and predicts the patient's next-visit Clinical Dementia Rating with 75% accuracy. We also have a bimodal scalar model, and we have fully implemented a raw 3D MRI pipeline which is currently pending model training. Thank you."

## 31. 10-minute presentation script
*(Expand on the 5-minute script by navigating to the Brain Twin tab to explain `Z_t`, opening the Experiments tab to show the baselines, and explaining the architectural difference between the scalar bimodal model and the new raw 3D CNN pipeline.)*

## 32. 15-minute detailed demonstration script
*(Expand on the 10-minute script by walking through the exact Demo Mode steps 1-10, demonstrating the Prediction Log SQLite persistence, and discussing the mathematical mechanics of the GRU hidden state and 3D Grad-CAM implementation limitations.)*
