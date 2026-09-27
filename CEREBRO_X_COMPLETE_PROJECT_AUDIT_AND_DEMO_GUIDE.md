# CEREBRO X — COMPLETE PROJECT AUDIT AND DEMO GUIDE

## TABLE OF CONTENTS
1. Executive Summary
2. Current Project Status
3. Complete Architecture
4. Technology Stack
5. Project Structure
6. Frontend
7. Backend
8. Database
9. API Documentation
10. ML Pipeline
11. MRI Pipeline
12. EEG Pipeline
13. Clinical Data Pipeline
14. Alzheimer's Component
15. Digital Brain Twin
16. Dataset Analysis
17. Model Training
18. Model Inference
19. End-to-End Data Flow
20. What Happens on Localhost
21. Complete Demo Walkthrough
22. 5-Minute Demo Script
23. Setup Guide
24. Commands Cheat Sheet
25. Feature Implementation Matrix
26. Bugs and Issues
27. Demo Failure Checklist
28. Interview Explanation
29. Interview Questions & Answers
30. PPT Structure
31. Viva Questions
32. Developer Code Walkthrough
33. Limitations
34. Future Work
35. Master Demo Checklist
36. CEREBRO X — CURRENT STATE IN ONE PAGE

---

## 1. Executive Summary
Cerebro X is an AI-Based Digital Brain Twin for Predicting Neurological Disease Progression. The system leverages Multimodal data (MRI, EEG, and Clinical records) to output longitudinal predictions, focusing predominantly on Alzheimer's Disease (Clinical Dementia Rating - CDR) progression.

## 2. Current Project Status
- **Backend:** Fully structured with FastAPI and SQLAlchemy.
- **Frontend:** Built with Next.js (React), features a Dashboard UI.
- **ML/AI:** Integrates PyTorch and Scikit-Learn models, handling clinical tabular data, bimodal clinical+MRI, and EEG screening.
- **Database:** SQLite is implemented and functional.

## 3. Complete Architecture
```text
Frontend (Next.js)
   |
   | HTTP Requests (JSON/Multipart)
   v
Backend API (FastAPI) [Port 8000]
   |
   +--> SQLite DB (cerebro_x.db)
   |
   +--> Preprocessing (Clinical, MRI arrays, EEG vectors)
          |
          v
        ML Models (PyTorch / scikit-learn)
          |
          v
      Prediction & Digital Brain Twin (Z_t vector)
          |
   <------+ (JSON Response with predictions & SHAP explainability)
   |
Dashboard Visualization
```

## 4. Technology Stack
| Layer | Technology | Where Used | Purpose |
|------|------|------|------|
| Frontend | Next.js, React 19, CSS | `frontend/` | UI and dashboard visualization |
| Backend | FastAPI, Python 3.10+ | `src/cerebro_x/api/` | REST API, model orchestration |
| Database | SQLite, SQLAlchemy | `cerebro_x.db` | Storing patient history & experiments |
| ML/DL | PyTorch, Scikit-Learn | `src/cerebro_x/models/` | Prediction models |
| Med Imaging | MONAI, NiBabel | MRI processing | 3D Image manipulation |
| Container | Docker, Docker-compose| `Dockerfile`, `docker-compose.yml` | Deployment |

## 5. Project Structure
```text
CEREBRO-X/
├── frontend/             # Next.js Application
│   ├── src/app/          # Pages and routing
│   ├── src/components/   # Reusable UI components
│   └── package.json
├── src/cerebro_x/        # Backend & ML Core
│   ├── api/              # FastAPI routers and main.py
│   ├── models/           # Deep learning models
│   ├── brain_twin/       # Digital twin logic
│   └── imaging/          # MRI processing scripts
├── cerebro_x.db          # SQLite Database
├── pyproject.toml        # Python Dependencies
├── requirements.txt      
└── docker-compose.yml
```

## 6. Frontend
- **Framework:** Next.js (App router)
- **Path:** `frontend/src/app`
- **Key Components:**
  - `DashboardWidget.tsx`: [IMPLEMENTED] Displays metrics.
  - `Sidebar.tsx`: [IMPLEMENTED] Navigation.
  - `page.tsx`: [IMPLEMENTED] Main entry view.

## 7. Backend
- **Framework:** FastAPI
- **Entry Point:** `src/cerebro_x/api/main.py`
- **Routers:** `prediction`, `brain_twin`, `experiments`, `eeg`, `history`, `mri_upload`, `patient`, `explain`
- Models are preloaded into memory upon startup via `lifespan` context manager.

## 8. Database
- **Technology:** SQLite
- **File:** `cerebro_x.db`
- **Status:** [IMPLEMENTED] Handles patient tracking, history, and experimental records via SQLAlchemy.

## 9. API Documentation
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Project info |
| GET | `/health` | Model load status |
| POST | `/predict/clinical` | CDR prediction via Clinical GRU |
| POST | `/predict/bimodal` | CDR prediction (Clinical + MRI) |
| POST | `/brain-twin/extract` | Z_t trajectory extraction (Digital Twin) |
| GET | `/experiments/` | List experiments |
| POST | `/eeg/screen` | EEG binary disease screener |

## 10. ML Pipeline
Data is received via HTTP, structured by Pydantic models. Numerical features are normalized, imaging is processed via PyTorch/MONAI, passed to deep neural nets for extraction, resulting in classification logits and SHAP attribution values.

## 11. MRI Pipeline
- **Status:** [IMPLEMENTED]
- **Process:** Accepts imaging formats, utilizes `nibabel` and `monai` for feature extraction. Fused in `/predict/bimodal` alongside clinical records.

## 12. EEG Pipeline
- **Status:** [IMPLEMENTED]
- **Process:** Takes EEG signal features to run through `/eeg/screen` for binary classification.

## 13. Clinical Data Pipeline
- **Status:** [IMPLEMENTED]
- **Process:** Relies on standardized demographic and cognitive features, processed by a GRU-based clinical model.

## 14. Alzheimer's Component
- **Focus:** Predicting Clinical Dementia Rating (CDR) progression.
- **Mechanism:** Longitudinal analysis assessing the risk of progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease.

## 15. Digital Brain Twin
- **Status:** [IMPLEMENTED]
- **Mechanism:** Located in `src/cerebro_x/brain_twin/extractor.py`. Maps multimodal input into a latent "Z_t trajectory" space acting as the patient's digital surrogate for progression tracking over time.

## 16. Dataset Analysis
- **ADNI / OASIS-2:** [REFERENCED] 150 subjects for clinical/MRI.
- **OpenNeuro ds004504:** [REFERENCED] 87 EEG subjects.

## 17-18. Model Training & Inference
- **Inference:** [IMPLEMENTED] Models are dynamically loaded at API boot.
- **Training:** Notebooks such as `Phase3_MRI_Training.ipynb` contain the training code, demonstrating that models were trained on-device/in-repo.

## 19. End-to-End Data Flow
1. User enters data on Frontend.
2. Next.js triggers POST to FastAPI.
3. API validates schema, queries DB if needed.
4. Input routed to `models` / `brain_twin`.
5. Inference executed, outputs (Z_t, predictions) generated.
6. JSON returned to Frontend for dashboard rendering.

## 20. What Happens on Localhost
- FastAPI boots on `:8000`, creates SQLite tables, loads `.pt`/models into memory.
- Next.js boots on `:3000`.
- User visits `http://localhost:3000`, React mounts and fetches data.

## 21. Complete Demo Walkthrough
1. Run `./run.ps1` to boot backend and frontend.
2. Open `localhost:3000` in a browser.
3. You will see the main Dashboard.
4. Select / Create a Patient.
5. Input clinical features, upload an MRI scan.
6. Click "Generate Brain Twin".
7. Wait for the API to return the longitudinal prediction (CDR trajectory).
8. View SHAP explainability charts.

## 22. 5-Minute Demo Script
- **00:00:** "Welcome to Cerebro X, an AI digital brain twin system."
- **01:00:** "Let's load patient X's clinical data and their latest MRI."
- **02:00:** "I click Predict. The FastAPI backend is fusing this bimodal data."
- **03:00:** "Here is the Digital Brain Twin trajectory showing their predicted CDR score for the next 3 years."
- **04:00:** "Note the SHAP explainability highlighting the hippocampus region."
- **05:00:** "All data is securely saved in our local SQLite db."

## 23. Setup Guide
**Prerequisites:** Python 3.10+, Node.js 18+.
**Commands:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd frontend
npm install
cd ..
./run.ps1
```

## 24. Commands Cheat Sheet
- **Start All:** `./run.ps1` or `docker-compose up --build -d`
- **Stop All:** `./stop.ps1` or `docker-compose down`
- **API Logs:** `docker logs cerebro-x-api -f`

## 25. Feature Implementation Matrix
| Feature | Status |
|--------|--------|
| Frontend Dashboard | IMPLEMENTED |
| FastAPI Backend | IMPLEMENTED |
| SQLite Database | IMPLEMENTED |
| Clinical GRU Model | IMPLEMENTED |
| Bimodal (MRI) Model| IMPLEMENTED |
| Digital Twin Extractor| IMPLEMENTED |

## 26. Bugs and Issues
- **LOW:** Database lock issues can occur if concurrent long-running MRI tasks block SQLite. 

## 27. Demo Failure Checklist
- [ ] Backend running (`localhost:8000/health` returns ok)
- [ ] Frontend running (`localhost:3000`)
- [ ] SQLite database file exists
- [ ] Test MRI sample is accessible

## 28. Interview Explanation
**30-Second:** "Cerebro X is a multimodal AI system that builds a digital surrogate of a patient's brain using MRI, EEG, and clinical data to predict Alzheimer's progression longitudinally."

## 29. Interview Questions & Answers
**Q:** How do you fuse the modalities?
**A:** We use intermediate fusion where embeddings from the CNN (MRI) and GRU (Clinical) are concatenated before passing to a classification head.

## 30-31. PPT & Viva Prep
*Focus on the novelty of "Digital Brain Twin" (extracting latent Z_t space as a surrogate marker) rather than just standard classification.*

## 32. Developer Code Walkthrough
`main.py` -> `prediction.py` router -> `services/model_loader.py` -> PyTorch model `forward()` -> Response JSON.

## 33. Limitations
Relies on pre-extracted features for certain EEG data. High compute requirements for raw 3D MRI inference.

## 34. Future Work
Integration with live EHR streams and expanding to other neurological diseases (e.g., Parkinson's).

## 35. Master Demo Checklist
- [x] API Health Check
- [x] UI Loading
- [x] Dummy patient data ready
- [x] Inference runs successfully

## 36. CEREBRO X — CURRENT STATE IN ONE PAGE
**Working:** FastAPI backend, Next.js dashboard, SQLite tracking, Multimodal Model Inference, Digital Twin Z_t extraction.
**Missing:** Advanced authentication / Login (Currently open for research).
**How to Run:** `docker-compose up -d`
**Key Fact:** Cerebro X doesn't just predict; it tracks a "Digital Brain Twin" trajectory over time using a dedicated `brain_twin` extractor.
