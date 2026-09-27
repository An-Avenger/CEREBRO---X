# Frontend Design Audit

## Routes Identified
1. `/` (OverviewPage in `src/app/page.tsx`): Displays key metrics, model comparison table, phase tracker, key findings, and quick actions.
2. `/predict` (PredictPage in `src/app/predict/page.tsx`): Patient visits entry, NextGen biomarkers (APOE4, p-tau181), MRI upload, and CDR prediction output with probabilities.
3. `/brain-twin` (BrainTwinPage in `src/app/brain-twin/page.tsx`): Extracts $Z_t$ trajectory from the temporal GRU, visualized via bar charts and heatmaps.
4. `/explainability` (ExplainabilityPage in `src/app/explainability/page.tsx`): Displays pre-generated SHAP feature importance charts and MRI Grad-CAM scaffold visualizations.
5. `/experiments` (ExperimentsPage in `src/app/experiments/page.tsx`): Fetches and displays all backend experiment metrics (`EXP-*`).
6. `/history` (HistoryPage in `src/app/history/page.tsx`): Displays the `PredictionRecord` logs stored in the database.

## Key Components
- `Sidebar` (`src/components/Sidebar.tsx`): Main navigation menu, logo, and backend status indicator.
- `RootLayout` (`src/app/layout.tsx`): Contains the `.app-shell`, the `<Sidebar />`, and the `<main>` content area.

## State Management & API Integration
- React `useState` and `useEffect` are used for localized state (e.g., visits array, loading status).
- `fetch` calls standard REST endpoints on the FastAPI backend:
  - `POST /predict/clinical`, `POST /predict/bimodal`
  - `POST /mri/upload`
  - `POST /brain-twin/extract`
  - `GET /experiments/`
  - `GET /history/`
- Prediction logic relies entirely on the API response objects (`predicted_cdr_class`, `class_probabilities`, etc.).

## Existing Visual System
- Dark mode generic "AI" aesthetic using a navy/cyan palette.
- Background: `#0b0f1a` with radial glowing cyan/violet gradients.
- Typography: Inter + JetBrains Mono.
- Highlight Colors: Neon cyan, violet, emerald, rose, amber.
- Component Style: Deep shadows, glassy blur effects (`.glass`), glowing borders.

## Functionality Constraints
- All backend routes must be preserved. The API payload structures (e.g., `visits`, `mri`) must remain identical.
- Existing features (prediction, extracting twin, viewing experiments, uploading MRI, displaying mock explainability images) must NOT be removed.
- Explanatory text and caveats ("Research Prototype Only", "CNN3D pathway not active") must be retained.
- The redesign will purely target structural HTML/CSS (classes, layouts, tokens) and visual presentation without modifying the React data flow or underlying domain models.
