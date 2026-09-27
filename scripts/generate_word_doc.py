#!/usr/bin/env python
"""
scripts/generate_word_doc.py
----------------------------
Generates the complete Master Handoff Word document for Cerebro X.
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.table import WD_TABLE_ALIGNMENT
import datetime

def set_cell_bg(cell, hex_color):
    """Sets a table cell's background color."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12 if level == 1 else 6)
    return h

def add_para(doc, text, bold=False, italic=False, font_size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(font_size)
    return p

def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    if bold_prefix:
        run_b = p.add_run(bold_prefix)
        run_b.bold = True
    p.add_run(text)
    return p

def add_code(doc, code_text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    run = p.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(0x19, 0x69, 0x2C)
    p.paragraph_format.space_after = Pt(4)
    return p

def add_section_divider(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'AAAAAA')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p

def main():
    doc = Document()

    # ─── PAGE MARGINS ────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin   = Inches(1.0)
        section.right_margin  = Inches(1.0)

    # ─── TITLE BLOCK ─────────────────────────────────────────────────
    title = doc.add_heading('CEREBRO X', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.runs[0].font.size = Pt(28)
    title.runs[0].font.color.rgb = RGBColor(0x1A, 0x53, 0x76)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = subtitle.add_run('Master Project Handoff Document & User Manual')
    sr.font.size = Pt(14)
    sr.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph()

    # Details table
    detail_table = doc.add_table(rows=5, cols=2)
    detail_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    detail_table.style = 'Table Grid'
    rows_data = [
        ("Student", "Aryan Sharma (25MCS1018)"),
        ("Project", "Cerebro X — Alzheimer's Progression Prediction AI"),
        ("Degree", "M.Tech Computer Science"),
        ("Date", "August 19, 2026"),
        ("Status", "Phase 4 Complete | Phase 5 (Kaggle GPU Training) — PENDING"),
    ]
    for i, (label, value) in enumerate(rows_data):
        row = detail_table.rows[i]
        set_cell_bg(row.cells[0], 'D6E4F0')
        row.cells[0].text = label
        row.cells[0].paragraphs[0].runs[0].bold = True
        row.cells[1].text = value

    doc.add_paragraph()
    add_section_divider(doc)

    # ─── SECTION 1: WHAT THIS PROJECT IS ─────────────────────────────
    add_heading(doc, '1.  What Is This Project? (The Goal)', level=1)
    add_para(doc,
        "Cerebro X is an M.Tech research AI system that attempts to predict how severe a patient's dementia "
        "will be at their NEXT hospital visit, using data from their CURRENT visit.\n\n"
        "The target output is called CDR (Clinical Dementia Rating):\n"
        "   CDR 0.0  = Healthy (no dementia)\n"
        "   CDR 0.5  = Very Mild Dementia\n"
        "   CDR 1.0  = Mild Dementia\n"
        "   CDR 2.0  = Moderate Dementia\n\n"
        "The AI learns from a real patient dataset called OASIS-2 — a longitudinal study of 150 older adults "
        "who were scanned multiple times over several years. The AI looks at clinical numbers (age, education, "
        "brain volume, memory test scores) and eventually 3D MRI brain scan images to predict where the patient "
        "is headed."
    )

    # ─── SECTION 2: RULES (NON-NEGOTIABLE) ───────────────────────────
    add_heading(doc, '2.  Non-Negotiable Rules (Enforced Throughout)', level=1)
    add_bullet(doc, "NEVER invent results. If a model hasn't run, no numbers are reported.", "NO HALLUCINATIONS: ")
    add_bullet(doc, "ALL patient data is split by patient ID, never by individual visits. This prevents the model from 'remembering' a patient.", "NO DATA LEAKAGE: ")
    add_bullet(doc, "OASIS-2 is a longitudinal metadata CSV from Kaggle. It is NOT OASIS-3 (which has MRI). Never confuse these.", "DATASET HONESTY: ")
    add_bullet(doc, "Any model abandoned mid-way must not reappear in reports.", "NO RESURRECTIONS: ")

    add_section_divider(doc)

    # ─── SECTION 3: DATASET ──────────────────────────────────────────
    add_heading(doc, '3.  The Dataset', level=1)
    add_bullet(doc, "OASIS-2 Longitudinal MRI Data (via Kaggle: jboysen/mri-and-alzheimers)", "Source: ")
    add_bullet(doc, "data/raw/oasis_longitudinal.csv", "File Location: ")
    add_bullet(doc, "150 right-handed patients, all aged 60+, each scanned 2-5 times over many years.", "Subjects: ")
    add_bullet(doc, "373 total rows (one per visit). We reduced this to 223 usable visit-pairs for prediction.", "Size: ")

    add_heading(doc, 'Key Clinical Columns Used', level=2)
    col_table = doc.add_table(rows=9, cols=2)
    col_table.style = 'Table Grid'
    col_table.alignment = WD_TABLE_ALIGNMENT.LEFT
    headers = col_table.rows[0]
    set_cell_bg(headers.cells[0], '1A5376')
    set_cell_bg(headers.cells[1], '1A5376')
    headers.cells[0].text = 'Column Name'; headers.cells[0].paragraphs[0].runs[0].bold = True; headers.cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    headers.cells[1].text = 'What It Means'; headers.cells[1].paragraphs[0].runs[0].bold = True; headers.cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    cols_data = [
        ("CDR", "Clinical Dementia Rating (our TARGET to predict)"),
        ("MMSE", "Mini-Mental State Exam score (memory test, 0-30)"),
        ("Age", "Patient's age at time of visit"),
        ("EDUC", "Years of education"),
        ("SES", "Socioeconomic status (1-5)"),
        ("nWBV", "Normalized Whole Brain Volume (brain shrinkage indicator)"),
        ("eTIV", "Estimated Total Intracranial Volume"),
        ("ASF", "Atlas Scaling Factor (brain size normalization)"),
    ]
    for i, (col, meaning) in enumerate(cols_data):
        row = col_table.rows[i + 1]
        if i % 2 == 0:
            set_cell_bg(row.cells[0], 'EAF2F8')
            set_cell_bg(row.cells[1], 'EAF2F8')
        row.cells[0].text = col
        row.cells[1].text = meaning

    add_section_divider(doc)

    # ─── SECTION 4: TECH STACK ───────────────────────────────────────
    add_heading(doc, '4.  Complete Technology Stack', level=1)
    tech_table = doc.add_table(rows=12, cols=3)
    tech_table.style = 'Table Grid'
    header_cells = tech_table.rows[0].cells
    for c in header_cells:
        set_cell_bg(c, '1A5376')
    header_cells[0].text = 'Library'; header_cells[0].paragraphs[0].runs[0].bold = True; header_cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    header_cells[1].text = 'Version'; header_cells[1].paragraphs[0].runs[0].bold = True; header_cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    header_cells[2].text = 'Why We Use It'; header_cells[2].paragraphs[0].runs[0].bold = True; header_cells[2].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    tech_data = [
        ("Python",       "3.13",     "Core programming language"),
        ("pandas",       ">=2.0",    "Loading and cleaning the CSV data"),
        ("numpy",        ">=1.24",   "Numerical array math"),
        ("scikit-learn", ">=1.3",    "Classical ML models (LR, RF, Dummy) + imputation + scaling"),
        ("torch (PyTorch)", ">=2.0", "Deep Learning framework for Neural Networks"),
        ("nibabel",      ">=5.1",    "Reading real 3D MRI .nii.gz brain scan files"),
        ("monai",        ">=1.3",    "Medical AI tools (transforms, augmentation for medical images)"),
        ("joblib",       ">=1.3",    "Saving and loading trained model weights to disk"),
        ("matplotlib",   ">=3.7",    "Plotting confusion matrices and training curves"),
        ("seaborn",      ">=0.12",   "Enhanced statistical visualization"),
        ("pytest",       ">=7.4",    "Automated testing to ensure code stays bug-free"),
    ]
    for i, (lib, ver, why) in enumerate(tech_data):
        row = tech_table.rows[i + 1]
        if i % 2 == 0:
            set_cell_bg(row.cells[0], 'EAF2F8')
            set_cell_bg(row.cells[1], 'EAF2F8')
            set_cell_bg(row.cells[2], 'EAF2F8')
        row.cells[0].text = lib
        row.cells[1].text = ver
        row.cells[2].text = why

    add_section_divider(doc)

    # ─── SECTION 5: PHASES ───────────────────────────────────────────
    add_heading(doc, '5.  Project Phases — What We Did', level=1)

    add_heading(doc, 'Phase 0 — Repository Foundation  [DONE]', level=2)
    add_para(doc, "Set up the entire project skeleton. Nothing trains yet — this is all about structure.")
    add_bullet(doc, "Created the Python package structure under src/cerebro_x/")
    add_bullet(doc, "Wrote the configuration files (YAML) that control dataset paths and experiment settings")
    add_bullet(doc, "Wrote the data schema rules (what columns must exist, what types they must be)")
    add_bullet(doc, "Created 10+ documentation files in the docs/ folder")
    add_bullet(doc, "Set up pyproject.toml and requirements.txt")
    add_bullet(doc, "Created .gitignore to avoid committing data/models to Git")

    add_heading(doc, 'Phase 1 — Data Audit & Pair Building  [DONE]', level=2)
    add_para(doc, "Loaded the real CSV, cleaned it, and built the 223 prediction pairs.")
    add_bullet(doc, "audit_oasis2.py: Scanned the CSV, printed statistics about missing values, CDR distribution, visit counts")
    add_bullet(doc, "build_pairs.py: Linked every patient's current visit to their next visit, creating the 223 labeled examples")
    add_bullet(doc, "splits.py: Splits patients — not visits — into train/validation/test groups (prevents data leakage)")
    add_bullet(doc, "Output: data/processed/next_visit_pairs.csv — the cleaned, ready-to-train dataset")

    add_heading(doc, 'Phase 2 — Classical Baseline Models  [DONE]', level=2)
    add_para(doc, "Trained three traditional machine learning models on the 223 pairs.")
    add_bullet(doc, "train_baselines.py: Loaded pairs, split patients, extracted features, trained all 3 models")
    add_bullet(doc, "Used class_weight='balanced' to force models to pay attention to rare severely-demented patients")
    add_bullet(doc, "Evaluation metric: Balanced Accuracy (not raw accuracy, which would cheat on imbalanced data)")
    add_bullet(doc, "Output: artifacts/EXP-BASELINE-001/ — metrics, model files, confusion matrix plots")

    add_heading(doc, 'Actual Phase 2 Results (Verified — from metrics.json)', level=3)
    results_table = doc.add_table(rows=4, cols=5)
    results_table.style = 'Table Grid'
    h_cells = results_table.rows[0].cells
    for c in h_cells: set_cell_bg(c, '2E4057')
    for c, lbl in zip(h_cells, ['Model', 'Train Accuracy', 'Train Balanced Acc', 'Test Accuracy', 'Test Balanced Acc']):
        c.text = lbl
        c.paragraphs[0].runs[0].bold = True
        c.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    results_data = [
        ("Dummy Classifier",   "56.2%", "25.0%",  "39.3%", "25.0%"),
        ("Logistic Regression","85.1%", "87.9%",  "67.9%", "72.0%  ← Best"),
        ("Random Forest",      "100%",  "100%",   "64.3%", "45.7%  (Overfit)"),
    ]
    for i, rd in enumerate(results_data):
        row = results_table.rows[i + 1]
        if i == 1:
            for c in row.cells: set_cell_bg(c, 'D5F5E3')  # highlight best
        for cell, val in zip(row.cells, rd):
            cell.text = val

    add_heading(doc, 'Phase 3 — PyTorch Deep Learning Scaffold  [DONE]', level=2)
    add_para(doc, "Upgraded from traditional math to PyTorch Neural Networks.")
    add_bullet(doc, "LongitudinalClinicalDataset: Custom PyTorch Dataset — reads CSV, imputes missing values, scales features, converts to Tensors")
    add_bullet(doc, "ClinicalMLP: Multi-Layer Perceptron with Dropout (p=0.3) and Batch Normalization for tabular data")
    add_bullet(doc, "train_nn_baseline.py: Full PyTorch training loop with AdamW optimizer and weighted CrossEntropyLoss")
    add_bullet(doc, "10 epochs ran locally on CPU. Loss decreased each epoch. Model saved: artifacts/EXP-NN-BASELINE-001/clinical_mlp_cpu.pt")

    add_heading(doc, 'Phase 4 — MRI Integration & Multi-Modal Architecture  [DONE]', level=2)
    add_para(doc, "Built the architecture to look at brain images + clinical data simultaneously.")
    add_bullet(doc, "MultiModalDataset: Extended dataset that returns (clinical_tensor, mri_tensor, label). Generates fake 3D 'noise' MRI for local testing.")
    add_bullet(doc, "Lightweight3DCNN: 4-block 3D CNN with MaxPooling + AdaptiveAvgPool. Takes (1, 64, 64, 64) brain volume → outputs 64-dim embedding.")
    add_bullet(doc, "MultiModalCerebroNetwork: Fusion model. Clinical branch (MLP → 32-dim) + MRI branch (CNN → 64-dim) → concatenated → final 4-class prediction.")
    add_bullet(doc, "train_multimodal.py: Full training loop with batch_size=2, drop_last=True, class-weighted CrossEntropyLoss.")
    add_bullet(doc, "All 3 tests in test_multimodal.py: PASSED. Training ran 2 epochs. Model saved: artifacts/EXP-MULTIMODAL-001/multimodal_cpu.pt")

    add_section_divider(doc)

    # ─── SECTION 6: COMPLETE FILE MAP ────────────────────────────────
    add_heading(doc, '6.  Complete File & Folder Map', level=1)
    add_para(doc, "Every file and folder that exists in E:\\PROJECTS\\CEREBRO-X as of August 19, 2026.", italic=True)

    file_map = [
        ("ROOT LEVEL", "", False),
        (".gitignore", "Tells Git to ignore data files, model weights, __pycache__, etc.", True),
        ("pyproject.toml", "Python package configuration, test settings, and project metadata.", True),
        ("requirements.txt", "All Python library dependencies (see Section 4).", True),
        ("README.md", "Project overview for GitHub.", True),
        ("Architecture.md", "High-level architectural decisions (historical, written before our build).", True),
        ("Design.md", "Design philosophy document (historical).", True),
        ("PRD.md", "Product Requirements Document (historical).", True),
        ("Phases.md", "Project phase plan (historical).", True),
        ("Rules.md", "Non-negotiable coding and research rules.", True),
        ("CEREBRO_X_HANDOFF_MANUAL.docx", "THIS DOCUMENT — the master handoff file.", True),
        ("", "", False),
        ("configs/", "", False),
        ("configs/base.yaml", "Master config — dataset paths, random seed, experiment name.", True),
        ("configs/datasets/oasis2_kaggle.yaml", "Dataset-specific constants (column names, CDR classes, etc.).", True),
        ("configs/experiments/exp_baseline_001.yaml", "Config snapshot for Experiment 1.", True),
        ("", "", False),
        ("data/", "", False),
        ("data/raw/oasis_longitudinal.csv", "The real OASIS-2 CSV downloaded from Kaggle. Do NOT commit to Git.", True),
        ("data/processed/next_visit_pairs.csv", "The 223 cleaned longitudinal pairs, ready for training.", True),
        ("", "", False),
        ("docs/", "", False),
        ("docs/MEMORY.md", "Living log of every action actually completed.", True),
        ("docs/DATASET_CARD.md", "Formal description of the OASIS-2 dataset.", True),
        ("docs/EXPERIMENT_PROTOCOL.md", "Rules for how experiments must be conducted.", True),
        ("docs/KAGGLE_TRAINING.md", "Instructions for running training on Kaggle GPU.", True),
        ("docs/MRI_INTEGRATION.md", "Plan and notes for integrating real MRI files.", True),
        ("docs/RESEARCH_LIMITATIONS.md", "Known limitations of this project (small sample size, etc.).", True),
        ("docs/ADNI_EXTERNAL_VALIDATION.md", "Plan for external validation on ADNI dataset.", True),
        ("docs/adr/ADR-001-primary-dataset.md", "Decision record: why OASIS-2 over OASIS-3.", True),
        ("docs/adr/ADR-002-prediction-target.md", "Decision record: why CDR as the prediction target.", True),
        ("", "", False),
        ("src/cerebro_x/", "", False),
        ("src/cerebro_x/data/schemas.py", "PairColumns schema class — defines required column names.", True),
        ("src/cerebro_x/data/provenance.py", "MD5 hash checker for dataset integrity verification.", True),
        ("src/cerebro_x/data/build_longitudinal_pairs.py", "Logic to link current_visit → next_visit per patient.", True),
        ("src/cerebro_x/data/pytorch/__init__.py", "Package init.", True),
        ("src/cerebro_x/data/pytorch/datasets.py", "LongitudinalClinicalDataset + MultiModalDataset (PyTorch).", True),
        ("src/cerebro_x/data/oasis2/", "OASIS-2 specific data loaders.", True),
        ("src/cerebro_x/evaluation/metrics.py", "evaluate_predictions(), plot_confusion_matrix_custom().", True),
        ("src/cerebro_x/evaluation/splits.py", "subject_level_split() — leak-proof patient-based splitting.", True),
        ("src/cerebro_x/features/clinical.py", "build_feature_matrix() — extracts and engineers 19 features.", True),
        ("src/cerebro_x/models/baselines.py", "get_all_baselines() — Dummy, LogReg, RandomForest pipelines.", True),
        ("src/cerebro_x/models/deep/__init__.py", "Package init.", True),
        ("src/cerebro_x/models/deep/mlp.py", "ClinicalMLP — feed-forward NN for tabular clinical data.", True),
        ("src/cerebro_x/models/deep/cnn3d.py", "Lightweight3DCNN — 4-block 3D CNN for MRI volumes.", True),
        ("src/cerebro_x/models/deep/fusion.py", "MultiModalCerebroNetwork — fuses clinical + MRI predictions.", True),
        ("src/cerebro_x/utils/io.py", "setup_logging(), make_artifact_dir(), save_json().", True),
        ("", "", False),
        ("scripts/", "", False),
        ("scripts/audit_oasis2.py", "Reads the CSV and prints statistics about the raw dataset.", True),
        ("scripts/build_pairs.py", "Generates data/processed/next_visit_pairs.csv.", True),
        ("scripts/train_baselines.py", "Trains the 3 Phase 2 classical ML models.", True),
        ("scripts/train_nn_baseline.py", "Trains the Phase 3 ClinicalMLP neural network.", True),
        ("scripts/train_multimodal.py", "Trains the Phase 4 Multi-Modal Fusion network.", True),
        ("scripts/generate_word_doc.py", "Generates this Word document.", True),
        ("", "", False),
        ("tests/", "", False),
        ("tests/conftest.py", "Shared pytest fixtures (synthetic patient data).", True),
        ("tests/test_loader.py", "Tests that the CSV loads and validates correctly.", True),
        ("tests/test_longitudinal_pairs.py", "Tests the pair-building logic and leakage prevention.", True),
        ("tests/test_splits.py", "Tests subject-level splitting with no patient overlap.", True),
        ("tests/test_features.py", "Tests feature extraction and engineering.", True),
        ("tests/test_metrics.py", "Tests evaluation metric calculations.", True),
        ("tests/test_models.py", "Tests classical model initialization and fitting.", True),
        ("tests/test_pytorch.py", "Tests PyTorch dataset shapes and MLP forward pass.", True),
        ("tests/test_multimodal.py", "Tests MultiModalDataset, 3D CNN, and Fusion network shapes.", True),
        ("", "", False),
        ("artifacts/", "", False),
        ("artifacts/EXP-BASELINE-001/", "Phase 2 outputs: metrics.json, model weights (.joblib), confusion matrix plots.", True),
        ("artifacts/EXP-NN-BASELINE-001/", "Phase 3 output: clinical_mlp_cpu.pt (PyTorch model weights).", True),
        ("artifacts/EXP-MULTIMODAL-001/", "Phase 4 output: multimodal_cpu.pt (Fusion network weights).", True),
        ("artifacts/AUDIT-001/", "Phase 1 output: raw dataset audit report.", True),
    ]

    for name, desc, is_file in file_map:
        if not name:
            doc.add_paragraph()
            continue
        if not is_file:
            p = doc.add_paragraph()
            run = p.add_run(f"  {name}")
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(0x1A, 0x53, 0x76)
        else:
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_after = Pt(1)
            r_name = p.add_run(name)
            r_name.font.name = 'Courier New'
            r_name.font.size = Pt(9)
            r_name.bold = True
            if desc:
                r_desc = p.add_run(f"  —  {desc}")
                r_desc.font.size = Pt(9)

    add_section_divider(doc)

    # ─── SECTION 7: USER MANUAL ───────────────────────────────────────
    add_heading(doc, '7.  User Manual — How To Run Everything', level=1)
    add_para(doc, "All commands below are run in a terminal (Command Prompt or PowerShell) inside the project folder: E:\\PROJECTS\\CEREBRO-X", italic=True)

    add_heading(doc, 'Step 1 — Verify Nothing Is Broken (Run Tests)', level=2)
    add_para(doc, "Run this EVERY time you open the project to confirm all code is intact:")
    add_code(doc, "python -m pytest tests/ -v")
    add_para(doc, "You should see a list of green PASSED messages. If anything is RED (FAILED), stop and fix it before doing anything else.")

    add_heading(doc, 'Step 2 — Check / Rebuild the Dataset', level=2)
    add_para(doc, "Only needed if data/processed/next_visit_pairs.csv is missing:")
    add_code(doc, "python scripts/audit_oasis2.py --config configs/datasets/oasis2_kaggle.yaml")
    add_code(doc, "python scripts/build_pairs.py --config configs/base.yaml")

    add_heading(doc, 'Step 3 — Train the Classical Baseline Models', level=2)
    add_para(doc, "Trains the Dummy, Logistic Regression, and Random Forest models. Very fast (< 10 seconds). Results in artifacts/EXP-BASELINE-001/")
    add_code(doc, "python scripts/train_baselines.py")

    add_heading(doc, 'Step 4 — Train the PyTorch Text AI (Phase 3)', level=2)
    add_para(doc, "Trains the ClinicalMLP Neural Network on your CPU. Takes about 30 seconds for 10 epochs. Results in artifacts/EXP-NN-BASELINE-001/")
    add_code(doc, "python scripts/train_nn_baseline.py")

    add_heading(doc, 'Step 5 — Test the Multi-Modal Fusion AI (Phase 4)', level=2)
    add_para(doc, "Runs the MRI+Clinical Fusion network on your CPU using FAKE/DUMMY 3D images to prove the math is correct. Only runs 2 epochs on 10 patients. Results in artifacts/EXP-MULTIMODAL-001/")
    add_code(doc, "python scripts/train_multimodal.py")

    add_heading(doc, 'Step 6 — Full Training on Kaggle/Colab GPU (Phase 5 — PENDING)', level=2)
    add_para(doc,
        "1. Zip the entire project folder\n"
        "2. Upload to Kaggle as a 'Dataset' or push to GitHub and clone in Kaggle\n"
        "3. Add the OASIS-2 real .nii.gz MRI files as a Kaggle Dataset input\n"
        "4. In your Kaggle Notebook, run: python scripts/train_multimodal.py\n"
        "5. The train_multimodal.py script will detect the real MRI files via the mri_dir parameter\n"
        "6. Training on a P100/T4 GPU should take minutes instead of hours"
    )

    add_section_divider(doc)

    # ─── SECTION 8: NOTE TO FUTURE AI ─────────────────────────────────
    add_heading(doc, '8.  Note to the AI Assistant (Critical Resume Point)', level=1)
    note_box = doc.add_paragraph()
    note_box.paragraph_format.left_indent = Inches(0.3)
    note_box.paragraph_format.right_indent = Inches(0.3)
    note_run = note_box.add_run(
        "DEAR FUTURE AI AGENT — READ THIS FIRST BEFORE DOING ANYTHING:\n\n"
        "1. PHASES 0–4 ARE FULLY COMPLETE. Do not re-build any of the following: schemas.py, splits.py, "
        "metrics.py, baselines.py, mlp.py, cnn3d.py, fusion.py, or any test file. They all pass.\n\n"
        "2. THE TARGET VARIABLE is next_CDR (mapped 0.0→0, 0.5→1, 1.0→2, 2.0→3 for CrossEntropyLoss).\n\n"
        "3. DATA LEAKAGE PREVENTION is already enforced in splits.py via subject_level_split(). "
        "NEVER split by visit row. ALWAYS split by Subject ID.\n\n"
        "4. CLASSICAL BASELINE SCORE TO BEAT: Logistic Regression = 72.0% Balanced Accuracy on test set.\n\n"
        "5. THE MRI PIPELINE is scaffolded using dummy tensors. Real .nii.gz files are NOT available locally. "
        "Do not try to download terabytes of MRI data to the user's PC.\n\n"
        "6. NEXT STEP: The user needs to upload this codebase to Kaggle or Google Colab and run "
        "train_multimodal.py with the real MRI dataset. See docs/KAGGLE_TRAINING.md for detailed instructions.\n\n"
        "7. DO NOT INVENT NUMBERS. Any metrics not in artifacts/EXP-BASELINE-001/metrics.json "
        "have NOT been measured yet."
    )
    note_run.font.size = Pt(10)
    note_run.font.color.rgb = RGBColor(0x6E, 0x26, 0x00)

    add_section_divider(doc)
    add_para(doc, "Document generated automatically by scripts/generate_word_doc.py on August 19, 2026.", italic=True, font_size=9)

    # ─── SAVE ────────────────────────────────────────────────────────
    out_path = 'E:/PROJECTS/CEREBRO-X/CEREBRO_X_HANDOFF_MANUAL.docx'
    doc.save(out_path)
    print(f"[SUCCESS] Word document saved to: {out_path}")


if __name__ == '__main__':
    main()
