import docx
import os

def create_documentation():
    doc = docx.Document()
    
    doc.add_heading('Cerebro-X: Complete Project Documentation', 0)
    
    doc.add_heading('1. Project Overview', level=1)
    doc.add_paragraph(
        "Cerebro-X is an end-to-end Machine Learning research system that builds a 'Longitudinal Digital Brain Twin'. "
        "The primary goal of this system is to learn how a patient's brain health evolves over time and predict future cognitive decline. "
        "Unlike standard snapshot-based models, Cerebro-X processes the patient's entire visit history through a temporal GRU (Gated Recurrent Unit) "
        "network. This allows the model to capture how symptoms and structural brain changes progress between visits."
    )
    doc.add_paragraph(
        "The project provides a complete pipeline from raw data preprocessing (MRI and clinical data) to temporal modeling, explainability (SHAP/Grad-CAM), "
        "and a full-stack research interface (FastAPI + React/Next.js)."
    )

    doc.add_heading('2. Dataset Information', level=1)
    doc.add_paragraph(
        "The system was developed and validated primarily on the OASIS-3 (Open Access Series of Imaging Studies) dataset, with ADNI as a secondary/external validation cohort."
    )
    doc.add_paragraph(
        "OASIS-2/3 Specs:\n"
        "- 150+ patients, 373+ sessions.\n"
        "- 223 longitudinal visit pairs built for modeling.\n"
        "- Clinical features include: Age, MMSE (Mini-Mental State Examination), CDR (Clinical Dementia Rating), "
        "brain volumes (nWBV, eTIV), and socioeconomic status.\n"
        "- The target prediction is CDR 4-class prediction or cognitive change."
    )

    doc.add_heading('3. Models Used', level=1)
    doc.add_paragraph(
        "To ensure scientific rigor, a series of baselines were evaluated against the advanced Cerebro-X model. The models include:"
    )
    doc.add_paragraph("1. Dummy Classifier: Balanced Accuracy of 25.0% (CPU)")
    doc.add_paragraph("2. Random Forest: Balanced Accuracy of 45.7% (CPU)")
    doc.add_paragraph("3. Logistic Regression (Baseline): Balanced Accuracy of 72.0% (CPU)")
    doc.add_paragraph("4. Last-Visit Baseline: Balanced Accuracy of 53.7% (CPU)")
    doc.add_paragraph("5. Temporal GRU (Cerebro-X): The main longitudinal architecture which achieved 55.7% Balanced Accuracy predicting progression using sequential data (trained on Kaggle T4 GPU).")

    doc.add_heading('4. Architecture', level=1)
    doc.add_paragraph(
        "Cerebro-X is divided into five layers to deliberately separate research code from application code:"
    )
    doc.add_paragraph("1. Data Layer: OASIS-3 / ADNI Adapters.")
    doc.add_paragraph("2. Processing Layer: MRI pipeline (Quality Control, spatial normalization) and Clinical pipeline.")
    doc.add_paragraph("3. Representation Learning: Independent MRI Encoders and Clinical Encoders fused into a Patient Brain State (Z_t).")
    doc.add_paragraph("4. Prediction and Explainability: Temporal Longitudinal model predicting future states, analyzed by SHAP and Grad-CAM.")
    doc.add_paragraph("5. Application/API: FastAPI research backend and a React/Next.js frontend dashboard.")

    doc.add_heading('5. Steps Followed (Development Phases)', level=1)
    
    phases = [
        ("Phase 0 - Project Initialization", "Created the repository, research documentation, architecture design, and environment configuration."),
        ("Phase 1 - Dataset Acquisition and Audit", "Established the available data from OASIS-3 and ADNI, downloaded metadata, and selected CDR 4-class prediction as the target."),
        ("Phase 2 - Data Engineering", "Processed MRI and clinical data, handled missingness, matched visits, and created a reproducible clean dataset with proper subject-level splitting."),
        ("Phase 3 - Baseline Models", "Trained scientifically meaningful baselines (Clinical-only, MRI-only, static fusion) and evaluated them using predefined splits."),
        ("Phase 4 - Cerebro X Representation Model", "Created patient-specific latent brain state extractors using MRI and clinical encoders."),
        ("Phase 5 - Longitudinal Digital Twin (GRU)", "Turned the static representation into a temporal model (GRU) predicting future decline using sequential visit history."),
        ("Phase 6 - Explainability", "Implemented SHAP for clinical feature attribution and Grad-CAM for MRI to understand model behavior."),
        ("Phase 7 - Cross-Dataset Generalization", "Evaluated Cerebro-X on independent cohorts to measure performance degradation and shift."),
        ("Phase 8 - Research API", "Exposed the trained models through a FastAPI application with prediction and explanation endpoints."),
        ("Phase 9 - Frontend Dashboard", "Built a Next.js research dashboard featuring an MRI viewer, longitudinal timeline, explainability views, and patient overview."),
        ("Phase 10 - Database and Experiment Management", "Set up PostgreSQL to persist research outputs (predictions, explanation metadata, and experiment records)."),
        ("Phase 11 - Validation and Robustness", "Stress-tested the system against missing values, corrupted scans, and class imbalance. Result: 39 robustness tests + 75 unit tests (114 total) passing."),
        ("Phase 12 - Thesis/Paper Preparation", "Compiled the implementation, ablations, baseline comparisons, and explainability into a full research report/thesis."),
        ("Phase 13 - Deployment", "Packaged the research prototype using Docker (React -> FastAPI -> Model Service -> PostgreSQL)."),
        ("Phase 14 - Final Research Audit", "Verified dataset provenance, no leakage, reproducible seeds, and documented all software versions and limitations.")
    ]

    for phase, desc in phases:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(phase + ": ").bold = True
        p.add_run(desc)

    doc.add_heading('6. Hand-off Manual / Detailed Documentation', level=1)
    
    handoff_path = 'CEREBRO_X_HANDOFF_MANUAL.docx'
    if os.path.exists(handoff_path):
        doc.add_paragraph("The following content is imported from the original Handoff Manual for completeness:")
        try:
            old_doc = docx.Document(handoff_path)
            for paragraph in old_doc.paragraphs:
                if paragraph.text.strip():
                    doc.add_paragraph(paragraph.text)
        except Exception as e:
            doc.add_paragraph(f"[Error reading existing handoff manual: {e}]")
    else:
        doc.add_paragraph("No existing CEREBRO_X_HANDOFF_MANUAL.docx found in the directory.")

    output_path = 'CEREBRO-X_Complete_Project_Documentation.docx'
    doc.save(output_path)
    print(f"Documentation saved successfully to {output_path}")

if __name__ == '__main__':
    create_documentation()
