"""
Router: CDR prediction endpoints.
Exposes clinical GRU and bimodal fusion models.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from cerebro_x.api.database import get_db
from cerebro_x.api.models import Patient, PredictionRecord
from cerebro_x.api.schemas.patient import (
    PredictionRequest,
    BimodalPredictionRequest,
    PredictionResponse,
)
from cerebro_x.api.services.model_loader import get_registry
from cerebro_x.api.services.inference import run_clinical_inference, run_bimodal_inference
from cerebro_x.features.preprocessor import CDR_CLASS_TO_VALUE, CDR_CLASS_TO_LABEL

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/clinical",
    response_model=PredictionResponse,
    summary="Predict next-visit CDR (Clinical GRU)",
    description=(
        "Predicts the next-visit CDR class using the trained longitudinal GRU model "
        "(EXP-LONGITUDINAL-001). Accepts one or more chronological visit records. "
        "Test accuracy: 75.0% | Balanced accuracy: 55.7% | Dataset: OASIS-2 (150 subjects)."
    ),
)
def predict_clinical(
    request: PredictionRequest,
    registry=Depends(get_registry),
    db: Session = Depends(get_db),
):
    if not registry.status.get("clinical_gru"):
        raise HTTPException(503, "Clinical GRU model not loaded. Check artifacts/EXP-LONGITUDINAL-001/")

    visits = [v.model_dump() for v in request.visits]
    preprocessor = registry.scalers.get("clinical_preprocessor")

    try:
        result = run_clinical_inference(
            registry.models["clinical_gru"], visits, preprocessor=preprocessor
        )
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    # Persist only identified patients.  ``subject_id`` is intentionally
    # optional in the public schema to support one-off, de-identified research
    # predictions; inserting ``None`` into the Patient primary key would make
    # those otherwise valid requests fail at commit time.
    if request.subject_id:
        patient = db.query(Patient).filter(Patient.id == request.subject_id).first()
        if not patient:
            patient = Patient(id=request.subject_id)
            db.add(patient)

        last_visit = request.visits[-1]
        record = PredictionRecord(
            patient_id=request.subject_id,
            age=last_visit.age,
            educ=last_visit.educ,
            ses=last_visit.ses,
            mmse=last_visit.mmse,
            cdr=last_visit.cdr,
            nwbv=last_visit.nwbv,
            etiv=last_visit.etiv,
            asf=last_visit.asf,
            predicted_cdr_class=result["predicted_cdr_class"],
            predicted_cdr_value=result["predicted_cdr_value"],
            predicted_cdr_label=result["predicted_cdr_label"],
            class_probabilities=result["class_probabilities"],
            apoe4=last_visit.apoe4,
            p_tau=last_visit.p_tau,
        )
        db.add(record)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise HTTPException(500, "Could not save the identified patient prediction.")

    return PredictionResponse(subject_id=request.subject_id, **result)


@router.post(
    "/bimodal",
    response_model=PredictionResponse,
    summary="Predict next-visit CDR (Clinical + MRI Scalar Bimodal Fusion)",
    description=(
        "Predicts the next-visit CDR class using the bimodal fusion model that combines "
        "clinical visit history (GRU) with MRI-derived scalars (nWBV, eTIV, ASF). "
        "STATUS: LEGACY SCALAR EXPERIMENT. "
        "MRI inputs are OASIS-2 scalar features — NOT raw NIfTI CNN embeddings. "
        "(EXP-FUSION-BIMODAL-001). Test accuracy: 71.4% | Balanced accuracy: 52.9%"
    ),
)
def predict_bimodal(
    request: BimodalPredictionRequest,
    registry=Depends(get_registry),
):
    if not registry.status.get("bimodal_fusion"):
        raise HTTPException(503, "Bimodal fusion model not loaded. Check artifacts/EXP-FUSION-BIMODAL-001/")

    visits = [v.model_dump() for v in request.visits]
    mri = request.mri.model_dump()
    preprocessor = registry.scalers.get("clinical_preprocessor")

    try:
        result = run_bimodal_inference(
            registry.models["bimodal_fusion"], visits, mri, preprocessor=preprocessor
        )

        # Apply NextGen Biomarker Adjustment if provided (APOE4 / p-tau)
        last_visit = request.visits[-1]
        if last_visit.apoe4 is not None or last_visit.p_tau is not None:
            probs = dict(result["class_probabilities"])
            risk_multiplier = 1.0
            if last_visit.apoe4:
                risk_multiplier += 0.15
            if last_visit.p_tau is not None and last_visit.p_tau > 21.7:
                risk_multiplier += 0.20

            if risk_multiplier > 1.0:
                shift = min(probs["0"] * (risk_multiplier - 1.0), probs["0"])
                probs["0"] -= shift
                probs["1"] += shift * 0.5
                probs["2"] += shift * 0.3
                probs["3"] += shift * 0.2
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}

                pred_class_str = max(probs, key=probs.get)
                pred_class = int(pred_class_str)
                result["class_probabilities"] = probs
                result["predicted_cdr_class"] = pred_class
                result["predicted_cdr_value"] = CDR_CLASS_TO_VALUE[pred_class]
                result["predicted_cdr_label"] = CDR_CLASS_TO_LABEL[pred_class]

    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    result["model"] += " (Legacy Scalar Experiment)"
    
    return PredictionResponse(subject_id=request.subject_id, **result)


@router.post(
    "/export-dossier-html",
    summary="Export Clinical Dossier Report (Printable HTML / PDF)",
    description="Generates an executive, print-ready clinical research dossier with patient demographics, longitudinal predictions, MRI ROI attributions, and XAI charts.",
)
def export_dossier_html(payload: dict):
    from fastapi.responses import HTMLResponse
    import datetime

    subject_id = payload.get("subject_id", "DEMO-PATIENT-001")
    diagnosis = payload.get("predicted_class", "MCI")
    confidence = payload.get("confidence", 84.5)
    probs = payload.get("probabilities", {"CN": 0.05, "MCI": 0.85, "AD": 0.10})
    roi_breakdown = payload.get("roi_breakdown", [
        {"region": "Medial Temporal / Hippocampus", "importance_pct": 44.2},
        {"region": "Lateral Ventricles & Periventricular", "importance_pct": 28.6},
        {"region": "Cortical Grey Matter & Sulci", "importance_pct": 14.8},
        {"region": "Frontal Lobe & Anterior Network", "importance_pct": 7.4},
        {"region": "Posterior Parietal & Occipital", "importance_pct": 5.0},
    ])
    vis_img = payload.get("visualization_base64", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    diag_color = "#ef4444" if diagnosis == "AD" else "#f59e0b" if diagnosis == "MCI" else "#10b981"

    roi_rows = "".join([
        f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-weight: 500;">{r.get('region')}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: 600; color: #3b82f6;">{r.get('importance_pct')}%</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e2e8f0;">
                <div style="background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden; width: 120px; margin-left: auto;">
                    <div style="background: #3b82f6; height: 100%; width: {r.get('importance_pct')}%;"></div>
                </div>
            </td>
        </tr>
        """ for r in roi_breakdown
    ])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Cerebro-X Clinical Dossier - {subject_id}</title>
        <style>
            @media print {{
                .no-print {{ display: none; }}
                body {{ -webkit-print-color-adjust: exact; }}
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: #1e293b;
                background: #f8fafc;
                margin: 0;
                padding: 40px 20px;
            }}
            .container {{
                max-width: 860px;
                margin: 0 auto;
                background: #ffffff;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.06);
                border: 1px solid #e2e8f0;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                border-bottom: 2px solid #0f172a;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 14px;
                border-radius: 9999px;
                font-size: 14px;
                font-weight: 700;
                color: #ffffff;
                background: {diag_color};
            }}
            .card {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 24px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-bottom: 24px;
            }}
            .stat-box {{
                background: #f1f5f9;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }}
            .btn-print {{
                background: #0ea5e9;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: 600;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1 style="margin: 0; font-size: 26px; color: #0f172a;">CEREBRO-X DIGITAL BRAIN TWIN</h1>
                    <p style="margin: 4px 0 0 0; color: #64748b; font-size: 13px;">Multimodal AI Neurological Diagnostic Dossier</p>
                </div>
                <div style="text-align: right;">
                    <button class="btn-print no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>
                    <p style="margin: 6px 0 0 0; font-size: 12px; color: #94a3b8;">Generated: {timestamp}</p>
                </div>
            </div>

            <div class="card" style="border-left: 5px solid {diag_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b;">Target Patient Identification</div>
                        <h2 style="margin: 4px 0 0 0; font-size: 20px; color: #0f172a;">Patient Ref: {subject_id}</h2>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge">Predicted CDR: {diagnosis}</span>
                        <div style="font-size: 13px; color: #64748b; margin-top: 4px;">Confidence: {confidence}%</div>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="stat-box">
                    <div style="font-size: 12px; color: #64748b;">Cognitive Normal (CN)</div>
                    <div style="font-size: 18px; font-weight: 700; color: #10b981; margin-top: 4px;">{(probs.get('CN', 0)*100):.1f}%</div>
                </div>
                <div class="stat-box">
                    <div style="font-size: 12px; color: #64748b;">Mild Cognitive Impairment (MCI)</div>
                    <div style="font-size: 18px; font-weight: 700; color: #f59e0b; margin-top: 4px;">{(probs.get('MCI', 0)*100):.1f}%</div>
                </div>
                <div class="stat-box">
                    <div style="font-size: 12px; color: #64748b;">Alzheimer's Dementia (AD)</div>
                    <div style="font-size: 18px; font-weight: 700; color: #ef4444; margin-top: 4px;">{(probs.get('AD', 0)*100):.1f}%</div>
                </div>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; font-size: 16px; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
                    🧠 Anatomical Region Saliency & Attribution (Atlas Mapping)
                </h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <thead>
                        <tr style="text-align: left; color: #64748b; border-bottom: 1px solid #cbd5e1;">
                            <th style="padding: 8px 12px;">Neuroanatomical Target</th>
                            <th style="padding: 8px 12px; text-align: right;">Saliency Weight</th>
                            <th style="padding: 8px 12px; text-align: right;">Impact Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {roi_rows}
                    </tbody>
                </table>
            </div>

            {f'''
            <div class="card" style="text-align: center;">
                <h3 style="margin-top: 0; font-size: 16px; color: #0f172a; margin-bottom: 16px;">
                    🔬 Multimodal MRI SHAP & Integrated Gradients Heatmap
                </h3>
                <img src="{vis_img}" style="max-width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;" />
            </div>
            ''' if vis_img else ''}

            <div style="font-size: 11px; color: #94a3b8; text-align: center; margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 16px;">
                CEREBRO-X Research Prototype · Certified for Longitudinal Alzheimer's Disease Prediction Research Only · Not a medical diagnostic device.
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
