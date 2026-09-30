import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

ROOT = Path("E:/PROJECTS/CEREBRO-X")
OUT_DIR = ROOT / "CEREBRO_X_RESULTS_VISUALS"

dirs = [
    "01_Clinical_GRU",
    "02_MRI_Feature_Analysis",
    "03_EEG_Standalone",
    "04_Clinical_MRI_Fusion",
    "05_Model_Comparison",
    "06_Digital_Brain_Twin",
    "07_Explainability",
    "08_Final_Results_Overview"
]

for d in dirs:
    (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

def create_bar_chart(title, labels, values, ylabel, filename, palette="crest"):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=values, palette=palette)
    plt.title(title, pad=20)
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# 1. Clinical GRU
def gen_clinical():
    metrics = {"Accuracy": 75.0, "Balanced Acc": 55.7, "Macro F1": 55.1}
    create_bar_chart(
        "Clinical-Only Longitudinal Prediction Results",
        list(metrics.keys()),
        list(metrics.values()),
        "Percentage (%)",
        OUT_DIR / "01_Clinical_GRU/clinical_metrics.png"
    )
    
    # Confusion Matrix Placeholder (since real labels/preds aren't loaded here easily)
    # We will simulate the exact distribution that gives 75% acc based on earlier metrics.
    # But user said NO FABRICATION of confusion matrices. We will just output the metrics bar chart.

# 2. MRI Feature Analysis
def gen_mri_features():
    # If the real 3D CNN model is not trained: create "3D MRI CNN Performance: TRAINING PENDING"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, "3D MRI CNN Performance:\nTRAINING PENDING", 
            ha='center', va='center', fontsize=20, color='red', weight='bold')
    ax.axis('off')
    plt.savefig(OUT_DIR / "02_MRI_Feature_Analysis/mri_cnn_pending.png", dpi=300)
    plt.close()

# 3. EEG Standalone
def gen_eeg():
    metrics = {"Accuracy": 50.0, "Balanced Acc": 47.8, "Macro F1": 47.6}
    create_bar_chart(
        "Standalone EEG Screening Results\n(Independent cohort - Not patient-level fused)",
        list(metrics.keys()),
        list(metrics.values()),
        "Percentage (%)",
        OUT_DIR / "03_EEG_Standalone/eeg_metrics.png",
        palette="viridis"
    )

# 4. Clinical + MRI Fusion
def gen_bimodal():
    metrics = {"Accuracy": 71.4, "Balanced Acc": 52.9}
    create_bar_chart(
        "Clinical + MRI Multimodal Fusion Results\n(Legacy Scalar Feature Fusion)",
        list(metrics.keys()),
        list(metrics.values()),
        "Percentage (%)",
        OUT_DIR / "04_Clinical_MRI_Fusion/bimodal_metrics.png",
        palette="magma"
    )

# 5. Model Comparison
def gen_comparison():
    models = ["Clinical Only (GRU)", "Clinical+MRI (Scalar)", "EEG Only", "MRI 3D CNN"]
    accs = [75.0, 71.4, 50.0, 0] # 0 = Pending
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=models, y=accs, palette="deep")
    plt.title("CEREBRO X RESULTS - Modality Comparison", pad=20)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    
    for i, v in enumerate(accs):
        if v == 0:
            plt.text(i, 5, "Training\nPending", ha='center', va='bottom', color='red', weight='bold')
        else:
            plt.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_Model_Comparison/multimodal_comparison.png", dpi=300)
    plt.close()

# 8. Final Overview
def gen_overview():
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    plt.text(0.5, 0.9, "Cerebro X — Experimental Results Overview", 
             ha='center', va='center', fontsize=24, weight='bold', color='#1A5376')
             
    # Draw boxes
    box_props = dict(boxstyle="round,pad=1", facecolor="#EAF2F8", edgecolor="#1A5376", lw=2)
    
    plt.text(0.25, 0.65, "1. Clinical-Only\n75.0% Accuracy", ha='center', va='center', fontsize=16, bbox=box_props)
    plt.text(0.75, 0.65, "2. Clinical + MRI (Scalar)\n71.4% Accuracy", ha='center', va='center', fontsize=16, bbox=box_props)
    plt.text(0.25, 0.40, "3. EEG-Only\n50.0% Accuracy", ha='center', va='center', fontsize=16, bbox=box_props)
    
    box_pending = dict(boxstyle="round,pad=1", facecolor="#FADBD8", edgecolor="#943126", lw=2)
    plt.text(0.75, 0.40, "4. MRI 3D CNN\nTraining Pending", ha='center', va='center', fontsize=16, bbox=box_pending)
    
    plt.text(0.5, 0.20, "CURRENT MULTIMODAL CAPABILITY\nClinical + MRI scalar feature fusion", 
             ha='center', va='center', fontsize=14, color='#1A5376', weight='bold')
             
    plt.text(0.5, 0.08, "FUTURE WORK\nPatient-aligned Clinical + MRI + EEG → True tri-modal fusion", 
             ha='center', va='center', fontsize=14, color='#6E2C00', style='italic')
             
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_Final_Results_Overview/final_presentation_figure.png", dpi=300)
    plt.close()

def build_summary():
    content = """# CEREBRO X RESULTS SUMMARY

## 01_Clinical_GRU/clinical_metrics.png
- **Dataset**: OASIS-2 Longitudinal
- **Number of subjects**: 150
- **Model**: TemporalCerebroNet (GRU)
- **Input features**: 19 Clinical Features
- **Metric**: Accuracy (75.0%), Balanced Acc (55.7%), Macro F1 (55.1%)
- **Source**: artifacts/manifest.json (EXP-LONGITUDINAL-001)
- **Status**: Implemented

## 02_MRI_Feature_Analysis/mri_cnn_pending.png
- **Dataset**: OASIS-2 MRI
- **Model**: Lightweight3DCNN
- **Metric**: N/A
- **Status**: Pending Real Data Training (Explicitly declared as per instructions)

## 03_EEG_Standalone/eeg_metrics.png
- **Dataset**: OpenNeuro ds004504
- **Number of subjects**: 87
- **Model**: EEGSpectralEncoder
- **Input features**: Spectral power bands
- **Metric**: Accuracy (50.0%)
- **Status**: Implemented (Independent cohort)

## 04_Clinical_MRI_Fusion/bimodal_metrics.png
- **Dataset**: OASIS-2
- **Model**: LegacyScalarBimodalCerebroNet
- **Input features**: Clinical + MRI scalar features (nWBV, eTIV)
- **Metric**: Accuracy (71.4%)
- **Status**: Implemented (Legacy)

## 05_Model_Comparison/multimodal_comparison.png
- **Dataset**: Multiple
- **Models**: GRU, Legacy Fusion, EEG Encoder, CNN3D
- **Metrics**: Accuracy comparison across all validated phases
- **Status**: Implemented / Pending where applicable

## 08_Final_Results_Overview/final_presentation_figure.png
- **Dataset**: Global
- **Status**: High-level presentation infographic.
"""
    with open(ROOT / "CEREBRO_X_RESULTS_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    gen_clinical()
    gen_mri_features()
    gen_eeg()
    gen_bimodal()
    gen_comparison()
    gen_overview()
    build_summary()
    print("Visualizations successfully generated.")
