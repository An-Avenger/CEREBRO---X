# CEREBRO X RESULTS SUMMARY

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
