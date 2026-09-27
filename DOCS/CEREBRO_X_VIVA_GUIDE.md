# CEREBRO-X VIVA & INTERVIEW GUIDE

Use these questions and answers to prepare for project presentations, academic vivas, and technical interviews.

---

## PROJECT BASICS

**Q: What is Cerebro-X?**
A: Cerebro-X is an AI-powered Digital Brain Twin framework designed to predict the progression of Alzheimer's Disease using longitudinal clinical data.

**Q: What is a Digital Brain Twin?**
A: In our architecture, the "Brain Twin" is the learned latent representation (denoted as `Z_t`) of the patient’s longitudinal neurological state. It is not a literal 3D video game brain, but a mathematical embedding that tracks how a patient's cognitive profile changes over time.

**Q: Why Alzheimer’s?**
A: Alzheimer's is a progressive disease where historical trajectory is a much stronger predictor of future decline than a single cross-sectional snapshot.

**Q: Why OASIS-2?**
A: OASIS-2 is a standard open-access longitudinal MRI and clinical dataset, making it ideal for temporal modeling.

**Q: What is longitudinal data?**
A: Data collected from the same subjects repeatedly over time.

**Q: What is CDR?**
A: Clinical Dementia Rating. It is a numeric scale used to quantify the severity of dementia symptoms.

---

## DEEP LEARNING

**Q: Why GRU?**
A: Gated Recurrent Units (GRUs) are excellent at modeling sequential data. They maintain a hidden state that updates with each new visit, capturing the patient's historical trajectory.

**Q: Why not simple MLP?**
A: A Multi-Layer Perceptron (MLP) only looks at the current visit, ignoring the crucial rate of cognitive decline that occurs between visits.

**Q: What is hidden state?**
A: The internal memory of the GRU that retains information from previous timesteps.

**Q: What is Z_t?**
A: It is the 64-dimensional output vector from the GRU representing the patient's learned state at time `t`.

**Q: What is an embedding?**
A: A dense vector representation of data where similar clinical states are mapped closer together in the mathematical space.

**Q: What is the output layer?**
A: A linear layer (`nn.Linear`) that maps the 64-dimensional embedding to 4 output logits corresponding to the CDR classes.

**Q: Why 4 classes?**
A: The OASIS-2 dataset CDR values group naturally into 4 categories: 0 (None), 0.5 (Very Mild), 1 (Mild), and 2 (Moderate).

**Q: How was data leakage prevented?**
A: By strictly structuring the dataset into "current visit → next visit" pairs. The model is never allowed to see future visit features when predicting the future CDR.

---

## MRI

**Q: Why NIfTI?**
A: NIfTI (`.nii`) is the standard format for neuroimaging, storing raw 3D voxel data and affine orientation matrices.

**Q: Why 3D CNN?**
A: Because MRI scans are volumetric. A 2D CNN would lose spatial depth context between slices.

**Q: Why 64×64×64?**
A: Raw MRI scans are massive (e.g., 256³). Downsampling to 64³ retains macroscopic structural atrophy patterns while fitting in standard GPU memory for faster training.

**Q: What is voxel preprocessing?**
A: Our pipeline loads the file via `nibabel`, forces standard RAS orientation, clips extreme intensity outliers, normalizes the brain tissue to a standard Z-score, and resizes the volume.

**Q: What is Grad-CAM?**
A: Gradient-weighted Class Activation Mapping. It uses the gradients flowing into the final convolutional layer to highlight which regions of the brain most influenced the prediction.

**Q: Why does Grad-CAM require a trained CNN?**
A: Grad-CAM relies on the learned weights of the model. Running it on an untrained (randomly initialized) model would produce random, meaningless heatmaps.

---

## MULTIMODAL

**Q: What is multimodal learning?**
A: Combining different types of data (e.g., tabular clinical data + 3D imaging data + 1D EEG signal) into a single predictive model.

**Q: How are clinical and MRI features fused?**
A: Currently, we use late fusion. The clinical data generates a 64-D vector, the scalar MRI features generate a 32-D vector, and they are concatenated into a 96-D vector before final classification.

**Q: Why is current scalar MRI fusion different from raw MRI CNN fusion?**
A: The scalar fusion relies on 4 simple numbers (like total brain volume). The raw CNN extracts complex spatial patterns directly from the pixels. We built the CNN pipeline, but are waiting on training data to swap the scalar branch for the CNN branch.

---

## EVALUATION

**Q: What is accuracy vs balanced accuracy?**
A: Accuracy is total correct divided by total samples. Balanced accuracy averages the recall across all classes, preventing the model from looking "good" just by predicting the majority class in an imbalanced dataset.

**Q: What is macro F1?**
A: The unweighted average of the F1 scores for each class. It treats all classes equally, punishing models that ignore minority classes.

**Q: What are the limitations of 150 subjects?**
A: Deep learning models are data-hungry. 150 subjects is extremely small, increasing the risk of severe overfitting.

---

## SYSTEM

**Q: Why FastAPI and Next.js?**
A: FastAPI is standard for Python ML serving due to its speed and asynchronous capabilities. Next.js provides a robust, component-driven UI for clinical dashboards.

**Q: Where are models loaded?**
A: They are loaded into memory (`model_loader.py`) as singletons when the FastAPI server starts, preventing the need to reload large `.pt` files on every request.

---

## LIMITATIONS

**Q: What is incomplete?**
A: We have not yet trained the raw 3D CNN on the MRI volumes because the local dataset lacks the raw `.nii` files. Therefore, live MRI Grad-CAM and true voxel-level multimodal fusion are currently scaffolded but waiting for weights. 

**Q: Why is this not a clinical diagnostic system?**
A: It is a research prototype trained on a tiny, highly controlled cohort. It lacks FDA approval, external cross-cohort validation (e.g., on ADNI or OASIS-3), and longitudinal robustness guarantees.
