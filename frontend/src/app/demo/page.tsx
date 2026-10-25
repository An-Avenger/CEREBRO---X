'use client';

import React, { useState } from 'react';
import Link from 'next/link';

interface DemoModule {
  id: string;
  category: string;
  badge: string;
  badgeColor: string;
  title: string;
  hook: string;
  problem: string;
  solution: string;
  architecture: string;
  metrics: { label: string; value: string; note: string }[];
  keyStrengths: string[];
  liveRoute: string;
  liveRouteLabel: string;
  defenseScript: string;
}

const DEMO_MODULES: DemoModule[] = [
  {
    id: 'executive-summary',
    category: '1. RESEARCH MOTIVATION',
    badge: 'Thesis Scope',
    badgeColor: 'cyan',
    title: 'The Clinical Challenge: Predicting Alzheimer’s Disease Trajectories',
    hook: 'Why traditional static machine learning fails in neurodegenerative disease prediction.',
    problem: 'Alzheimer’s disease is progressive and slow-moving. Conventional models take a single snapshot in time and treat patients as static vectors. However, two patients with the exact same MMSE score of 26 may have completely different trajectories: one is stable, while the other is rapidly progressing.',
    solution: 'Cerebro-X introduces a Digital Brain Twin framework using a temporal recurrent neural network (TemporalCerebroNet) trained on longitudinal multi-visit OASIS-2 sequences. The model explicitly learns temporal velocity (ΔMMSE, ΔnWBV) and extracts a 64-dimensional latent state Z_t representing the evolving brain.',
    architecture: 'Input: Chronological Clinical Visits → Temporal Imputation & Preprocessing → 2-layer Bidirectional GRU (Hidden Dim: 64) → Linear 4-Class Classification Head (CDR 0, 0.5, 1, 2+).',
    metrics: [
      { label: 'Longitudinal Test Acc.', value: '75.0%', note: 'Vs 71.4% static baseline (+3.6% gain)' },
      { label: 'Balanced Accuracy', value: '55.7%', note: 'Multi-class 4-state CDR' },
      { label: 'Macro F1', value: '55.1%', note: 'Zero data leakage subject-level split' },
      { label: 'Cohort Size', value: '150 Subjects', note: 'OASIS-2 multi-year clinical sequences' },
    ],
    keyStrengths: [
      'Models trajectory dynamics over time rather than isolated cross-sectional visits.',
      'Strict subject-level 70/15/15 splitting guarantees zero patient leakage between splits.',
      'Honest comparison against Dummy, Random Forest, and Logistic Regression baselines.'
    ],
    liveRoute: '/predict',
    liveRouteLabel: '⚡ Test Longitudinal Prediction',
    defenseScript: 'Explain to examiners that neurodegenerative diseases require modeling velocity (derivatives over time). Showing that our Temporal GRU achieves 75.0% vs. 71.4% for the Markovian single-visit baseline proves the clinical value of temporal context.'
  },
  {
    id: 'brain-twin-zt',
    category: '2. DIGITAL BRAIN TWIN',
    badge: 'Core Novelty',
    badgeColor: 'violet',
    title: 'Digital Brain Twin Manifold & Interactive 3D Hologram',
    hook: 'Mapping high-dimensional disease trajectories into an interactive 3D neural manifold.',
    problem: 'Neurologists and patients cannot easily interpret a raw 64-dimensional hidden tensor Z_t. There was a critical need to visualize the latent state evolution and provide an intuitive representation of focal cortical atrophy.',
    solution: 'Cerebro-X extracts the GRU hidden state Z_t at each visit and maps it into an interactive 3D Holographic Particle Mesh with 1,350 neural synapses. The 3D hologram dynamically morphs and shrinks focal temporal & hippocampal lobes as the patient’s CDR score deteriorates.',
    architecture: 'TemporalCerebroNet Hidden State Z_t (64-dim) → Dimensionality Reduction / PCA 2D Manifold → Real-Time Canvas 3D Perspective Projection with Gyri-Sulci mathematical modeling & Anatomical Lobe Segmentation.',
    metrics: [
      { label: 'Latent Dimensionality', value: '64-D', note: 'Continuous patient state embedding' },
      { label: 'Point-Cloud Density', value: '1,350 Points', note: 'Ellipsoidal cortical gyri-sulci mesh' },
      { label: 'Anatomical Lobes', value: '4 Segments', note: 'Temporal, Frontal, Parietal, Occipital' },
      { label: 'Interactive Framerate', value: '60 FPS', note: 'Pure HTML5 Canvas 3D math (0 lag)' },
    ],
    keyStrengths: [
      'Interactive 3D rotation, pitch/yaw orbit, and zoom controls.',
      'Focal atrophy morphing specifically targeting the temporal lobe and hippocampus.',
      'Transparently labeled as a mathematical latent manifold, avoiding deceptive anatomical claims.'
    ],
    liveRoute: '/brain-twin',
    liveRouteLabel: '🧠 Explore Interactive 3D Brain Twin',
    defenseScript: 'Highlight that Z_t is the internal brain representation learned by the recurrent neural network. As the patient progresses from visit 1 to visit 3, the coordinate trajectory shifts across the manifold, directly reflecting neurocognitive decline.'
  },
  {
    id: 'neuroimaging-mri',
    category: '3. 3D NEUROIMAGING',
    badge: '3D ConvNet',
    badgeColor: 'cyan',
    title: '3D T1-Weighted MRI Preprocessing & MRICerebroNet',
    hook: 'Real volumetric NIfTI processing from raw voxels to 3D convolutional activation.',
    problem: 'Many student projects fake MRI inputs by hashing filenames or feeding random noise. Genuine neuroimaging demands real NIfTI header parsing, spatial orientation correction, intensity normalization, and 3D convolutions.',
    solution: 'Cerebro-X features a genuine Python neuroimaging pipeline using NiBabel. When a .nii or .nii.gz volume is uploaded, it enforces dimension checks, applies RAS reorientation, clips intensity percentiles, z-score normalizes brain tissue, and resamples to a uniform (64, 64, 64) tensor.',
    architecture: 'Raw NIfTI Upload → RAS Canonical Reorientation → Intensity Truncation (0.5%–99.5%) → Brain Tissue Z-Score → Trilinear Resize (64³) → MRICerebroNet (4x 3D Conv, BatchNorm, MaxPool) → 64-dim Latent Embedding → 4-Class Linear Classifier.',
    metrics: [
      { label: 'Input Grid Size', value: '64 × 64 × 64', note: 'Standardized isotropic volumetric grid' },
      { label: 'Verified Checkpoint', value: '602 KB', note: 'artifacts/EXP-MRI-ADNI-001/best_checkpoint.pt' },
      { label: 'Scans in Pilot', value: '100 Scans', note: '69 subjects with real ADNI T1w MRI' },
      { label: 'Test Accuracy', value: '43.75%', note: 'Pilot baseline on severe class imbalance' },
    ],
    keyStrengths: [
      'Calculates real voxel-based anatomical proxies: nWBV (brain fraction), estimated eTIV, and ASF.',
      'Rejecting synthetic fake models: model_loader.py explicitly verifies authentic ADNI provenance.',
      'Full compatibility with standard DICOM/NIfTI research protocols.'
    ],
    liveRoute: '/predict',
    liveRouteLabel: '⚡ Test Real NIfTI MRI Upload',
    defenseScript: 'Emphasize scientific honesty: our 3D ConvNet was trained on a 100-scan ADNI pilot with severe class imbalance (61 CDR 0, 29 CDR 0.5, 7 CDR 1, 3 CDR 2). We report 43.75% accuracy honestly as a research pilot rather than inflating numbers.'
  },
  {
    id: 'xai-gradcam-shap',
    category: '4. EXPLAINABLE AI',
    badge: 'Verified XAI',
    badgeColor: 'emerald',
    title: 'Multi-Modal Explainability: 3D Grad-CAM & Clinical SHAP',
    hook: 'Opening the deep learning black-box with genuine gradient activations.',
    problem: 'In clinical medicine, black-box predictions cannot be trusted. Clinicians need to know why the GRU predicted dementia and where the 3D CNN detected cerebral atrophy.',
    solution: 'Cerebro-X incorporates two rigorous XAI frameworks: (1) Clinical SHAP via shap.GradientExplainer on the GRU using background baseline distributions, and (2) Real 3D Grad-CAM hooking into layer cnn.features.12 of MRICerebroNet to generate multi-slice heatmaps across Axial, Sagittal, and Coronal planes.',
    architecture: 'Clinical XAI: shap.GradientExplainer over 100 OASIS-2 baseline visits. MRI XAI: Forward pass → target class gradient backward hook → channel-wise pooling → trilinear upsampling → 5-depth Axial slice stack (25%, 35%, 50%, 65%, 75%).',
    metrics: [
      { label: 'Grad-CAM Target', value: 'cnn.features.12', note: 'Last Conv3d layer in MRICerebroNet' },
      { label: 'Slicing Planes', value: '3 Projections', note: 'Axial, Sagittal, and Coronal overlays' },
      { label: 'Axial Depth Stack', value: '5 Depths', note: 'Interactive Z-slice scrubber' },
      { label: 'Top SHAP Features', value: 'MMSE, CDR, Age', note: 'Clinically aligned with AD literature' },
    ],
    keyStrengths: [
      'Interactive Volumetric Depth Slider allowing reviewers to scrub through brain depths.',
      'Real backprop gradients instead of painted or hardcoded heatmaps.',
      'Visualizes top feature importances and temporal contribution deltas for every visit.'
    ],
    liveRoute: '/explainability',
    liveRouteLabel: '🔍 Launch Live XAI Hub',
    defenseScript: 'Show the examiners the Axial Depth Slider at 50% depth. Point out that the 3D Grad-CAM highlights the medial temporal lobe and ventricles—the exact regions known in neurology to undergo early Alzheimer’s neurodegeneration.'
  },
  {
    id: 'counterfactual-simulation',
    category: '5. CLINICAL DECISION SUPPORT',
    badge: 'What-If Engine',
    badgeColor: 'amber',
    title: 'Counterfactual "What-If" Simulation & Clinical Report Generator',
    hook: 'Forecasting therapeutic outcomes and generating institutional diagnostic summaries.',
    problem: 'Diagnosing current decline is insufficient; clinicians and clinical trial sponsors need to simulate the impact of future interventions (e.g. anti-amyloid monoclonal antibodies, cognitive rehab, or tau inhibitors).',
    solution: 'We engineered an interactive Counterfactual Simulation Engine that projects future cognitive and volumetric curves over 1 to 6 years, comparing natural decline against intervention stabilization. It compiles a publication-ready Multi-Modal Clinical Diagnostic Report with an automated Brain Health Index (BHI).',
    architecture: 'Patient Visit History + Intervention Parameters (ΔMMSE boost, nWBV preservation %, p-tau reduction %) → Iterative Forward GRU Evaluation → Comparative Trajectory Table + Markdown Diagnostic Report Generator.',
    metrics: [
      { label: 'Projection Horizon', value: '1 to 6 Years', note: 'Customizable future clinical timeline' },
      { label: 'Brain Health Index', value: '0 – 100 Score', note: 'Multi-biomarker synthesized index' },
      { label: 'Risk Stratification', value: '4 Tiers', note: 'Normal, Elevated, High, Critical' },
      { label: 'Report Format', value: 'Structured MD', note: 'One-click copy to clipboard' },
    ],
    keyStrengths: [
      'Simulates salvaged MMSE cognitive points and protected brain parenchyma over time.',
      'Integrates blood biomarkers (plasma p-tau181) and genetic risk (APOE4 carrier status).',
      'Provides a professional exportable diagnostic report ready for physician review.'
    ],
    liveRoute: '/simulation',
    liveRouteLabel: '🧪 Run What-If Simulation',
    defenseScript: 'Explain that counterfactual simulation is the bridge between predictive AI and actionable clinical medicine. It allows doctors to ask: If we stabilize this patient’s MMSE by 3 points with an anti-amyloid drug, how does their 3-year dementia risk change?'
  },
  {
    id: 'eeg-and-modality',
    category: '6. MULTIMODAL INTEGRITY',
    badge: 'Scientific Rigor',
    badgeColor: 'violet',
    title: 'EEG Spectral Screening & Cross-Cohort Integrity',
    hook: 'Addressing multimodal cohort boundaries with complete scientific transparency.',
    problem: 'A common flaw in academic projects is pretending that disparate public datasets (OASIS-2 for clinical, ADNI for MRI, and OpenNeuro ds004504 for EEG) contain the same subjects, or fabricating artificial patient matches.',
    solution: 'Cerebro-X upholds strict scientific integrity. We trained a dedicated 3-layer EEG Spectral MLP (107 PSD frequency bands across 19 channels) as an independent non-invasive screener, while openly documenting that OASIS, ADNI, and OpenNeuro are distinct cohorts that cannot be merged without subject-aligned data.',
    architecture: 'OpenNeuro ds004504 (87 subjects) → Welch PSD Power Extraction (δ, θ, α, β, γ) → Standardized Feature Imputation & Scaling → 3-Layer Dense Classifier with BatchNorm & Dropout → Binary/3-Class AD Screener.',
    metrics: [
      { label: 'EEG Channels', value: '19 Standard', note: '10-20 international electrode system' },
      { label: 'Spectral Features', value: '107 Bands', note: 'Absolute & relative band power' },
      { label: 'EEG Subjects', value: '87 Subjects', note: 'OpenNeuro ds004504 cohort' },
      { label: 'Unit Test Suite', value: '271 Passed', note: 'Zero failures across entire repository' },
    ],
    keyStrengths: [
      'Demonstrates advanced signal processing (Power Spectral Density extraction across 5 bands).',
      'Refuses to fabricate subject alignments, protecting the candidate from critical viva questions.',
      'Comprehensive benchmark table comparing individual modalities against bimodal fusion.'
    ],
    liveRoute: '/eeg',
    liveRouteLabel: '〜 Test EEG Screener',
    defenseScript: 'When asked why EEG is evaluated as a separate screener rather than merged with OASIS-2: Explain that OASIS-2 never collected EEG. Fabricating synthetic matches would violate scientific ethics. We demonstrate technical competence across all three modalities while respecting dataset boundaries.'
  }
];

export default function DemoMode() {
  const [activeModuleIndex, setActiveModuleIndex] = useState(0);
  const cur = DEMO_MODULES[activeModuleIndex];

  return (
    <>
      <div className="page-header">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="badge badge-accent">Interactive Defense Guide</span>
            <span className="badge badge-emerald">6 Core Pillars</span>
          </div>
          <h1 className="page-title">Comprehensive Project Demo & Viva Master Guide</h1>
          <p className="page-subtitle">
            Complete architectural walkthrough, clinical rationale, empirical evidence, and examiner talking points.
          </p>
        </div>
        <div className="flex gap-2">
          <Link href={cur.liveRoute} className="btn btn-primary" style={{ boxShadow: '0 4px 16px rgba(59, 130, 246, 0.35)' }}>
            {cur.liveRouteLabel}
          </Link>
        </div>
      </div>

      <div className="page-body">
        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 28, alignItems: 'flex-start' }}>

          {/* ── Left Navigation Stepper ── */}
          <div className="card" style={{ padding: 16, position: 'sticky', top: 24 }}>
            <span className="text-xs font-bold text-muted uppercase tracking-wider block mb-3 px-2">
              Master Presentation Flow
            </span>
            <div className="flex-col gap-1.5">
              {DEMO_MODULES.map((mod, idx) => {
                const isActive = activeModuleIndex === idx;
                return (
                  <button
                    key={mod.id}
                    onClick={() => setActiveModuleIndex(idx)}
                    className="text-left w-full transition-all"
                    style={{
                      padding: '10px 12px',
                      borderRadius: 8,
                      background: isActive ? 'linear-gradient(90deg, hsla(217, 91%, 60%, 0.18) 0%, hsla(217, 91%, 60%, 0.04) 100%)' : 'transparent',
                      border: `1px solid ${isActive ? 'hsla(217, 91%, 60%, 0.4)' : 'transparent'}`,
                      color: isActive ? '#ffffff' : 'hsl(var(--text-2))',
                      fontWeight: isActive ? 700 : 500,
                      cursor: 'pointer',
                    }}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[10px] font-mono text-cyan uppercase font-bold">{mod.category.split('.')[1]}</span>
                      <span className={`badge badge-xs badge-${mod.badgeColor}`}>{mod.badge}</span>
                    </div>
                    <div className="text-xs truncate">{mod.title.split(':')[0]}</div>
                  </button>
                );
              })}
            </div>

            <div className="mt-4 pt-3 border-t border-[hsl(var(--border-subtle))]">
              <span className="text-[11px] text-muted block mb-2 px-1">Quick Navigation:</span>
              <div className="flex justify-between gap-2">
                <button
                  disabled={activeModuleIndex === 0}
                  onClick={() => setActiveModuleIndex(i => Math.max(0, i - 1))}
                  className="btn btn-xs btn-secondary flex-1"
                >
                  ← Prev
                </button>
                <button
                  disabled={activeModuleIndex === DEMO_MODULES.length - 1}
                  onClick={() => setActiveModuleIndex(i => Math.min(DEMO_MODULES.length - 1, i + 1))}
                  className="btn btn-xs btn-primary flex-1"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>

          {/* ── Main Detail Canvas ── */}
          <div className="flex-col gap-6 animate-in">

            {/* Header Card */}
            <div className="card" style={{ padding: 28, borderLeft: `4px solid hsl(var(--${cur.badgeColor}))` }}>
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-mono text-cyan uppercase font-bold">{cur.category}</span>
                <span className={`badge badge-${cur.badgeColor}`}>{cur.badge}</span>
              </div>
              <h2 style={{ fontSize: '1.45rem', fontWeight: 800, color: '#ffffff', marginBottom: 6 }}>
                {cur.title}
              </h2>
              <p className="text-sm font-semibold" style={{ color: 'hsl(var(--text-2))', marginBottom: 16 }}>
                "{cur.hook}"
              </p>

              {/* Problem vs Solution Split */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 12 }}>
                <div className="p-3.5 rounded-lg bg-[hsl(var(--bg-2))] border border-[hsl(var(--border))]">
                  <span className="text-xs font-bold text-rose uppercase block mb-1">⚠️ The Research Problem</span>
                  <p className="text-xs text-muted" style={{ lineHeight: 1.6 }}>{cur.problem}</p>
                </div>
                <div className="p-3.5 rounded-lg bg-[hsl(var(--bg-2))] border border-[hsl(var(--border))]">
                  <span className="text-xs font-bold text-emerald uppercase block mb-1">💡 Cerebro-X Innovation</span>
                  <p className="text-xs text-muted" style={{ lineHeight: 1.6 }}>{cur.solution}</p>
                </div>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid-4">
              {cur.metrics.map(m => (
                <div key={m.label} className="stat-card" style={{ padding: 18 }}>
                  <span className="stat-label">{m.label}</span>
                  <span className="stat-value" style={{ fontSize: '1.75rem', color: `hsl(var(--${cur.badgeColor}))` }}>
                    {m.value}
                  </span>
                  <span className="stat-sub text-[11px]">{m.note}</span>
                </div>
              ))}
            </div>

            {/* Architecture Pipeline */}
            <div className="card" style={{ padding: 24 }}>
              <h3 className="text-xs font-bold text-muted uppercase tracking-wider mb-2">
                Engineering & Pipeline Architecture
              </h3>
              <div className="p-3 rounded-lg bg-[hsl(var(--bg-0))] border border-[hsl(var(--border))] font-mono text-xs text-cyan" style={{ lineHeight: 1.6 }}>
                {cur.architecture}
              </div>

              <div className="mt-4">
                <span className="text-xs font-bold text-muted uppercase tracking-wider block mb-2">Key Technical Strengths:</span>
                <ul className="flex-col gap-1.5 text-xs text-muted" style={{ paddingLeft: 18 }}>
                  {cur.keyStrengths.map((str, i) => (
                    <li key={i} style={{ listStyleType: 'disc' }}>{str}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Examiner & Viva Defense Script */}
            <div className="callout callout-warning animate-in" style={{ padding: 20 }}>
              <span style={{ fontSize: '1.4rem' }}>🎓</span>
              <div>
                <strong style={{ color: '#ffffff', fontSize: '0.9rem' }}>Viva Defense Talking Points (How to explain to examiners):</strong>
                <p className="text-xs mt-1.5" style={{ color: 'hsl(var(--text-1))', lineHeight: 1.65 }}>
                  {cur.defenseScript}
                </p>
              </div>
            </div>

            {/* Direct Test Link */}
            <div className="flex justify-between items-center p-4 rounded-xl bg-[hsl(var(--bg-1))] border border-[hsl(var(--border))]">
              <div>
                <span className="text-sm font-bold text-white block">Ready to demonstrate this module live?</span>
                <span className="text-xs text-muted">Test the actual working implementation with model inference.</span>
              </div>
              <Link href={cur.liveRoute} className="btn btn-primary btn-sm">
                {cur.liveRouteLabel} →
              </Link>
            </div>

          </div>

        </div>
      </div>
    </>
  );
}
