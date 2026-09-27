'use client';

import React, { useState } from 'react';
import styles from '../page.module.css';

const DEMO_STEPS = [
  {
    title: "STEP 1: Project Overview",
    content: (
      <div className={styles.card}>
        <h3>CEREBRO-X</h3>
        <p><strong>AI-Powered Digital Brain Twin for Alzheimer's Disease Progression Prediction</strong></p>
        <p>This project uses longitudinal OASIS-2 data to model changes in cognitive status. Instead of treating each visit independently, the temporal model processes the sequence of visits.</p>
        <ul>
          <li>OASIS-2 longitudinal data</li>
          <li>Longitudinal patient modeling</li>
          <li>Next-visit CDR prediction</li>
          <li>Digital Brain Twin latent state Z_t</li>
          <li>Clinical + MRI scalar multimodal model</li>
          <li>EEG standalone screening</li>
          <li>Explainability components</li>
          <li>MRI NIfTI pipeline status</li>
        </ul>
        <div style={{ padding: '12px', background: 'rgba(255, 60, 60, 0.1)', borderLeft: '4px solid #ff3c3c', marginTop: '16px' }}>
          <strong>WARNING: Research prototype only. Not validated for clinical use.</strong>
        </div>
      </div>
    )
  },
  {
    title: "STEP 2: Load Patient",
    content: (
      <div className={styles.card}>
        <h3>Load Sample Patient</h3>
        <p>To begin, navigate to the <strong>Predict CDR</strong> tab on the left and click <strong>"Load Sample Patient"</strong>.</p>
        <p>This will populate the dashboard with data from an existing patient in the OASIS-2 dataset.</p>
        <p>The model uses longitudinal observations, meaning it looks at the trajectory across Visit 1, Visit 2, etc., rather than just a single snapshot in time.</p>
        <ul>
          <li>Subject ID</li>
          <li>Visit sequence</li>
          <li>Clinical variables (Age, EDUC, MMSE, etc.)</li>
          <li>MRI-derived scalar variables (nWBV, eTIV, ASF)</li>
        </ul>
      </div>
    )
  },
  {
    title: "STEP 3: Clinical Prediction",
    content: (
      <div className={styles.card}>
        <h3>Clinical Prediction</h3>
        <p>On the <strong>Predict CDR</strong> tab, click <strong>"Predict Next-Visit CDR (Clinical)"</strong>.</p>
        <p>This calls the real backend endpoint: <code>POST /predict/clinical</code>.</p>
        <p><strong>Prediction target: next-visit CDR</strong></p>
        <p><em>(Note: This is strictly predicting the CDR value at the patient's next scheduled visit. It is not a 3-year prediction or a formal clinical diagnosis.)</em></p>
        <div style={{ marginTop: '16px', background: 'var(--bg-layer)', padding: '16px', borderRadius: '8px' }}>
          <h4>Reported OASIS-2 test-set results:</h4>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li>✓ Accuracy: <strong>75.0%</strong></li>
            <li>✓ Balanced Accuracy: <strong>55.7%</strong></li>
            <li>✓ Macro F1: <strong>55.1%</strong></li>
          </ul>
        </div>
      </div>
    )
  },
  {
    title: "STEP 4: Bimodal Prediction",
    content: (
      <div className={styles.card}>
        <h3>Bimodal Prediction</h3>
        <p>On the same page, you can run the <strong>Clinical + MRI scalar model</strong> by clicking <strong>"Predict Next-Visit CDR (Bimodal)"</strong>.</p>
        <p>This calls <code>POST /predict/bimodal</code>.</p>
        <div style={{ display: 'flex', gap: '16px', margin: '16px 0', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-layer)', padding: '16px', borderRadius: '8px' }}>
          <div style={{ textAlign: 'center' }}>Clinical Representation</div>
          <div>+</div>
          <div style={{ textAlign: 'center' }}>MRI scalar representation</div>
          <div>→</div>
          <div style={{ textAlign: 'center' }}>Fused representation</div>
          <div>→</div>
          <div style={{ textAlign: 'center' }}>CDR Prediction</div>
        </div>
        <p>The current deployed bimodal model combines longitudinal clinical information with MRI-derived scalar features.</p>
        <p><em>Note: This is the Clinical + MRI scalar model, not the raw 3D CNN MRI model.</em></p>
      </div>
    )
  },
  {
    title: "STEP 5: Digital Brain Twin",
    content: (
      <div className={styles.card}>
        <h3>Digital Brain Twin (Z_t)</h3>
        <p>Navigate to the <strong>Brain Twin</strong> tab and click <strong>"Extract Z_t State"</strong> to call <code>POST /brain-twin/extract</code>.</p>
        <p><strong>Z_t is a learned latent representation of the patient's longitudinal state.</strong></p>
        <p>It is extracted from the hidden state of the TemporalCerebroNet (GRU). It is a 64-dimensional vector that encapsulates the patient's cognitive trajectory over time.</p>
        <p><em>Status: The current visualization shows this 64-D vector statically. It is a mathematical latent space, not a literal anatomical 3D reconstruction of the brain.</em></p>
      </div>
    )
  },
  {
    title: "STEP 6: Explainability",
    content: (
      <div className={styles.card}>
        <h3>Explainability</h3>
        <p>Navigate to the <strong>Explainability</strong> tab.</p>
        <ul>
          <li><strong>Clinical Explainability:</strong> <span style={{ color: 'var(--accent)' }}>WORKING (Heuristic)</span> - Uses linear formula attributions.</li>
          <li><strong>EEG Explainability:</strong> <span style={{ color: 'var(--text-side-2)' }}>SCAFFOLDED</span> - Standalone model exists.</li>
          <li><strong>MRI Explainability:</strong> <span style={{ color: 'var(--text-side-2)' }}>PARTIAL</span> - 3D MRI Grad-CAM is implemented but unavailable for live inference because a trained CNN3D checkpoint is not currently present.</li>
        </ul>
      </div>
    )
  },
  {
    title: "STEP 7: Experiments",
    content: (
      <div className={styles.card}>
        <h3>Experiments Registry</h3>
        <p>Navigate to the <strong>Experiments</strong> tab. This fetches data from <code>GET /experiments/</code>.</p>
        <p>Here you can see the neutrally presented models and baselines:</p>
        <ul>
          <li>Temporal GRU</li>
          <li>Bimodal Fusion (Scalar)</li>
          <li>Dummy Majority</li>
          <li>Random Forest</li>
          <li>Logistic Regression</li>
          <li>Last-Visit Baseline</li>
        </ul>
      </div>
    )
  },
  {
    title: "STEP 8: Prediction History",
    content: (
      <div className={styles.card}>
        <h3>Prediction Log</h3>
        <p>Navigate to the <strong>Prediction Log</strong> tab to call <code>GET /history/</code>.</p>
        <p>Clinical prediction requests are persisted in a local SQLite database. Here you can view past predictions, patient IDs, timestamps, and the predicted vs. actual outcomes.</p>
      </div>
    )
  },
  {
    title: "STEP 9: MRI Pipeline Status",
    content: (
      <div className={styles.card}>
        <h3>New MRI Pipeline Status</h3>
        <p>A new raw NIfTI MRI pipeline has been implemented and is ready for real data.</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'var(--bg-layer)', padding: '16px', borderRadius: '8px', margin: '16px 0' }}>
          <div>NIfTI Upload → NiBabel Loading</div>
          <div>↓</div>
          <div>Validation & Orientation Normalization</div>
          <div>↓</div>
          <div>Intensity Preprocessing & Resizing to 64³</div>
          <div>↓</div>
          <div>3D CNN</div>
          <div>↓</div>
          <div>MRI Embedding → Prediction / Grad-CAM</div>
        </div>
        <p style={{ color: '#ff3c3c', fontWeight: 'bold' }}>Training checkpoint currently unavailable.</p>
        <p>Because there are no local OASIS-2 MRI volumes and no trained CNN3D checkpoint, attempting to upload an MRI without the checkpoint will return an honest status message rather than a fake prediction.</p>
      </div>
    )
  },
  {
    title: "STEP 10: Final Summary",
    content: (
      <div className={styles.card}>
        <h3>Final Implementation Status</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: '16px' }}>
          <div>
            <h4 style={{ color: 'var(--accent)' }}>WORKING</h4>
            <ul style={{ fontSize: '0.9rem' }}>
              <li>OASIS-2 longitudinal dataset pipeline</li>
              <li>Clinical GRU</li>
              <li>Next-visit CDR prediction</li>
              <li>Clinical latent Z_t</li>
              <li>Clinical + MRI scalar bimodal prediction</li>
              <li>EEG standalone screening</li>
              <li>Experiment registry & Prediction history</li>
              <li>FastAPI backend & Next.js dashboard</li>
              <li>Real NIfTI preprocessing pipeline</li>
              <li>3D CNN architecture & 3D Grad-CAM code</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: '#ff3c3c' }}>PARTIAL / NOT DEPLOYED</h4>
            <ul style={{ fontSize: '0.9rem' }}>
              <li>Trained raw MRI 3D CNN checkpoint</li>
              <li>Live raw MRI prediction & Grad-CAM</li>
              <li>True Clinical + MRI CNN embedding fusion</li>
              <li>Patient-aligned EEG fusion</li>
              <li>Full tri-modal fusion</li>
              <li>Multi-year progression prediction</li>
              <li>Fully dynamic anatomical Brain Twin</li>
              <li>External OASIS-3 validation</li>
            </ul>
          </div>
        </div>
      </div>
    )
  }
];

export default function DemoMode() {
  const [currentStep, setCurrentStep] = useState(0);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Cerebro-X Demo Mode</h1>
          <p className={styles.subtitle}>Guided walkthrough of the digital brain twin platform.</p>
        </div>
      </header>

      <main className={styles.main}>
        <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-start' }}>
          
          {/* Sidebar stepper */}
          <div style={{ width: '250px', flexShrink: 0, position: 'sticky', top: '24px' }}>
            <div className={styles.card}>
              <h3 style={{ marginBottom: '16px' }}>Demo Sequence</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {DEMO_STEPS.map((step, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCurrentStep(idx)}
                    style={{
                      textAlign: 'left',
                      padding: '8px 12px',
                      background: currentStep === idx ? 'var(--accent)' : 'transparent',
                      color: currentStep === idx ? '#000' : 'inherit',
                      border: '1px solid ' + (currentStep === idx ? 'var(--accent)' : 'var(--border-color)'),
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '0.9rem'
                    }}
                  >
                    {step.title}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main content */}
          <div style={{ flexGrow: 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <button 
                className="btn btn-secondary"
                disabled={currentStep === 0}
                onClick={() => setCurrentStep(s => Math.max(0, s - 1))}
              >
                ← Previous Step
              </button>
              <button 
                className="btn btn-primary"
                disabled={currentStep === DEMO_STEPS.length - 1}
                onClick={() => setCurrentStep(s => Math.min(DEMO_STEPS.length - 1, s + 1))}
              >
                Next Step →
              </button>
            </div>
            
            <div style={{ animation: 'fadeIn 0.3s ease-in-out' }}>
              <h2 style={{ marginBottom: '24px', color: 'var(--accent)' }}>{DEMO_STEPS[currentStep].title}</h2>
              {DEMO_STEPS[currentStep].content}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
