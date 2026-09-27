'use client';

import Image from 'next/image';

const SHAP_INSIGHTS = [
  {
    rank: 1,
    feature: 'nWBV (Normalised Whole Brain Volume)',
    impact: 'Very High',
    description: 'Primary structural biomarker. Atrophy in brain volume directly predicts cognitive decline.',
  },
  {
    rank: 2,
    feature: 'MMSE (Mini-Mental State Exam)',
    impact: 'High',
    description: 'Cognitive screening score. Drops of ≥2 points between visits strongly predict CDR worsening.',
  },
  {
    rank: 3,
    feature: 'Age',
    impact: 'High',
    description: 'Strong prior for disease risk. The model learns age-adjusted trajectories.',
  },
  {
    rank: 4,
    feature: 'CDR (Current Clinical Dementia Rating)',
    impact: 'Medium',
    description: 'Present CDR heavily informs next CDR — persistence effect. Delta is also informative.',
  },
  {
    rank: 5,
    feature: 'eTIV (Estimated Total Intracranial Volume)',
    impact: 'Medium',
    description: 'Correction factor for head size. Interacts with nWBV to normalise atrophy.',
  },
];

export default function ExplainabilityPage() {
  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Explainability</h1>
          <p className="page-subtitle">Modality-specific model interpretations (SHAP, Grad-CAM, EEG Attribution)</p>
        </div>
      </div>

      <div className="page-body flex-col gap-10">

        {/* ── CLINICAL EXPLANATION ── */}
        <section aria-labelledby="clinical-heading" className="animate-in">
          <h2 id="clinical-heading" className="text-sm font-bold text-muted"
            style={{ textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 16, color: 'hsl(260 40% 60%)' }}>
            ■ Clinical Explanation
          </h2>
          
          <div className="card card-pastel-lavender" style={{ padding: 32, boxShadow: 'none' }}>
            <div className="grid-2" style={{ gridTemplateColumns: '1fr 340px', gap: 32 }}>

              {/* SHAP plots */}
              <div className="flex-col gap-6">
                <div className="card" style={{ background: 'hsl(var(--bg-1))' }}>
                  <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 12 }}>
                    Top-15 Feature Importance (Global)
                  </h3>
                  <div style={{
                    position: 'relative', width: '100%', height: 320,
                    borderRadius: 8, overflow: 'hidden', background: 'white',
                    border: '1px solid hsl(var(--border))'
                  }}>
                    <Image
                      src="/shap_clinical.png"
                      alt="SHAP feature importance summary plot across all test patients"
                      fill
                      style={{ objectFit: 'contain' }}
                    />
                  </div>
                  <p className="text-xs text-muted" style={{ marginTop: 10 }}>
                    SHAP TreeExplainer on longitudinal feature vectors
                  </p>
                </div>

                <div className="card" style={{ background: 'hsl(var(--bg-1))' }}>
                  <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 12 }}>
                    Patient-Level Waterfall
                  </h3>
                  <p className="text-xs text-muted" style={{ marginBottom: 12 }}>
                    Breakdown of feature contributions. Positive values push toward higher CDR.
                  </p>
                  <div style={{
                    position: 'relative', width: '100%', height: 280,
                    borderRadius: 8, overflow: 'hidden', background: 'white',
                    border: '1px solid hsl(var(--border))'
                  }}>
                    <Image
                      src="/shap_waterfall.png"
                      alt="SHAP waterfall plot"
                      fill
                      style={{ objectFit: 'contain' }}
                    />
                  </div>
                </div>
              </div>

              {/* Feature rankings */}
              <div className="flex-col gap-3">
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 8 }}>Top Clinical Drivers</h3>
                {SHAP_INSIGHTS.map((item) => (
                  <div key={item.rank} className="card card-sm" style={{ background: 'hsl(var(--bg-1))' }}>
                    <div className="flex items-center gap-2" style={{ marginBottom: 6 }}>
                      <span style={{
                        width: 22, height: 22, borderRadius: '50%', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', fontSize: '0.6875rem',
                        fontWeight: 700, background: 'hsl(var(--pastel-lavender))', color: 'hsl(260 50% 40%)',
                        flexShrink: 0,
                      }}>
                        {item.rank}
                      </span>
                      <span style={{ fontSize: '0.8125rem', fontWeight: 600 }}>{item.feature}</span>
                      <span className="badge badge-neutral" style={{ marginLeft: 'auto' }}>
                        {item.impact}
                      </span>
                    </div>
                    <p className="text-xs text-muted" style={{ lineHeight: 1.55 }}>{item.description}</p>
                  </div>
                ))}
              </div>

            </div>
          </div>
        </section>

        {/* ── MRI EXPLANATION ── */}
        <section aria-labelledby="mri-heading" className="animate-in animate-delay-1">
          <h2 id="mri-heading" className="text-sm font-bold text-muted"
            style={{ textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 16, color: 'hsl(10 70% 65%)' }}>
            ■ MRI Explanation
          </h2>
          
          <div className="card card-pastel-peach" style={{ padding: 32, boxShadow: 'none' }}>
            <div className="grid-2" style={{ gridTemplateColumns: '1fr 1fr', gap: 32 }}>
              <div className="card" style={{ background: 'hsl(var(--bg-1))' }}>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 12 }}>
                  Spatial Attention Heatmap (Grad-CAM)
                </h3>
                <div style={{
                  position: 'relative', width: '100%', height: 250,
                  borderRadius: 8, overflow: 'hidden', background: 'hsl(var(--bg-2))',
                  border: '1px solid hsl(var(--border))'
                }}>
                  <Image
                    src="/gradcam_mri.png"
                    alt="MRI Grad-CAM spatial attention heatmap"
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
                <p className="text-xs text-muted" style={{ marginTop: 10 }}>
                  Heatmap indicating structural regions driving model predictions.
                </p>
              </div>

              <div className="card" style={{ background: 'hsl(var(--bg-1))' }}>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 12 }}>
                  Pathway Status
                </h3>
                <div className="callout callout-warning" style={{ marginBottom: 16, background: 'hsl(var(--bg-0))' }}>
                  <span>⚠️</span>
                  <div>
                    <strong>CNN3D pathway not active</strong>
                    <p className="text-sm" style={{ marginTop: 4 }}>
                      Grad-CAM requires 3D MRI NIfTI volumes. The current Cerebro-X pipeline 
                      uses MRI-derived scalars rather than raw volumes.
                    </p>
                  </div>
                </div>

                <div className="flex-col gap-2" style={{ fontSize: '0.8125rem' }}>
                  {[
                    { step: '1', label: 'MRI NIfTI input',   active: false },
                    { step: '2', label: 'CNN3D encoder',      active: false },
                    { step: '3', label: 'MRI embedding',      active: false },
                    { step: '4', label: 'Grad-CAM gradient',  active: false },
                    { step: '5', label: 'MRI scalar fallback',active: true  },
                    { step: '6', label: 'Bimodal fusion',     active: true  },
                  ].map((s) => (
                    <div key={s.step} className="flex items-center gap-2">
                      <span style={{
                        width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '0.6875rem', fontWeight: 700,
                        background: s.active ? 'hsl(var(--pastel-sage))' : 'hsl(var(--bg-2))',
                        color: s.active ? 'hsl(var(--emerald))' : 'hsl(var(--text-3))',
                        border: `1px solid ${s.active ? 'hsl(var(--emerald) / 0.4)' : 'hsl(var(--border))'}`,
                      }}>
                        {s.step}
                      </span>
                      <span style={{ color: s.active ? 'hsl(var(--text-1))' : 'hsl(var(--text-3))' }}>
                        {s.label}
                      </span>
                      {s.active && <span className="badge badge-emerald" style={{ marginLeft: 'auto' }}>Active</span>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── EEG EXPLANATION ── */}
        <section aria-labelledby="eeg-heading" className="animate-in animate-delay-2">
          <h2 id="eeg-heading" className="text-sm font-bold text-muted"
            style={{ textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 16, color: 'hsl(140 30% 50%)' }}>
            ■ EEG Explanation
          </h2>
          
          <div className="card card-pastel-sage" style={{ padding: 48, boxShadow: 'none', textAlign: 'center' }}>
            <p style={{ fontSize: '2.5rem', marginBottom: 12 }}>🌊</p>
            <p style={{ fontWeight: 600, marginBottom: 6, color: 'hsl(var(--text-1))', fontSize: '1.125rem' }}>EEG Modality Not Available</p>
            <p className="text-muted text-sm">
              The EEG pathway and corresponding attribution maps have not been implemented in the current research phase.
            </p>
          </div>
        </section>

      </div>
    </>
  );
}
