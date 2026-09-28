'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';

// ─── Types ────────────────────────────────────────────────────────────────────

type SHAPFeature = {
  name: string;
  feature_key: string;
  importance: number;
  mean_abs_shap: number;
  mean_shap: number;
  direction: 'positive' | 'negative';
  relative_importance_pct: number;
};

type TemporalAttribution = {
  visit: number;
  shap_values: Record<string, number>;
};

type ClinicalExplainResponse = {
  success: boolean;
  method: string;
  model?: string;
  subject_id?: string;
  prediction?: { class: number; label: string; probability: number };
  features?: SHAPFeature[];
  temporal_attributions?: TemporalAttribution[];
  background?: { source: string; num_samples: number; contamination: string };
  warning?: string;
  disclaimer?: string;
  error?: string;
};

type GradCAMResponse = {
  success: boolean;
  explanation_method?: string;
  prediction?: { class: number; label: string; probability: number };
  visualizations?: { axial?: string; sagittal?: string; coronal?: string } | null;
  mri_embedding?: { dimension: number };
  heatmap_stats?: { min: number; max: number; mean: number };
  target_layer?: string;
  disclaimer?: string;
  error?: string;
  detail?: any;
};

// ─── Constants ────────────────────────────────────────────────────────────────

const API_BASE = 'http://127.0.0.1:8000';

const SAMPLE_PATIENT = {
  subject_id: 'DEMO-001',
  visits: [
    {
      age: 72.0, educ: 16.0, ses: 2.0, mmse: 28.0, cdr: 0.0,
      nwbv: 0.763, etiv: 1480.0, asf: 1.23, gender: 'F', hand: 'R',
    },
    {
      age: 74.0, educ: 16.0, ses: 2.0, mmse: 26.0, cdr: 0.5,
      nwbv: 0.743, etiv: 1478.0, asf: 1.23, gender: 'F', hand: 'R',
    },
  ],
};

const CDR_COLORS = ['#22c55e', '#facc15', '#f97316', '#ef4444'] as const;

// ─── Method Badge ─────────────────────────────────────────────────────────────

function MethodBadge({ method }: { method: string }) {
  const isRealSHAP    = method === 'SHAP_GradientExplainer';
  const isRealGradCAM = method === '3D-GradCAM';
  const isHeuristic   = method === 'heuristic_fallback' || method === 'heuristic';

  if (isRealSHAP)
    return (
      <span className="badge" style={{ background: 'hsl(142 70% 25% / 0.2)', color: 'hsl(142 70% 65%)', border: '1px solid hsl(142 70% 40% / 0.4)', fontSize: '0.75rem' }}>
        ✓ Real SHAP (GradientExplainer)
      </span>
    );
  if (isRealGradCAM)
    return (
      <span className="badge" style={{ background: 'hsl(200 70% 25% / 0.2)', color: 'hsl(200 70% 65%)', border: '1px solid hsl(200 70% 40% / 0.4)', fontSize: '0.75rem' }}>
        ✓ Real 3D Grad-CAM
      </span>
    );
  if (isHeuristic)
    return (
      <span className="badge" style={{ background: 'hsl(38 80% 25% / 0.2)', color: 'hsl(38 80% 65%)', border: '1px solid hsl(38 80% 40% / 0.4)', fontSize: '0.75rem' }}>
        ⚠ Heuristic Fallback
      </span>
    );
  return (
    <span className="badge badge-neutral" style={{ fontSize: '0.75rem' }}>{method}</span>
  );
}

// ─── Feature Bar ──────────────────────────────────────────────────────────────

function FeatureBar({ feat, maxImportance }: { feat: SHAPFeature; maxImportance: number }) {
  const pct = maxImportance > 0 ? (feat.mean_abs_shap / maxImportance) * 100 : 0;
  const dirColor = feat.direction === 'positive' ? 'hsl(0 70% 65%)' : 'hsl(210 80% 65%)';
  return (
    <div style={{ marginBottom: 8 }}>
      <div className="flex items-center gap-2" style={{ marginBottom: 3 }}>
        <span style={{ fontSize: '0.8125rem', flex: 1, color: 'hsl(var(--text-1))' }}>
          {feat.name}
        </span>
        <span style={{ fontSize: '0.75rem', color: dirColor, fontWeight: 600 }}>
          {feat.direction === 'positive' ? '▲' : '▼'} {feat.mean_shap.toFixed(4)}
        </span>
        <span style={{ fontSize: '0.75rem', color: 'hsl(var(--text-3))', minWidth: 40, textAlign: 'right' }}>
          {feat.relative_importance_pct.toFixed(1)}%
        </span>
      </div>
      <div style={{ height: 6, background: 'hsl(var(--bg-3))', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          background: `linear-gradient(90deg, ${dirColor}88, ${dirColor})`,
          borderRadius: 4,
          transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ExplainabilityPage() {
  // Clinical SHAP state
  const [clinicalLoading, setClinicalLoading] = useState(false);
  const [clinicalResult, setClinicalResult] = useState<ClinicalExplainResponse | null>(null);
  const [clinicalError, setClinicalError] = useState<string | null>(null);
  const [activeVisit, setActiveVisit] = useState(0);

  // MRI Grad-CAM state
  const [mriLoading, setMriLoading] = useState(false);
  const [mriResult, setMriResult] = useState<GradCAMResponse | null>(null);
  const [mriError, setMriError] = useState<string | null>(null);
  const [mriPlane, setMriPlane] = useState<'axial' | 'sagittal' | 'coronal'>('axial');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Clinical SHAP ──────────────────────────────────────────────────────────

  const runClinicalSHAP = async () => {
    setClinicalLoading(true);
    setClinicalError(null);
    setClinicalResult(null);
    try {
      const res = await fetch(`${API_BASE}/explain/clinical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(SAMPLE_PATIENT),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        const msg = typeof detail === 'string' ? detail : (detail.message || detail.error || JSON.stringify(detail));
        setClinicalError(msg);
      } else {
        setClinicalResult(data);
      }
    } catch (e: any) {
      setClinicalError(`Network error: ${e.message}`);
    } finally {
      setClinicalLoading(false);
    }
  };

  // ── MRI Grad-CAM ──────────────────────────────────────────────────────────

  const runMRIGradCAM = async (file: File) => {
    setMriLoading(true);
    setMriError(null);
    setMriResult(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/explain/mri`, {
        method: 'POST',
        body: form,
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || data;
        const msg = typeof detail === 'string' ? detail : (detail.message || detail.error || JSON.stringify(detail));
        setMriError(msg);
      } else {
        setMriResult(data);
      }
    } catch (e: any) {
      setMriError(`Network error: ${e.message}`);
    } finally {
      setMriLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) runMRIGradCAM(file);
  };

  // ── UI ────────────────────────────────────────────────────────────────────

  const topFeatures = clinicalResult?.features?.slice(0, 10) ?? [];
  const maxImportance = topFeatures[0]?.mean_abs_shap ?? 1;

  const currentVisitAttrib = clinicalResult?.temporal_attributions?.find(t => t.visit === activeVisit);

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Live Explainability</h1>
          <p className="page-subtitle">
            Real SHAP (GradientExplainer) · Real 3D Grad-CAM · Live API integration
          </p>
        </div>
      </div>

      <div className="page-body flex-col gap-10">

        {/* ── Disclaimer ── */}
        <div className="callout callout-warning animate-in" role="alert">
          <span>⚠️</span>
          <div>
            <strong>Research Prototype.</strong> Clinical SHAP uses the trained TemporalCerebroNet.{' '}
            MRI Grad-CAM uses a synthetic-trained scaffold — see{' '}
            <code style={{ fontSize: '0.8rem' }}>scripts/train_cnn3d.py</code> for details.
            Neither is validated for clinical use.
          </div>
        </div>

        {/* ── CLINICAL SHAP SECTION ── */}
        <section aria-labelledby="clinical-heading" className="animate-in">
          <div className="flex items-center gap-4" style={{ marginBottom: 16 }}>
            <h2 id="clinical-heading" className="text-sm font-bold text-muted"
              style={{ textTransform: 'uppercase', letterSpacing: '0.07em', color: 'hsl(260 40% 60%)' }}>
              ■ Clinical SHAP Explanation
            </h2>
            {clinicalResult && <MethodBadge method={clinicalResult.method} />}
          </div>

          <div className="card card-pastel-lavender" style={{ padding: 28 }}>

            {/* Controls */}
            <div className="flex items-center gap-4" style={{ marginBottom: 20 }}>
              <button
                id="run-clinical-shap-btn"
                className="btn btn-accent"
                onClick={runClinicalSHAP}
                disabled={clinicalLoading}
                style={{ minWidth: 180 }}
              >
                {clinicalLoading ? '⏳ Computing SHAP…' : '⚡ Run Clinical SHAP'}
              </button>
              <span className="text-sm text-muted">
                Uses demo patient (2 visits, CDR 0→0.5 progression)
              </span>
            </div>

            {/* Error */}
            {clinicalError && (
              <div className="callout callout-warning" style={{ marginBottom: 16 }}>
                <span>⚠️</span>
                <div>
                  <strong>Error:</strong> {clinicalError}
                  {clinicalError.includes('background') && (
                    <p className="text-xs" style={{ marginTop: 4 }}>
                      Run <code>python scripts/build_shap_background.py</code> then restart the server.
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Heuristic warning */}
            {clinicalResult?.method === 'heuristic_fallback' && (
              <div className="callout callout-warning" style={{ marginBottom: 16 }}>
                <span>⚠️</span>
                <div>
                  <strong>Heuristic Fallback Active</strong>
                  <p className="text-sm" style={{ marginTop: 4 }}>{clinicalResult.warning}</p>
                </div>
              </div>
            )}

            {/* Results */}
            {clinicalResult?.success && (
              <div className="grid-2" style={{ gridTemplateColumns: '1fr 320px', gap: 24 }}>

                {/* Feature importance bars */}
                <div className="flex-col gap-4">

                  {/* Prediction banner */}
                  {clinicalResult.prediction && (
                    <div className="card" style={{ background: 'hsl(var(--bg-1))', padding: 16 }}>
                      <div className="flex items-center gap-4">
                        <div>
                          <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Prediction</p>
                          <p style={{ fontWeight: 700, fontSize: '1rem', color: CDR_COLORS[clinicalResult.prediction.class] }}>
                            {clinicalResult.prediction.label}
                          </p>
                        </div>
                        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                          <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Confidence</p>
                          <p style={{ fontWeight: 700, fontSize: '1.25rem' }}>
                            {(clinicalResult.prediction.probability * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Feature importance */}
                  {topFeatures.length > 0 && (
                    <div className="card" style={{ background: 'hsl(var(--bg-1))', padding: 20 }}>
                      <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: 16 }}>
                        Top Feature Importances — {clinicalResult.method === 'SHAP_GradientExplainer' ? 'mean |SHAP|' : 'Heuristic contribution'}
                      </h3>
                      {topFeatures.map(f => (
                        <FeatureBar key={f.feature_key} feat={f} maxImportance={maxImportance} />
                      ))}
                    </div>
                  )}

                  {/* Temporal attributions */}
                  {clinicalResult.temporal_attributions && clinicalResult.temporal_attributions.length > 0 && (
                    <div className="card" style={{ background: 'hsl(var(--bg-1))', padding: 20 }}>
                      <div className="flex items-center gap-3" style={{ marginBottom: 12 }}>
                        <h3 style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                          Temporal Attributions (per visit)
                        </h3>
                        <div className="flex gap-1" style={{ marginLeft: 'auto' }}>
                          {clinicalResult.temporal_attributions.map(t => (
                            <button
                              key={t.visit}
                              onClick={() => setActiveVisit(t.visit)}
                              className={`btn btn-sm ${activeVisit === t.visit ? 'btn-accent' : 'btn-ghost'}`}
                              style={{ minWidth: 60 }}
                            >
                              Visit {t.visit}
                            </button>
                          ))}
                        </div>
                      </div>
                      {currentVisitAttrib && (
                        <div className="flex-col gap-1" style={{ maxHeight: 200, overflowY: 'auto' }}>
                          {Object.entries(currentVisitAttrib.shap_values)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 8)
                            .map(([key, val]) => (
                              <div key={key} className="flex items-center gap-2"
                                style={{ fontSize: '0.8125rem', padding: '2px 0' }}>
                                <span style={{ flex: 1, color: 'hsl(var(--text-2))' }}>{key}</span>
                                <span style={{
                                  fontWeight: 600,
                                  color: val >= 0 ? 'hsl(0 70% 65%)' : 'hsl(210 80% 65%)',
                                }}>
                                  {val >= 0 ? '+' : ''}{val.toFixed(5)}
                                </span>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Sidebar: background info + top drivers */}
                <div className="flex-col gap-3">
                  {clinicalResult.method === 'SHAP_GradientExplainer' && clinicalResult.background && (
                    <div className="card card-sm" style={{ background: 'hsl(var(--bg-1))' }}>
                      <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: 8 }}>
                        Background Dataset
                      </h4>
                      <div className="flex-col gap-1" style={{ fontSize: '0.75rem', color: 'hsl(var(--text-3))' }}>
                        <p><strong>Source:</strong> {clinicalResult.background.source}</p>
                        <p><strong>Samples:</strong> {clinicalResult.background.num_samples}</p>
                        <p><strong>Contamination:</strong> {clinicalResult.background.contamination}</p>
                      </div>
                    </div>
                  )}

                  <div className="card card-sm" style={{ background: 'hsl(var(--bg-1))' }}>
                    <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: 8 }}>
                      Top Clinical Drivers
                    </h4>
                    {topFeatures.slice(0, 5).map((f, i) => (
                      <div key={f.feature_key} className="flex items-center gap-2"
                        style={{ marginBottom: 6 }}>
                        <span style={{
                          width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: '0.625rem', fontWeight: 700,
                          background: 'hsl(var(--pastel-lavender))', color: 'hsl(260 50% 40%)',
                        }}>{i + 1}</span>
                        <span style={{ fontSize: '0.8125rem', flex: 1 }}>{f.name}</span>
                        <span className="badge badge-neutral" style={{ fontSize: '0.6875rem' }}>
                          {f.relative_importance_pct.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>

                  {clinicalResult.disclaimer && (
                    <p className="text-xs text-muted" style={{ lineHeight: 1.6 }}>
                      {clinicalResult.disclaimer}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Initial state */}
            {!clinicalLoading && !clinicalResult && !clinicalError && (
              <div style={{ textAlign: 'center', padding: '32px 0', color: 'hsl(var(--text-3))' }}>
                <p style={{ fontSize: '2rem', marginBottom: 8 }}>🧠</p>
                <p>Click <strong>Run Clinical SHAP</strong> to compute real SHAP attributions from the live API.</p>
              </div>
            )}
          </div>
        </section>

        {/* ── MRI GRAD-CAM SECTION ── */}
        <section aria-labelledby="mri-heading" className="animate-in animate-delay-1">
          <div className="flex items-center gap-4" style={{ marginBottom: 16 }}>
            <h2 id="mri-heading" className="text-sm font-bold text-muted"
              style={{ textTransform: 'uppercase', letterSpacing: '0.07em', color: 'hsl(10 70% 65%)' }}>
              ■ MRI 3D Grad-CAM Explanation
            </h2>
            {mriResult?.explanation_method && <MethodBadge method={mriResult.explanation_method} />}
          </div>

          <div className="card card-pastel-peach" style={{ padding: 28 }}>

            {/* Upload control */}
            <div className="flex items-center gap-4" style={{ marginBottom: 20 }}>
              <button
                id="upload-mri-btn"
                className="btn btn-accent"
                onClick={() => fileInputRef.current?.click()}
                disabled={mriLoading}
                style={{ minWidth: 180 }}
              >
                {mriLoading ? '⏳ Processing MRI…' : '🧠 Upload NIfTI + Run Grad-CAM'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".nii,.nii.gz"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
                id="mri-file-input"
              />
              <span className="text-sm text-muted">Accepts .nii / .nii.gz T1 MRI files</span>
            </div>

            {/* MRI error */}
            {mriError && (
              <div className="callout callout-warning" style={{ marginBottom: 16 }}>
                <span>⚠️</span>
                <div>
                  <strong>MRI Grad-CAM Error:</strong>{' '}
                  {typeof mriError === 'string' ? mriError : JSON.stringify(mriError)}
                  {mriError.includes('CNN3D_CHECKPOINT_NOT_AVAILABLE') && (
                    <p className="text-xs" style={{ marginTop: 4 }}>
                      Run <code>python scripts/train_cnn3d.py</code> then restart the backend.
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* MRI results */}
            {mriResult?.success && (
              <div className="flex-col gap-6">

                {/* Prediction banner */}
                {mriResult.prediction && (
                  <div className="card" style={{ background: 'hsl(var(--bg-1))', padding: 16 }}>
                    <div className="flex items-center gap-6">
                      <div>
                        <p className="text-xs text-muted" style={{ marginBottom: 4 }}>CNN Prediction</p>
                        <p style={{ fontWeight: 700, color: CDR_COLORS[mriResult.prediction.class] }}>
                          {mriResult.prediction.label}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Confidence</p>
                        <p style={{ fontWeight: 700, fontSize: '1.1rem' }}>
                          {((mriResult.prediction.probability ?? 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                      {mriResult.target_layer && (
                        <div>
                          <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Target Layer</p>
                          <p style={{ fontSize: '0.8125rem', fontFamily: 'monospace' }}>{mriResult.target_layer}</p>
                        </div>
                      )}
                      {mriResult.mri_embedding && (
                        <div style={{ marginLeft: 'auto' }}>
                          <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Embedding</p>
                          <p style={{ fontSize: '0.8125rem' }}>{mriResult.mri_embedding.dimension}-dim</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Slice visualizations */}
                {mriResult.visualizations && (
                  <div className="card" style={{ background: 'hsl(var(--bg-1))', padding: 20 }}>
                    <div className="flex items-center gap-3" style={{ marginBottom: 14 }}>
                      <h3 style={{ fontSize: '0.9375rem', fontWeight: 600 }}>
                        Grad-CAM Overlays
                      </h3>
                      <div className="flex gap-1" style={{ marginLeft: 'auto' }}>
                        {(['axial', 'sagittal', 'coronal'] as const).map(plane => (
                          <button
                            key={plane}
                            onClick={() => setMriPlane(plane)}
                            className={`btn btn-sm ${mriPlane === plane ? 'btn-accent' : 'btn-ghost'}`}
                          >
                            {plane.charAt(0).toUpperCase() + plane.slice(1)}
                          </button>
                        ))}
                      </div>
                    </div>
                    {mriResult.visualizations[mriPlane] ? (
                      <div style={{
                        position: 'relative', width: '100%', height: 300,
                        borderRadius: 8, overflow: 'hidden',
                        background: '#0f0f14',
                        border: '1px solid hsl(var(--border))',
                      }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={mriResult.visualizations[mriPlane]!}
                          alt={`${mriPlane} Grad-CAM overlay`}
                          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                        />
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', padding: 40, color: 'hsl(var(--text-3))' }}>
                        <p>No {mriPlane} slice available</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Heatmap stats */}
                {mriResult.heatmap_stats && (
                  <div className="grid-3">
                    {Object.entries(mriResult.heatmap_stats).filter(([k]) => k !== 'shape').map(([k, v]) => (
                      <div key={k} className="card card-sm" style={{ background: 'hsl(var(--bg-1))', textAlign: 'center' }}>
                        <p className="text-xs text-muted" style={{ marginBottom: 4, textTransform: 'capitalize' }}>{k}</p>
                        <p style={{ fontWeight: 700 }}>{typeof v === 'number' ? v.toFixed(4) : v}</p>
                      </div>
                    ))}
                  </div>
                )}

                {mriResult.disclaimer && (
                  <p className="text-xs text-muted" style={{ lineHeight: 1.6 }}>
                    {mriResult.disclaimer}
                  </p>
                )}
              </div>
            )}

            {/* Initial state */}
            {!mriLoading && !mriResult && !mriError && (
              <div style={{ textAlign: 'center', padding: '32px 0', color: 'hsl(var(--text-3))' }}>
                <p style={{ fontSize: '2rem', marginBottom: 8 }}>🔬</p>
                <p>Upload a T1-weighted NIfTI MRI file to generate a real 3D Grad-CAM heatmap.</p>
                <p className="text-xs" style={{ marginTop: 8 }}>
                  Requires: CNN3D checkpoint (<code>scripts/train_cnn3d.py</code>) + backend running
                </p>
              </div>
            )}

            {/* Pathway status when no result yet */}
            {!mriResult && (
              <div className="card" style={{ background: 'hsl(var(--bg-1))', marginTop: 16, padding: 20 }}>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 12 }}>
                  Grad-CAM Pipeline
                </h3>
                <div className="flex-col gap-2" style={{ fontSize: '0.8125rem' }}>
                  {[
                    { step: '1', label: 'NIfTI upload + validation', desc: 'nibabel reads real MRI data' },
                    { step: '2', label: 'Preprocessing → 64³ tensor', desc: 'z-score norm + trilinear resize' },
                    { step: '3', label: 'MRICerebroNet forward pass', desc: '4-class CDR logits' },
                    { step: '4', label: 'Grad-CAM backward pass', desc: 'Real gradients on last Conv3d' },
                    { step: '5', label: 'Heatmap upsample + normalize', desc: 'Trilinear → input shape [0,1]' },
                    { step: '6', label: 'Axial / Sagittal / Coronal PNGs', desc: 'Base64 overlay images' },
                  ].map(s => (
                    <div key={s.step} className="flex items-center gap-3">
                      <span style={{
                        width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '0.6875rem', fontWeight: 700,
                        background: 'hsl(var(--pastel-peach))', color: 'hsl(10 60% 40%)',
                      }}>{s.step}</span>
                      <div>
                        <span style={{ fontWeight: 600 }}>{s.label}</span>
                        <span style={{ color: 'hsl(var(--text-3))', marginLeft: 8 }}>{s.desc}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

      </div>
    </>
  );
}
