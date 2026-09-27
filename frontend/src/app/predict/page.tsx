'use client';

import { useState, useRef } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

const CDR_COLORS: Record<number, string> = {
  0: 'hsl(var(--emerald))',
  1: 'hsl(var(--amber))',
  2: 'hsl(10 70% 65%)', // Muted Coral
  3: 'hsl(var(--rose))',
};

const CDR_VERDICT_CLASS: Record<number, string> = {
  0: 'verdict-normal',
  1: 'verdict-verymild',
  2: 'verdict-mild',
  3: 'verdict-moderate',
};

interface Visit {
  age: string;
  educ: string;
  ses: string;
  mmse: string;
  cdr: string;
  nwbv: string;
  etiv: string;
  asf: string;
  apoe4: boolean;
  p_tau: string;
}

interface PredictionResult {
  predicted_cdr_class: number;
  predicted_cdr_value: number;
  predicted_cdr_label: string;
  class_probabilities: Record<string, number>;
  n_visits_used: number;
  model: string;
  apoe4_adjusted?: boolean;
  ptau_adjusted?: boolean;
}

const EMPTY_VISIT: Visit = {
  age: '72', educ: '12', ses: '2', mmse: '28',
  cdr: '0', nwbv: '0.75', etiv: '1500', asf: '1.2',
  apoe4: false, p_tau: '',
};

const SAMPLE_PATIENT: Visit[] = [
  { age: '70', educ: '14', ses: '2', mmse: '29', cdr: '0',   nwbv: '0.78', etiv: '1520', asf: '1.18', apoe4: false, p_tau: '15.2' },
  { age: '72', educ: '14', ses: '2', mmse: '27', cdr: '0.5', nwbv: '0.76', etiv: '1515', asf: '1.19', apoe4: true, p_tau: '24.5' },
];

const FIELD_META: { key: keyof Visit; label: string; min?: string; max?: string; step?: string; unit?: string }[] = [
  { key: 'age',   label: 'Age',       min: '18',  max: '100', step: '1',    unit: 'yrs' },
  { key: 'educ',  label: 'Education', min: '1',   max: '23',  step: '1',    unit: 'yrs' },
  { key: 'ses',   label: 'SES',       min: '1',   max: '5',   step: '1'   },
  { key: 'mmse',  label: 'MMSE',      min: '0',   max: '30',  step: '1'   },
  { key: 'cdr',   label: 'CDR',       min: '0',   max: '3',   step: '0.5' },
  { key: 'nwbv',  label: 'nWBV',      min: '0.5', max: '1.0', step: '0.001' },
  { key: 'etiv',  label: 'eTIV',      min: '900', max: '2000',step: '1',    unit: 'mm³' },
  { key: 'asf',   label: 'ASF',       min: '0.8', max: '1.8', step: '0.001' },
];

export default function PredictPage() {
  const [visits, setVisits] = useState<Visit[]>([{ ...EMPTY_VISIT }]);
  const [subjectId, setSubjectId] = useState('OAS2_DEMO');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadingMRI, setUploadingMRI] = useState(false);
  const [mriInfo, setMriInfo] = useState<{
    filename: string;
    nwbv: number;
    etiv: number;
    asf: number;
    original_shape: number[];
    brain_fraction: number;
    cnn_status: string;
    warnings: string[];
    pipeline_steps_completed: string[];
  } | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addVisit = () =>
    setVisits((v) => [...v, { ...EMPTY_VISIT, age: String(Number(v[v.length - 1].age) + 2) }]);

  const removeVisit = (i: number) =>
    setVisits((v) => v.filter((_, idx) => idx !== i));

  const updateVisit = (i: number, field: keyof Visit, val: string | boolean) =>
    setVisits((v) => v.map((visit, idx) => idx === i ? { ...visit, [field]: val } : visit));

  const loadSample = () => {
    setVisits(SAMPLE_PATIENT.map(v => ({ ...v })));
    setSubjectId('OAS2_SAMPLE');
    setResult(null);
    setError(null);
    setMriInfo(null);
  };

  const handleMRIUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate extension before even uploading
    const name = file.name.toLowerCase();
    if (!name.endsWith('.nii') && !name.endsWith('.nii.gz')) {
      setError('Invalid file type. Please upload a NIfTI file (.nii or .nii.gz).');
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    setUploadingMRI(true);
    setError(null);
    setMriInfo(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API}/mri/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        // Real error from backend pipeline
        const detail = data?.detail || data;
        const errMsg = detail?.error || detail?.message || `MRI processing failed (HTTP ${res.status})`;
        const errCode = detail?.error_code || 'UNKNOWN';
        throw new Error(`[${errCode}] ${errMsg}`);
      }

      // Real features from actual voxel data
      const feats = data.extracted_features;
      const proc = data.processing_details;

      updateVisit(visits.length - 1, 'nwbv', String(feats.nwbv ?? ''));
      updateVisit(visits.length - 1, 'etiv', String(feats.etiv ?? ''));
      updateVisit(visits.length - 1, 'asf', String(feats.asf ?? ''));

      // Store real MRI processing info for display
      setMriInfo({
        filename: data.filename,
        nwbv: feats.nwbv,
        etiv: feats.etiv,
        asf: feats.asf,
        original_shape: proc?.original_shape ?? [],
        brain_fraction: proc?.brain_fraction ?? feats.nwbv,
        cnn_status: data.cnn_status ?? 'Unknown',
        warnings: data.warnings ?? [],
        pipeline_steps_completed: data.pipeline_steps_completed ?? [],
      });

    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploadingMRI(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const predict = async (endpoint: 'clinical' | 'bimodal') => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formattedVisits = visits.map((v) => ({
        age: Number(v.age),
        educ: Number(v.educ),
        ses: Number(v.ses),
        mmse: Number(v.mmse),
        cdr: Number(v.cdr),
        nwbv: Number(v.nwbv),
        etiv: Number(v.etiv),
        asf: Number(v.asf),
        apoe4: v.apoe4,
        p_tau: v.p_tau ? Number(v.p_tau) : null,
      }));

      const body: Record<string, unknown> = {
        subject_id: subjectId,
        visits: formattedVisits,
      };

      if (endpoint === 'bimodal') {
        const latest = formattedVisits[formattedVisits.length - 1];
        body.mri = {
          nwbv: latest.nwbv,
          etiv: latest.etiv,
          asf: latest.asf,
          nwbv_delta: 0.0,
        };
      }

      const res = await fetch(`${API}/predict/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const probEntries = result
    ? Object.entries(result.class_probabilities).map(([label, prob], i) => ({ label, prob, i }))
    : [];

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Prediction Workstation</h1>
          <p className="page-subtitle">Multimodal clinical forecasting using Genetics and MRI Data</p>
        </div>
        <div className="flex gap-2">
          <button id="load-sample-btn" className="btn btn-secondary" onClick={loadSample}>
            Load Sample
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => predict('clinical')}
            disabled={loading}
          >
            {loading ? '⏳' : 'Predict (Clinical)'}
          </button>
          <button
            className="btn btn-accent"
            onClick={() => predict('bimodal')}
            disabled={loading}
          >
            {loading ? '⏳ Running...' : '⚡ Predict (NextGen Bimodal)'}
          </button>
        </div>
      </div>

      <div className="page-body">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 32, alignItems: 'start' }}>

          {/* ── Left: Form Sections ── */}
          <div className="flex-col gap-8">
            
            {/* SECTION 1: Patient Information */}
            <section>
              <h2 className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Section 1: Patient Information</h2>
              <div className="card card-sm">
                <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 16 }}>
                  <label htmlFor="subject-id" style={{ flexShrink: 0 }}>Subject ID</label>
                  <input
                    id="subject-id"
                    type="text"
                    value={subjectId}
                    onChange={(e) => setSubjectId(e.target.value)}
                    placeholder="e.g. OAS2_0001"
                  />
                </div>
              </div>
            </section>

            {/* SECTION 2: Imaging Modalities */}
            <section>
              <h2 className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Section 2: Imaging Modalities</h2>
              <div className="card card-sm flex-col gap-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p style={{ fontWeight: 600 }}>MRI Structural Scan</p>
                    <p className="text-xs text-muted">Upload T1-weighted NIfTI (.nii / .nii.gz) — real volume processing via nibabel</p>
                  </div>
                  <div className="flex gap-2 items-center">
                    <input
                      type="file"
                      accept=".nii,.nii.gz"
                      ref={fileInputRef}
                      style={{ display: 'none' }}
                      onChange={handleMRIUpload}
                    />
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploadingMRI}
                    >
                      {uploadingMRI ? '⏳ Processing MRI...' : '📤 Upload MRI (.nii)'}
                    </button>
                  </div>
                </div>

                {/* Real MRI processing result panel */}
                {mriInfo && (
                  <div className="animate-in" style={{
                    background: 'hsl(var(--bg-2))', borderRadius: 8, padding: 14,
                    border: '1px solid hsl(var(--border))', fontSize: '0.8125rem'
                  }}>
                    <div className="flex items-center gap-2" style={{ marginBottom: 10 }}>
                      <span style={{ fontSize: '1rem' }}>✅</span>
                      <span style={{ fontWeight: 600, color: 'hsl(var(--emerald))' }}>Real NIfTI processed: {mriInfo.filename}</span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 10 }}>
                      <div style={{ background: 'hsl(var(--bg-1))', borderRadius: 6, padding: 8, textAlign: 'center' }}>
                        <p className="text-xs text-muted">nWBV (voxel-derived)</p>
                        <p style={{ fontWeight: 700, color: 'hsl(var(--accent))' }}>{mriInfo.nwbv.toFixed(4)}</p>
                      </div>
                      <div style={{ background: 'hsl(var(--bg-1))', borderRadius: 6, padding: 8, textAlign: 'center' }}>
                        <p className="text-xs text-muted">eTIV (estimated mm³)</p>
                        <p style={{ fontWeight: 700, color: 'hsl(var(--accent))' }}>{mriInfo.etiv?.toFixed(0) ?? 'N/A'}</p>
                      </div>
                      <div style={{ background: 'hsl(var(--bg-1))', borderRadius: 6, padding: 8, textAlign: 'center' }}>
                        <p className="text-xs text-muted">Original Shape</p>
                        <p style={{ fontWeight: 700, fontSize: '0.75rem' }}>{mriInfo.original_shape.join('×') || 'N/A'}</p>
                      </div>
                    </div>

                    <div style={{ marginBottom: 8 }}>
                      <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Pipeline steps completed:</p>
                      <div className="flex" style={{ flexWrap: 'wrap', gap: 4 }}>
                        {mriInfo.pipeline_steps_completed.map((s) => (
                          <span key={s} className="badge badge-emerald" style={{ fontSize: '0.6875rem' }}>✓ {s.replace(/_/g, ' ')}</span>
                        ))}
                      </div>
                    </div>

                    {mriInfo.cnn_status.includes('NOT_AVAILABLE') && (
                      <div className="callout callout-warning" style={{ padding: '8px 12px', marginTop: 6 }}>
                        <span style={{ fontSize: '0.875rem' }}>⚠️</span>
                        <p className="text-xs">CNN3D embedding unavailable (no trained checkpoint). Using nWBV proxy for bimodal prediction.</p>
                      </div>
                    )}

                    {mriInfo.warnings.length > 0 && (
                      <div style={{ marginTop: 6 }}>
                        {mriInfo.warnings.map((w, i) => (
                          <p key={i} className="text-xs text-muted">⚠ {w}</p>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>


            {/* SECTION 3: Clinical History */}
            <section>
              <h2 className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Section 3: Clinical History</h2>
              <div className="flex-col gap-4">
                {visits.map((visit, idx) => (
                  <div key={idx} className="card animate-in" style={{ position: 'relative' }}>
                    <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
                      <div className="flex items-center gap-2">
                        <span className="badge badge-accent">Visit {idx + 1}</span>
                        {idx === visits.length - 1 && visits.length > 1 && (
                          <span className="badge badge-neutral">Latest</span>
                        )}
                      </div>
                      {visits.length > 1 && (
                        <button className="btn btn-ghost btn-sm" onClick={() => removeVisit(idx)}>✕</button>
                      )}
                    </div>

                    <div className="grid-4" style={{ gap: 12 }}>
                      {FIELD_META.map(({ key, label, min, max, step, unit }) => (
                        <div key={key} className="form-group">
                          <label>{label}{unit ? ` (${unit})` : ''}</label>
                          <input
                            type="number"
                            min={min} max={max} step={step}
                            value={visit[key] as string}
                            onChange={(e) => updateVisit(idx, key, e.target.value)}
                          />
                        </div>
                      ))}
                    </div>
                    
                    {/* Biomarkers Row */}
                    <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid hsl(var(--border))' }}>
                      <h4 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'hsl(var(--accent))', marginBottom: 12, letterSpacing: '0.05em' }}>
                        NextGen Biomarkers
                      </h4>
                      <div className="flex gap-4">
                        <div className="form-group flex-row items-center gap-2">
                          <input 
                            type="checkbox" 
                            id={`apoe4-${idx}`}
                            checked={visit.apoe4}
                            onChange={(e) => updateVisit(idx, 'apoe4', e.target.checked)}
                          />
                          <label htmlFor={`apoe4-${idx}`} style={{ margin: 0, fontWeight: 500 }}>APOE4 Carrier</label>
                        </div>
                        <div className="form-group flex-row items-center gap-2 ml-4">
                          <label style={{ margin: 0 }}>p-tau181 (pg/mL):</label>
                          <input 
                            type="number" 
                            step="0.1"
                            style={{ width: 100 }}
                            value={visit.p_tau}
                            placeholder="e.g. 24.5"
                            onChange={(e) => updateVisit(idx, 'p_tau', e.target.value)}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                <button className="btn btn-secondary" onClick={addVisit} style={{ alignSelf: 'flex-start' }} disabled={visits.length >= 8}>
                  + Add Visit {visits.length >= 8 ? '(max)' : `(${visits.length}/8)`}
                </button>
              </div>
            </section>
          </div>

          {/* ── Right: Result Panel (Section 4) ── */}
          <div className="flex-col gap-4">
            <h2 className="text-sm font-bold text-muted" style={{ marginBottom: '0', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Section 4: Prediction</h2>
            
            {!result && !loading && !error && (
              <div className="card" style={{ textAlign: 'center', padding: 40, borderStyle: 'dashed' }}>
                <p style={{ fontSize: '2.5rem', marginBottom: 12 }}>🧠</p>
                <p style={{ fontWeight: 600, marginBottom: 6, color: 'hsl(var(--text-1))' }}>Awaiting Input</p>
                <p className="text-muted text-sm">Enter clinical and biomarker data and click Predict.</p>
              </div>
            )}

            {loading && (
              <div className="card" style={{ textAlign: 'center', padding: 40 }}>
                <p style={{ fontSize: '2rem', marginBottom: 12 }}>⏳</p>
                <p className="text-sm text-muted">Running NextGen Inference…</p>
                <div className="loading-shimmer" style={{ height: 8, marginTop: 16, borderRadius: 4 }} />
              </div>
            )}

            {error && (
              <div className="callout callout-error">
                <span>⚠️</span>
                <div>
                  <strong>Prediction Failed</strong>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            )}

            {result && (
              <>
                <div className={`verdict-card ${CDR_VERDICT_CLASS[result.predicted_cdr_class]} animate-in`}>
                  <p className="text-xs text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.1em' }}>Predicted Next-Visit CDR</p>
                  <p className="verdict-cdr-value" style={{ color: CDR_COLORS[result.predicted_cdr_class] }}>{result.predicted_cdr_value.toFixed(2)}</p>
                  <p className="verdict-cdr-label" style={{ color: CDR_COLORS[result.predicted_cdr_class] }}>{result.predicted_cdr_label}</p>
                </div>

                <div className="card animate-in animate-delay-1">
                  <h3 style={{ marginBottom: 16, fontSize: '0.875rem', fontWeight: 600 }}>Class Probabilities</h3>
                  <div className="prob-bar-wrap">
                    {probEntries.map(({ label, prob, i }) => (
                      <div key={label} className="prob-row">
                        <span className="prob-label">{label.replace(' (CDR', ' — CDR')}</span>
                        <div className="prob-track">
                          <div
                            className={`prob-fill cdr-${[0, 0.5, 1, 2][i]?.toString().replace('.', '-') ?? '0'}`}
                            style={{ width: `${(prob * 100).toFixed(1)}%`, background: CDR_COLORS[i] }}
                          />
                        </div>
                        <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="card card-sm animate-in animate-delay-2">
                  <p className="text-xs text-muted" style={{ marginBottom: 6 }}>Model Trace</p>
                  <p className="font-mono text-sm" style={{ color: 'hsl(var(--accent))', marginBottom: 4 }}>{result.model}</p>
                  {result.apoe4_adjusted && <p className="text-xs" style={{ color: 'hsl(var(--amber))' }}>⚡ APOE4 Risk Adjustment Applied</p>}
                  {result.ptau_adjusted && <p className="text-xs" style={{ color: 'hsl(var(--rose))' }}>⚡ p-tau181 Risk Adjustment Applied</p>}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
