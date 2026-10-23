'use client';

import { useState } from 'react';
import type { Metadata } from 'next';

const API = 'http://127.0.0.1:8000';

interface EEGResult {
  predicted_class: number;
  predicted_label: string;
  class_probabilities: Record<string, number>;
  model: string;
  dataset: string;
  disclaimer: string;
}

const SAMPLE_HEALTHY = {
  global_delta: 1.2e-10,
  global_theta: 0.8e-10,
  global_alpha: 3.5e-10,
  global_beta: 1.1e-10,
  global_gamma: 0.3e-10,
  rel_global_delta: 0.15,
  rel_global_theta: 0.10,
  rel_global_alpha: 0.45,
  rel_global_beta: 0.20,
  rel_global_gamma: 0.05,
  theta_alpha_ratio: 0.22,
  spectral_edge_freq_95: 28.5,
};

const SAMPLE_DISEASE = {
  global_delta: 4.8e-10,
  global_theta: 3.2e-10,
  global_alpha: 1.1e-10,
  global_beta: 0.6e-10,
  global_gamma: 0.2e-10,
  rel_global_delta: 0.48,
  rel_global_theta: 0.32,
  rel_global_alpha: 0.11,
  rel_global_beta: 0.06,
  rel_global_gamma: 0.03,
  theta_alpha_ratio: 2.91,
  spectral_edge_freq_95: 14.2,
};

const FIELD_GROUPS = [
  {
    label: 'Absolute Band Power',
    color: 'hsl(220 70% 60%)',
    fields: [
      { key: 'global_delta', label: 'Delta (1–4 Hz)', step: '1e-12' },
      { key: 'global_theta', label: 'Theta (4–8 Hz)', step: '1e-12' },
      { key: 'global_alpha', label: 'Alpha (8–13 Hz)', step: '1e-12' },
      { key: 'global_beta',  label: 'Beta (13–30 Hz)', step: '1e-12' },
      { key: 'global_gamma', label: 'Gamma (30–45 Hz)', step: '1e-12' },
    ],
  },
  {
    label: 'Relative Band Power',
    color: 'hsl(160 60% 55%)',
    fields: [
      { key: 'rel_global_delta', label: 'Rel. Delta', step: '0.01' },
      { key: 'rel_global_theta', label: 'Rel. Theta', step: '0.01' },
      { key: 'rel_global_alpha', label: 'Rel. Alpha', step: '0.01' },
      { key: 'rel_global_beta',  label: 'Rel. Beta',  step: '0.01' },
      { key: 'rel_global_gamma', label: 'Rel. Gamma', step: '0.01' },
    ],
  },
  {
    label: 'Derived Metrics',
    color: 'hsl(38 70% 60%)',
    fields: [
      { key: 'theta_alpha_ratio',    label: 'Theta/Alpha Ratio', step: '0.01' },
      { key: 'spectral_edge_freq_95', label: 'SEF95 (Hz)',        step: '0.1'  },
    ],
  },
];

export default function EEGPage() {
  const [form, setForm] = useState<Record<string, string>>(
    Object.fromEntries(Object.entries(SAMPLE_HEALTHY).map(([k, v]) => [k, String(v)]))
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EEGResult | null>(null);
  const [error, setError]   = useState<string | null>(null);

  const loadSample = (sample: Record<string, number>) => {
    setForm(Object.fromEntries(Object.entries(sample).map(([k, v]) => [k, String(v)])));
    setResult(null);
    setError(null);
  };

  const runScreen = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, parseFloat(v)])
      );
      const res = await fetch(`${API}/eeg/screen`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        const d = data.detail || data;
        throw new Error(typeof d === 'string' ? d : d.message || d.error || JSON.stringify(d));
      }
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const isDisease = result?.predicted_class === 0;
  const resultColor = isDisease ? 'hsl(0 70% 65%)' : 'hsl(142 60% 55%)';

  return (
    <>
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">EEG Disease Screener</h1>
          <p className="page-subtitle">
            Spectral band power analysis · OpenNeuro ds004504 · Binary AD/FTD vs Control
          </p>
        </div>
        <div className="flex gap-2">
          <button className="btn btn-secondary" onClick={() => loadSample(SAMPLE_HEALTHY)}>
            Load Healthy Sample
          </button>
          <button className="btn btn-secondary" onClick={() => loadSample(SAMPLE_DISEASE)}>
            Load Disease Sample
          </button>
          <button
            id="run-eeg-btn"
            className="btn btn-accent"
            onClick={runScreen}
            disabled={loading}
          >
            {loading ? '⏳ Screening…' : '⚡ Run EEG Screen'}
          </button>
        </div>
      </div>

      <div className="page-body">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: 32, alignItems: 'start' }}>

          {/* ── Left: Input Form ── */}
          <div className="flex-col gap-6">

            {/* Disclaimer */}
            <div className="callout callout-warning animate-in" role="alert">
              <span>⚠️</span>
              <div>
                <strong>Separate cohort.</strong> This EEG model was trained on{' '}
                <strong>OpenNeuro ds004504</strong> (87 subjects — AD, FTD, Control).{' '}
                It is a <em>different patient cohort</em> from OASIS-2 and cannot be directly
                compared with CDR predictions. Binary accuracy: 50% | Balanced accuracy: 47.8%.{' '}
                Research use only.
              </div>
            </div>

            {/* Band Power Inputs */}
            {FIELD_GROUPS.map((group) => (
              <section key={group.label}>
                <h2
                  className="text-sm font-bold"
                  style={{
                    marginBottom: 12, textTransform: 'uppercase',
                    letterSpacing: '0.07em', color: group.color,
                  }}
                >
                  {group.label}
                </h2>
                <div className="card">
                  <div className="grid-2" style={{ gap: 12 }}>
                    {group.fields.map(({ key, label, step }) => (
                      <div key={key} className="form-group">
                        <label>{label}</label>
                        <input
                          type="number"
                          step={step}
                          value={form[key] ?? ''}
                          onChange={(e) =>
                            setForm((f) => ({ ...f, [key]: e.target.value }))
                          }
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            ))}

            {/* EEG band guide */}
            <div className="card card-sm" style={{ background: 'hsl(var(--bg-2))' }}>
              <h3 style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: 10 }}>
                EEG Band Interpretation Guide
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
                {[
                  { band: 'Delta', hz: '1–4 Hz', ad: '↑ in AD', color: 'hsl(0 60% 60%)' },
                  { band: 'Theta', hz: '4–8 Hz', ad: '↑ in AD', color: 'hsl(30 70% 60%)' },
                  { band: 'Alpha', hz: '8–13 Hz', ad: '↓ in AD', color: 'hsl(142 60% 55%)' },
                  { band: 'Beta',  hz: '13–30 Hz', ad: '↓ in AD', color: 'hsl(200 65% 60%)' },
                  { band: 'Gamma', hz: '30–45 Hz', ad: '↓ in AD', color: 'hsl(260 60% 65%)' },
                ].map((b) => (
                  <div key={b.band} style={{
                    background: 'hsl(var(--bg-1))', borderRadius: 8, padding: 10,
                    borderTop: `3px solid ${b.color}`,
                  }}>
                    <p style={{ fontWeight: 700, fontSize: '0.8125rem', marginBottom: 2 }}>{b.band}</p>
                    <p style={{ fontSize: '0.6875rem', color: 'hsl(var(--text-3))', marginBottom: 4 }}>{b.hz}</p>
                    <p style={{ fontSize: '0.6875rem', color: b.color, fontWeight: 600 }}>{b.ad}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Right: Results Panel ── */}
          <div className="flex-col gap-4">
            <h2
              className="text-sm font-bold text-muted"
              style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}
            >
              Screening Result
            </h2>

            {!result && !loading && !error && (
              <div className="card" style={{ textAlign: 'center', padding: 48, borderStyle: 'dashed' }}>
                <p style={{ fontSize: '2.5rem', marginBottom: 12 }}>🧠</p>
                <p style={{ fontWeight: 600, marginBottom: 6 }}>Awaiting EEG Input</p>
                <p className="text-sm text-muted">
                  Load a sample or enter band power values and click Run EEG Screen.
                </p>
              </div>
            )}

            {loading && (
              <div className="card" style={{ textAlign: 'center', padding: 48 }}>
                <p style={{ fontSize: '2rem', marginBottom: 12 }}>⏳</p>
                <p className="text-sm text-muted">Running spectral classifier…</p>
                <div className="loading-shimmer" style={{ height: 8, marginTop: 16, borderRadius: 4 }} />
              </div>
            )}

            {error && (
              <div className="callout callout-error">
                <span>⚠️</span>
                <div>
                  <strong>Screening Failed</strong>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            )}

            {result && (
              <>
                {/* Verdict card */}
                <div className="card animate-in" style={{
                  padding: 28, textAlign: 'center',
                  background: `${resultColor}12`,
                  border: `1.5px solid ${resultColor}44`,
                }}>
                  <p style={{ fontSize: '2.5rem', marginBottom: 8 }}>
                    {isDisease ? '🔴' : '🟢'}
                  </p>
                  <p className="text-xs text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>
                    EEG Screening Result
                  </p>
                  <p style={{ fontWeight: 800, fontSize: '1.25rem', color: resultColor, marginBottom: 4 }}>
                    {result.predicted_label}
                  </p>
                  <p className="text-xs text-muted">Binary classification (AD/FTD vs Control)</p>
                </div>

                {/* Probability bars */}
                <div className="card animate-in animate-delay-1">
                  <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: 14 }}>
                    Class Probabilities
                  </h3>
                  <div className="flex-col gap-3">
                    {Object.entries(result.class_probabilities).map(([label, prob]) => {
                      const isSelected = label === result.predicted_label;
                      const col = label.includes('Disease') ? 'hsl(0 70% 65%)' : 'hsl(142 60% 55%)';
                      return (
                        <div key={label}>
                          <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
                            <span style={{
                              fontSize: '0.8125rem', flex: 1,
                              fontWeight: isSelected ? 700 : 400,
                              color: isSelected ? col : 'hsl(var(--text-2))',
                            }}>
                              {label}
                            </span>
                            <span style={{ fontWeight: 700, fontSize: '0.875rem', color: col }}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div style={{ height: 8, background: 'hsl(var(--bg-3))', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              height: '100%', width: `${(prob * 100).toFixed(1)}%`,
                              background: col, borderRadius: 4,
                              transition: 'width 0.6s ease',
                            }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Model info */}
                <div className="card card-sm animate-in animate-delay-2" style={{ background: 'hsl(var(--bg-2))' }}>
                  <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Model</p>
                  <p className="font-mono text-sm" style={{ color: 'hsl(var(--accent))', marginBottom: 8 }}>
                    {result.model}
                  </p>
                  <p className="text-xs text-muted" style={{ marginBottom: 4 }}>Dataset</p>
                  <p style={{ fontSize: '0.8125rem', marginBottom: 8 }}>{result.dataset}</p>
                  <p style={{ fontSize: '0.75rem', color: 'hsl(var(--text-3))', lineHeight: 1.6 }}>
                    {result.disclaimer}
                  </p>
                </div>
              </>
            )}

            {/* How it works card */}
            <div className="card card-sm" style={{ background: 'hsl(var(--bg-2))' }}>
              <h4 style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: 10 }}>How It Works</h4>
              <div className="flex-col gap-2" style={{ fontSize: '0.75rem', color: 'hsl(var(--text-3))' }}>
                {[
                  '1. Enter 12 EEG spectral features',
                  '2. SimpleImputer fills any missing values',
                  '3. StandardScaler normalizes features',
                  '4. EEGSpectralEncoder (107-dim MLP) classifies',
                  '5. Binary result: Disease vs Control',
                ].map((s) => (
                  <p key={s}>{s}</p>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
