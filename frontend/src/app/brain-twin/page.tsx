'use client';

import { useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface Visit {
  age: string;
  educ: string;
  ses: string;
  mmse: string;
  cdr: string;
  nwbv: string;
  etiv: string;
  asf: string;
}

interface BrainTwinResult {
  n_visits: number;
  z_dim: number;
  trajectories: number[][];
  pca_2d: [number, number][] | null;
}

const SAMPLE: Visit[] = [
  { age: '70', educ: '14', ses: '2', mmse: '29', cdr: '0',   nwbv: '0.78', etiv: '1520', asf: '1.18' },
  { age: '72', educ: '14', ses: '2', mmse: '27', cdr: '0.5', nwbv: '0.76', etiv: '1515', asf: '1.19' },
  { age: '74', educ: '14', ses: '2', mmse: '22', cdr: '1.0', nwbv: '0.72', etiv: '1510', asf: '1.21' },
];

const BrainSilhouette = () => (
  <svg viewBox="0 0 200 160" style={{ width: '100%', maxWidth: '280px', margin: '0 auto', display: 'block', opacity: 0.8 }}>
    {/* Stylized clinical brain silhouette */}
    <path 
      fill="hsl(var(--bg-3))" 
      d="M100 10 C60 10 30 35 30 75 C30 95 40 115 55 125 C65 135 75 145 85 150 L115 150 C125 145 135 135 145 125 C160 115 170 95 170 75 C170 35 140 10 100 10 Z" 
    />
    <path 
      fill="none" stroke="hsl(var(--text-3))" strokeWidth="2" strokeDasharray="4 4"
      d="M100 10 V150" 
    />
    {/* Highlighted regions representing Z_t activation */}
    <circle cx="70" cy="60" r="15" fill="hsl(var(--accent))" opacity="0.6" />
    <circle cx="120" cy="70" r="10" fill="hsl(var(--accent))" opacity="0.4" />
    <circle cx="100" cy="40" r="20" fill="hsl(var(--accent))" opacity="0.3" />
  </svg>
);

function MiniBarChart({ values, label }: { values: number[]; label: string }) {
  const max = Math.max(...values.map(Math.abs));
  return (
    <div>
      <p className="text-xs text-muted" style={{ marginBottom: 6 }}>{label}</p>
      <div className="flex items-end gap-1" style={{ height: 48 }}>
        {values.map((v, i) => (
          <div
            key={i}
            title={`Visit ${i + 1}: ${v.toFixed(3)}`}
            style={{
              flex: 1,
              height: `${Math.max(4, Math.abs(v / (max || 1)) * 48)}px`,
              background: v >= 0
                ? 'hsl(var(--pastel-sage))'
                : 'hsl(var(--pastel-peach))',
              borderRadius: '3px 3px 0 0',
              transition: 'height 0.4s ease',
            }}
          />
        ))}
      </div>
      <div className="flex" style={{ gap: 4 }}>
        {values.map((_, i) => (
          <div key={i} style={{ flex: 1, textAlign: 'center', fontSize: '0.625rem', color: 'hsl(var(--text-3))' }}>
            V{i + 1}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function BrainTwinPage() {
  const [visits] = useState<Visit[]>(SAMPLE.map(v => ({ ...v })));
  const [result, setResult] = useState<BrainTwinResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'trajectory' | 'heatmap'>('trajectory');

  const extract = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const body = {
        visits: visits.map((v) => ({
          age: Number(v.age), educ: Number(v.educ), ses: Number(v.ses),
          mmse: Number(v.mmse), cdr: Number(v.cdr), nwbv: Number(v.nwbv),
          etiv: Number(v.etiv), asf: Number(v.asf),
        })),
      };

      const res = await fetch(`${API}/brain-twin/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const dimStats = result
    ? Array.from({ length: Math.min(result.z_dim, 16) }, (_, dim) => ({
        dim,
        values: result.trajectories.map((t) => t[dim] ?? 0),
        mean: result.trajectories.reduce((s, t) => s + (t[dim] ?? 0), 0) / result.n_visits,
      }))
    : [];

  const dimChange = dimStats.map((d) => ({
    ...d,
    delta: d.values[d.values.length - 1] - d.values[0],
  }));

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Digital Brain Twin</h1>
          <p className="page-subtitle">Longitudinal representation of cognitive progression</p>
        </div>
        <button id="extract-btn" className="btn btn-accent" onClick={extract} disabled={loading}>
          {loading ? '⏳ Extracting…' : '🧩 Extract Z_t State'}
        </button>
      </div>

      <div className="page-body">

        {/* Info callout */}
        <div className="callout callout-info animate-in" style={{ marginBottom: 24 }}>
          <span>ℹ️</span>
          <div>
            <strong>What is Z_t?</strong>
            <p className="text-sm" style={{ marginTop: 4 }}>
              Z_t is the 64-dimensional hidden state of the Temporal GRU at each timestep — 
              the model&apos;s internal representation of the patient&apos;s brain health at that visit.
              Tracking Z_t across visits reveals how the brain state evolves over time.
            </p>
          </div>
        </div>

        <div className="grid-2">

          {/* ── Left Column: Patient & History ── */}
          <div className="flex-col gap-6">
            <h3 className="text-sm font-bold text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Patient Trajectory
            </h3>
            
            <div className="card">
              <div className="flex items-center justify-between" style={{ marginBottom: 20 }}>
                <div>
                  <p className="text-xs text-muted">PATIENT</p>
                  <p className="font-bold">OAS2_SAMPLE</p>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <p className="text-xs text-muted">CURRENT CDR</p>
                  <p className="font-bold" style={{ color: 'hsl(var(--accent))' }}>1.0</p>
                </div>
              </div>

              <div className="timeline">
                {visits.map((v, i) => {
                  const dummyBHI = Math.max(0, 85 - (i * 12) - (Number(v.cdr) * 15)).toFixed(1);
                  return (
                    <div key={i} className="timeline-item">
                      <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                        <p className="font-bold text-sm">Visit {i + 1}</p>
                        <span className="badge badge-accent" style={{ background: 'hsl(var(--accent) / 0.15)', color: 'hsl(var(--accent))' }}>BHI: {dummyBHI}</span>
                      </div>
                      <p className="text-xs text-muted">Age {v.age} • MMSE: {v.mmse} • CDR: {v.cdr} • nWBV: {v.nwbv}</p>
                    </div>
                  );
                })}
                <div className="timeline-item" style={{ opacity: 0.5 }}>
                  <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                    <p className="font-bold text-sm" style={{ color: 'hsl(var(--accent))' }}>Predicted Visit</p>
                    <span className="badge badge-neutral">BHI: Projected</span>
                  </div>
                  <p className="text-xs text-muted">Future trajectory...</p>
                </div>
              </div>
            </div>
          </div>

          {/* ── Right Column: Brain Twin Visualization ── */}
          <div className="flex-col gap-6">
            <h3 className="text-sm font-bold text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Brain State Visualization
            </h3>
            
            <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <BrainSilhouette />
              
              <div style={{ marginTop: 24, textAlign: 'center' }}>
                <p className="font-bold text-sm">Structural Representation</p>
                <p className="text-xs text-muted">MRI DATA NOT CONNECTED — Placeholder State</p>
              </div>
            </div>

            {/* Result panel */}
            {error && (
              <div className="callout callout-error" id="twin-error">
                <span>⚠️</span>
                <div>
                  <strong>Extraction Failed</strong>
                  <p className="text-sm" style={{ marginTop: 4 }}>{error}</p>
                </div>
              </div>
            )}

            {result && (
              <div className="card animate-in">
                <div className="tabs">
                  <button
                    id="tab-trajectory"
                    className={`tab${activeTab === 'trajectory' ? ' active' : ''}`}
                    onClick={() => setActiveTab('trajectory')}
                  >
                    Latent Trajectory
                  </button>
                  <button
                    id="tab-heatmap"
                    className={`tab${activeTab === 'heatmap' ? ' active' : ''}`}
                    onClick={() => setActiveTab('heatmap')}
                  >
                    Net Change
                  </button>
                </div>

                {activeTab === 'trajectory' && (
                  <div id="trajectory-view">
                    <div className="grid-3" style={{ gap: 12 }}>
                      {dimStats.slice(0, 9).map((d) => (
                        <div key={d.dim} className="card card-sm" id={`dim-chart-${d.dim}`}>
                          <MiniBarChart values={d.values} label={`Z[${d.dim}]`} />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'heatmap' && (
                  <div id="heatmap-view">
                    <div className="flex-col gap-3">
                      {dimChange
                        .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
                        .slice(0, 8)
                        .map((d) => (
                          <div key={d.dim} className="flex items-center gap-3">
                            <span className="font-mono text-xs" style={{ width: 48, flexShrink: 0 }}>Z[{d.dim}]</span>
                            <div style={{ flex: 1, height: 8, background: 'hsl(var(--bg-2))', borderRadius: 4, overflow: 'hidden' }}>
                              <div style={{
                                marginLeft: d.delta < 0 ? `${50 + d.delta / (Math.max(...dimChange.map(x => Math.abs(x.delta))) || 1) * 50}%` : '50%',
                                width: `${Math.abs(d.delta) / (Math.max(...dimChange.map(x => Math.abs(x.delta))) || 1) * 50}%`,
                                height: '100%',
                                background: d.delta >= 0 ? 'hsl(var(--pastel-sage))' : 'hsl(var(--pastel-peach))',
                                borderRadius: 4,
                              }} />
                            </div>
                            <span className="font-mono text-xs" style={{ width: 60, textAlign: 'right' }}>
                              {d.delta >= 0 ? '+' : ''}{d.delta.toFixed(3)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

        </div>
      </div>
    </>
  );
}
