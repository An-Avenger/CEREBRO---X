'use client';

import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface Experiment {
  experiment_id: string;
  phase: number;
  description: string;
  metrics: Record<string, unknown> | null;
  artifact_path: string;
}

const EXP_META: Record<string, { icon: string; color: string; phase: string }> = {
  'EXP-BASELINE-001':      { icon: '📊', color: 'badge-cyan',    phase: 'Phase 3' },
  'EXP-NN-BASELINE-001':   { icon: '🤖', color: 'badge-blue',    phase: 'Phase 3' },
  'EXP-MRI-SCALAR-001':    { icon: '🧠', color: 'badge-violet',  phase: 'Phase 4' },
  'EXP-MULTIMODAL-001':    { icon: '⚗',  color: 'badge-violet',  phase: 'Phase 4' },
  'EXP-FUSION-BIMODAL-001':{ icon: '🔀', color: 'badge-violet',  phase: 'Phase 5' },
  'EXP-LONGITUDINAL-001':  { icon: '⭐', color: 'badge-emerald', phase: 'Phase 5' },
  'EXP-BRAIN-TWIN-001':    { icon: '🧩', color: 'badge-cyan',    phase: 'Phase 4' },
  'EXP-EXPLAIN-001':       { icon: '🔍', color: 'badge-amber',   phase: 'Phase 6' },
  'EXP-PROGRESSION-001':   { icon: '📈', color: 'badge-violet',  phase: 'Phase 7' },
  'EXP-EEG-STANDALONE-001':{ icon: '🌊', color: 'badge-rose',    phase: 'Phase 4' },
};

function MetricValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  if (typeof value === 'number') {
    const pct = value > 0 && value <= 1;
    return <strong className="td-highlight">{pct ? `${(value * 100).toFixed(1)}%` : value.toFixed(4)}</strong>;
  }
  return <span>{String(value)}</span>;
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/experiments/`)
      .then((r) => r.json())
      .then((data: Experiment[]) => {
        setExperiments(data);
        if (data.length > 0) setSelected(data[0].experiment_id);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const selectedExp = experiments.find((e) => e.experiment_id === selected);

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Experiments</h1>
          <p className="page-subtitle">All recorded experiment runs and their evaluation metrics</p>
        </div>
        <span className="badge badge-neutral">{experiments.length} experiments</span>
      </div>

      <div className="page-body">
        {loading && (
          <div className="flex-col gap-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="loading-shimmer" style={{ height: 72, borderRadius: 10 }} />
            ))}
          </div>
        )}

        {error && (
          <div className="callout callout-error" id="exp-error">
            <span>⚠️</span>
            <div>
              <strong>Could not load experiments</strong>
              <p className="text-sm" style={{ marginTop: 4 }}>{error}</p>
              <p className="text-xs text-muted" style={{ marginTop: 4 }}>Start the API: <code>python scripts/run_api.py</code></p>
            </div>
          </div>
        )}

        {!loading && !error && (
          <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 24, alignItems: 'start' }}>

            {/* Experiment list */}
            <div className="flex-col gap-2">
              {experiments.map((exp) => {
                const meta = EXP_META[exp.experiment_id] ?? { icon: '⚗', color: 'badge-cyan', phase: '' };
                return (
                  <button
                    key={exp.experiment_id}
                    id={`exp-btn-${exp.experiment_id}`}
                    className="card card-sm"
                    onClick={() => setSelected(exp.experiment_id)}
                    style={{
                      textAlign: 'left',
                      cursor: 'pointer',
                      border: selected === exp.experiment_id
                        ? '1px solid hsl(var(--accent))'
                        : '1px solid hsl(var(--border))',
                      background: selected === exp.experiment_id ? 'hsl(var(--bg-0))' : 'hsl(var(--bg-1))',
                      transition: 'all 0.15s',
                      width: '100%',
                    }}
                  >
                    <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
                      <span>{meta.icon}</span>
                      <span className={`badge ${meta.color}`}>{meta.phase}</span>
                    </div>
                    <p style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'hsl(var(--text-1))' }}>
                      {exp.experiment_id}
                    </p>
                    <p className="text-xs text-muted truncate" style={{ marginTop: 2 }}>
                      {exp.description || 'No description'}
                    </p>
                  </button>
                );
              })}
            </div>

            {/* Selected experiment detail */}
            {selectedExp && (
              <div className="flex-col gap-4 animate-in" id="exp-detail">
                <div className="card">
                  <div className="flex items-center gap-3" style={{ marginBottom: 16 }}>
                    <span style={{ fontSize: '1.75rem' }}>
                      {EXP_META[selectedExp.experiment_id]?.icon ?? '⚗'}
                    </span>
                    <div>
                      <h2 style={{ fontSize: '1.125rem', fontWeight: 700 }}>{selectedExp.experiment_id}</h2>
                      <p className="text-sm text-muted">{selectedExp.description || 'Cerebro-X experiment'}</p>
                    </div>
                    <span className={`badge ${EXP_META[selectedExp.experiment_id]?.color ?? 'badge-neutral'}`} style={{ marginLeft: 'auto' }}>
                      {EXP_META[selectedExp.experiment_id]?.phase}
                    </span>
                  </div>

                  {selectedExp.metrics ? (
                    <>
                      <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: 12, color: 'hsl(var(--text-2))' }}>
                        Recorded Metrics
                      </h3>
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Metric</th>
                              <th>Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(selectedExp.metrics).map(([k, v]) => (
                              <tr key={k}>
                                <td className="font-mono" style={{ color: 'hsl(var(--text-1))', fontWeight: 500 }}>{k}</td>
                                <td><MetricValue value={v} /></td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  ) : (
                    <div className="callout callout-info">
                      <span>ℹ️</span>
                      <p className="text-sm">No metrics.json found for this experiment.</p>
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        )}
      </div>
    </>
  );
}
