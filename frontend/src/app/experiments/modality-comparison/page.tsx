'use client';

import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface MultimodalResult {
  experiment_id: string;
  model_name: string;
  modalities: string;
  dataset: string;
  status: string;
  notes: string;
  accuracy: number | null;
  balanced_accuracy: number | null;
  macro_f1: number | null;
}

function MetricValue({ value }: { value: number | null }) {
  if (value === null || value === undefined) return <span className="text-muted">N/A</span>;
  return <strong className="td-highlight">{(value * 100).toFixed(1)}%</strong>;
}

export default function ModalityComparisonPage() {
  const [results, setResults] = useState<MultimodalResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/experiments/multimodal/comparison`)
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load comparison data');
        return r.json();
      })
      .then((data: MultimodalResult[]) => {
        setResults(data);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Modality Comparison</h1>
          <p className="page-subtitle">Ablation study of Clinical, MRI, and EEG fusion</p>
        </div>
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
          <div className="callout callout-error">
            <span>⚠️</span>
            <div>
              <strong>Could not load experiments</strong>
              <p className="text-sm" style={{ marginTop: 4 }}>{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && (
          <div className="card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: 16 }}>Scientific Protocol Matrix</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Experiment</th>
                    <th>Modalities</th>
                    <th>Dataset</th>
                    <th>Status</th>
                    <th>Accuracy</th>
                    <th>Balanced Acc</th>
                    <th>Macro F1</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((res) => (
                    <tr key={res.experiment_id}>
                      <td style={{ fontWeight: 600 }}>{res.experiment_id}</td>
                      <td>{res.modalities}</td>
                      <td className="text-sm">{res.dataset}</td>
                      <td>
                        <span className={`badge ${res.status === 'VALIDATED' ? 'badge-emerald' : res.status === 'LEGACY' ? 'badge-amber' : 'badge-neutral'}`}>
                          {res.status}
                        </span>
                      </td>
                      <td><MetricValue value={res.accuracy} /></td>
                      <td><MetricValue value={res.balanced_accuracy} /></td>
                      <td><MetricValue value={res.macro_f1} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="callout callout-info" style={{ marginTop: 24 }}>
              <span>ℹ️</span>
              <div>
                <strong>Scientific Limitations</strong>
                <p className="text-sm text-muted" style={{ marginTop: 4 }}>
                  Experiments marked as UNAVAILABLE signify that no patient-aligned cohort exists to perform a scientifically valid multimodal fusion. The TriModal architecture scaffold is preserved in code, but execution on synthetic/unaligned data is strictly prohibited to prevent falsified accuracy claims.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
