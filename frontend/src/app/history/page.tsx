'use client';

import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface HistoryRecord {
  id: number;
  patient_id: string;
  predicted_cdr_class: number;
  predicted_cdr_value: number;
  predicted_cdr_label: string;
  mmse: number | null;
  cdr: number | null;
  age: number | null;
  timestamp: string;
}

const CDR_BADGE: Record<number, string> = {
  0: 'badge-emerald',
  1: 'badge-amber',
  2: 'badge-rose',
  3: 'badge-rose',
};

const CDR_COLOR: Record<number, string> = {
  0: 'hsl(var(--emerald))',
  1: 'hsl(var(--amber))',
  2: 'hsl(10 70% 65%)',
  3: 'hsl(var(--rose))',
};

export default function HistoryPage() {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/history/`)
      .then((r) => r.json())
      .then(setRecords)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const clearHistory = async () => {
    if (!confirm('Clear all prediction records? This cannot be undone.')) return;
    await fetch(`${API}/history/`, { method: 'DELETE' }).catch(() => {});
    setRecords([]);
  };

  const formatDate = (ts: string) => {
    try {
      return new Date(ts).toLocaleString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch { return ts; }
  };

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Prediction Log</h1>
          <p className="page-subtitle">All CDR predictions saved to the research database</p>
        </div>
        <div className="flex gap-2">
          <span className="badge badge-neutral">{records.length} records</span>
          {records.length > 0 && (
            <button id="clear-history-btn" className="btn btn-ghost btn-sm" onClick={clearHistory}>
              🗑 Clear Log
            </button>
          )}
        </div>
      </div>

      <div className="page-body">

        {loading && (
          <div className="flex-col gap-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="loading-shimmer" style={{ height: 56, borderRadius: 10 }} />
            ))}
          </div>
        )}

        {error && (
          <div className="callout callout-error" id="history-error">
            <span>⚠️</span>
            <div>
              <strong>Could not load history</strong>
              <p className="text-sm" style={{ marginTop: 4 }}>{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && records.length === 0 && (
          <div className="card" style={{ textAlign: 'center', padding: 60 }}>
            <p style={{ fontSize: '3rem', marginBottom: 12 }}>📋</p>
            <p className="font-bold" style={{ marginBottom: 6 }}>No Predictions Yet</p>
            <p className="text-sm text-muted">
              Run a prediction on the <a href="/predict" style={{ color: 'hsl(var(--accent))' }}>Predict CDR</a> page
              to see records here.
            </p>
          </div>
        )}

        {!loading && !error && records.length > 0 && (
          <div className="table-wrap animate-in" id="history-table">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Subject ID</th>
                  <th>Predicted CDR</th>
                  <th>Age</th>
                  <th>MMSE</th>
                  <th>Input CDR</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {records.map((rec, idx) => (
                  <tr key={rec.id} id={`history-row-${rec.id}`}>
                    <td className="text-muted font-mono">{idx + 1}</td>
                    <td className="td-highlight">{rec.patient_id}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span
                          style={{
                            width: 8, height: 8, borderRadius: '50%',
                            background: CDR_COLOR[rec.predicted_cdr_class],
                            flexShrink: 0,
                          }}
                        />
                        <span className={`badge ${CDR_BADGE[rec.predicted_cdr_class]}`}>
                          CDR {rec.predicted_cdr_value}
                        </span>
                        <span className="text-xs text-muted truncate" style={{ maxWidth: 160 }}>
                          {rec.predicted_cdr_label}
                        </span>
                      </div>
                    </td>
                    <td>{rec.age ?? '—'}</td>
                    <td>{rec.mmse ?? '—'}</td>
                    <td>{rec.cdr ?? '—'}</td>
                    <td className="text-muted text-xs font-mono">{formatDate(rec.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </>
  );
}
