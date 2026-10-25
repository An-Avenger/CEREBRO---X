'use client';

import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface TrajectoryPoint {
  year: number;
  age: number;
  mmse: number;
  nwbv: number;
  predicted_cdr_class: number;
  predicted_cdr_label: string;
  dementia_risk_prob: number;
}

interface SimulationResponse {
  success: boolean;
  subject_id: string;
  simulation_horizon_years: number;
  intervention: {
    intervention_name: string;
    mmse_delta: number;
    nwbv_preservation_pct: number;
    ptau_reduction_pct: number;
    years_forward: number;
  };
  baseline_status: {
    cdr_class: number;
    cdr_label: string;
    probabilities: Record<string, number>;
  };
  trajectories: {
    natural_progression: TrajectoryPoint[];
    counterfactual_progression: TrajectoryPoint[];
  };
  summary: {
    projected_risk_reduction_pct: number;
    mmse_salvaged_points: number;
    nwbv_volume_protected: number;
    clinical_significance: string;
  };
}

const DEFAULT_PATIENT = {
  subject_id: 'OAS2_0002_DEMO',
  visits: [
    { age: 72.0, educ: 16.0, ses: 2.0, mmse: 28.0, cdr: 0.0, nwbv: 0.763, etiv: 1480.0, asf: 1.23, gender: 'F', hand: 'R', apoe4: false, p_tau: 16.5 },
    { age: 74.0, educ: 16.0, ses: 2.0, mmse: 26.0, cdr: 0.5, nwbv: 0.741, etiv: 1478.0, asf: 1.23, gender: 'F', hand: 'R', apoe4: true, p_tau: 24.8 },
  ]
};

export default function SimulationPage() {
  const [patient, setPatient] = useState(DEFAULT_PATIENT);
  const [interventionName, setInterventionName] = useState('Anti-Amyloid mAb + Intensive Cognitive Rehab');
  const [mmseBoost, setMmseBoost] = useState(3.0);
  const [nwbvProtection, setNwbvProtection] = useState(0.02);
  const [ptauReduction, setPtauReduction] = useState(0.25);
  const [yearsForward, setYearsForward] = useState(4);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SimulationResponse | null>(null);

  const [activeTab, setActiveTab] = useState<'trajectories' | 'radar' | 'report'>('trajectories');
  const [reportText, setReportText] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/simulation/what-if`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_id: patient.subject_id,
          visits: patient.visits,
          intervention: {
            intervention_name: interventionName,
            mmse_delta: Number(mmseBoost),
            nwbv_preservation_pct: Number(nwbvProtection),
            ptau_reduction_pct: Number(ptauReduction),
            years_forward: Number(yearsForward),
          }
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail?.message || data.detail || 'Simulation request failed.');
      }
      setResults(data);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation();
  }, []);

  const generateReport = async () => {
    setReportLoading(true);
    try {
      const res = await fetch(`${API}/simulation/clinical-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_id: patient.subject_id,
          clinician_name: 'Dr. C. Xavier, MD, PhD (Neurodegenerative Division)',
          institution: 'Cerebro-X Center for Digital Brain Twins',
          notes: `Counterfactual analysis evaluated for ${interventionName}. Patient shows positive potential for cognitive reserve preservation over ${yearsForward} years.`,
          visits: patient.visits,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Report generation failed');
      setReportText(data.markdown_report);
      setActiveTab('report');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setReportLoading(false);
    }
  };

  return (
    <>
      <div className="page-header">
        <div>
          <h1 className="page-title">Counterfactual "What-If" Simulation Engine</h1>
          <p className="page-subtitle">
            Simulate therapeutic interventions · Compare natural decline vs. cognitive stabilization · Longitudinal GRU
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={runSimulation}
            disabled={loading}
            className="btn btn-primary"
            style={{ minWidth: 160 }}
          >
            {loading ? 'Simulating...' : '▶ Run Simulation'}
          </button>
          <button
            onClick={generateReport}
            disabled={reportLoading}
            className="btn btn-ghost"
            style={{ border: '1px solid hsl(var(--border-subtle))' }}
          >
            {reportLoading ? 'Compiling...' : '📄 Generate Full Report'}
          </button>
        </div>
      </div>

      <div className="page-body flex-col gap-8">
        {error && (
          <div className="callout callout-error animate-in">
            <span>❌</span>
            <div><strong>Simulation Error:</strong> {error}</div>
          </div>
        )}

        {/* ── Top Grid: Configuration Panels ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 24 }}>
          {/* Patient Baseline */}
          <div className="card" style={{ padding: 24 }}>
            <h2 className="text-sm font-bold text-muted" style={{ textTransform: 'uppercase', marginBottom: 16, color: 'hsl(var(--violet))' }}>
              👤 Patient Case Context ({patient.subject_id})
            </h2>
            <div className="flex-col gap-3">
              <div className="flex justify-between text-sm py-1 border-b" style={{ borderColor: 'hsl(var(--border-subtle))' }}>
                <span className="text-muted">Chronological Visits:</span>
                <span className="font-semibold">{patient.visits.length} Longitudinal Checkpoints</span>
              </div>
              <div className="flex justify-between text-sm py-1 border-b" style={{ borderColor: 'hsl(var(--border-subtle))' }}>
                <span className="text-muted">Current Age / Gender:</span>
                <span className="font-semibold">{patient.visits[1].age} yrs / Female</span>
              </div>
              <div className="flex justify-between text-sm py-1 border-b" style={{ borderColor: 'hsl(var(--border-subtle))' }}>
                <span className="text-muted">Current Cognitive Status:</span>
                <span className="font-semibold" style={{ color: 'hsl(var(--amber))' }}>MMSE 26 / CDR 0.5 (Very Mild)</span>
              </div>
              <div className="flex justify-between text-sm py-1 border-b" style={{ borderColor: 'hsl(var(--border-subtle))' }}>
                <span className="text-muted">Brain Parenchyma (nWBV):</span>
                <span className="font-semibold">0.7410 (Volumetric atrophy observed)</span>
              </div>
              <div className="flex justify-between text-sm py-1">
                <span className="text-muted">APOE4 Carrier / p-tau181:</span>
                <span className="font-semibold" style={{ color: 'hsl(var(--rose))' }}>Positive / 24.8 pg/mL</span>
              </div>
            </div>
          </div>

          {/* Intervention Parameters */}
          <div className="card" style={{ padding: 24 }}>
            <h2 className="text-sm font-bold text-muted" style={{ textTransform: 'uppercase', marginBottom: 16, color: 'hsl(var(--cyan))' }}>
              🧪 Counterfactual Intervention Controls
            </h2>
            <div className="flex-col gap-4">
              <div>
                <label className="text-xs text-muted block mb-1">Therapeutic Protocol Name</label>
                <input
                  type="text"
                  value={interventionName}
                  onChange={(e) => setInterventionName(e.target.value)}
                  className="input w-full text-sm"
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">MMSE Cognitive Boost</span>
                    <span className="font-bold text-emerald">+{mmseBoost} pts</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="6"
                    step="0.5"
                    value={mmseBoost}
                    onChange={(e) => setMmseBoost(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">nWBV Brain Vol Protection</span>
                    <span className="font-bold text-cyan">+{(nwbvProtection * 100).toFixed(1)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.005"
                    max="0.05"
                    step="0.005"
                    value={nwbvProtection}
                    onChange={(e) => setNwbvProtection(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">p-tau181 Biofluid Reduction</span>
                    <span className="font-bold text-violet">{(ptauReduction * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.05"
                    max="0.50"
                    step="0.05"
                    value={ptauReduction}
                    onChange={(e) => setPtauReduction(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">Forward Projection Horizon</span>
                    <span className="font-bold">{yearsForward} Years</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="6"
                    step="1"
                    value={yearsForward}
                    onChange={(e) => setYearsForward(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Key Metrics Cards (When available) ── */}
        {results && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }} className="animate-in">
            <div className="card" style={{ padding: 20, borderLeft: '4px solid hsl(var(--emerald))' }}>
              <span className="text-xs text-muted block mb-1">Risk Reduction</span>
              <div className="text-2xl font-black text-emerald">
                {results.summary.projected_risk_reduction_pct}%
              </div>
              <span className="text-xs text-muted">Lower dementia progression risk</span>
            </div>
            <div className="card" style={{ padding: 20, borderLeft: '4px solid hsl(var(--cyan))' }}>
              <span className="text-xs text-muted block mb-1">Cognitive Points Salvaged</span>
              <div className="text-2xl font-black text-cyan">
                +{results.summary.mmse_salvaged_points} pts
              </div>
              <span className="text-xs text-muted">MMSE difference vs. natural curve</span>
            </div>
            <div className="card" style={{ padding: 20, borderLeft: '4px solid hsl(var(--violet))' }}>
              <span className="text-xs text-muted block mb-1">Brain Parenchyma Preserved</span>
              <div className="text-2xl font-black text-violet">
                +{(results.summary.nwbv_volume_protected * 100).toFixed(2)}%
              </div>
              <span className="text-xs text-muted">Intracranial volume preservation</span>
            </div>
            <div className="card" style={{ padding: 20, borderLeft: '4px solid hsl(var(--amber))' }}>
              <span className="text-xs text-muted block mb-1">Projected Outcome</span>
              <div className="text-lg font-bold" style={{ color: 'hsl(var(--amber))' }}>
                {results.trajectories.counterfactual_progression[results.trajectories.counterfactual_progression.length - 1].predicted_cdr_label.split('(')[0]}
              </div>
              <span className="text-xs text-muted">At Year {yearsForward} with intervention</span>
            </div>
          </div>
        )}

        {/* ── Tabs for Viewing Trajectories / Report ── */}
        <div className="card" style={{ padding: 24 }}>
          <div className="flex gap-4 border-b pb-3 mb-6" style={{ borderColor: 'hsl(var(--border-subtle))' }}>
            <button
              onClick={() => setActiveTab('trajectories')}
              className={`pb-2 px-3 text-sm font-semibold transition-colors ${
                activeTab === 'trajectories'
                  ? 'border-b-2 text-primary border-primary'
                  : 'text-muted hover:text-foreground'
              }`}
            >
              📈 Side-by-Side Trajectory Timeline
            </button>
            <button
              onClick={() => setActiveTab('report')}
              className={`pb-2 px-3 text-sm font-semibold transition-colors ${
                activeTab === 'report'
                  ? 'border-b-2 text-primary border-primary'
                  : 'text-muted hover:text-foreground'
              }`}
            >
              📋 Clinical Diagnostic Report (Markdown)
            </button>
          </div>

          {activeTab === 'trajectories' && results && (
            <div className="flex-col gap-6 animate-in">
              <h3 className="text-sm font-bold text-muted uppercase">
                Longitudinal Disease Modeling: Natural Decline vs. Therapeutic Stabilization
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr style={{ background: 'hsl(var(--surface-raised))', borderBottom: '1px solid hsl(var(--border-subtle))' }}>
                      <th style={{ padding: '10px 14px' }}>Timeline</th>
                      <th style={{ padding: '10px 14px' }}>Age</th>
                      <th style={{ padding: '10px 14px' }}>Natural MMSE</th>
                      <th style={{ padding: '10px 14px' }}>Intervention MMSE</th>
                      <th style={{ padding: '10px 14px' }}>Natural nWBV</th>
                      <th style={{ padding: '10px 14px' }}>Intervention nWBV</th>
                      <th style={{ padding: '10px 14px' }}>Natural Status</th>
                      <th style={{ padding: '10px 14px' }}>Intervention Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.trajectories.natural_progression.map((nat, idx) => {
                      const intv = results.trajectories.counterfactual_progression[idx];
                      return (
                        <tr
                          key={nat.year}
                          style={{
                            borderBottom: '1px solid hsl(var(--border-subtle))',
                            background: idx % 2 === 0 ? 'transparent' : 'hsl(var(--surface-raised) / 0.3)'
                          }}
                        >
                          <td style={{ padding: '10px 14px', fontWeight: 600 }}>
                            {nat.year === 0 ? 'Year 0 (Current)' : `Year +${nat.year}`}
                          </td>
                          <td style={{ padding: '10px 14px' }}>{nat.age} yrs</td>
                          <td style={{ padding: '10px 14px', color: 'hsl(var(--rose))' }}>{nat.mmse}</td>
                          <td style={{ padding: '10px 14px', color: 'hsl(var(--emerald))', fontWeight: 600 }}>{intv.mmse}</td>
                          <td style={{ padding: '10px 14px' }}>{nat.nwbv}</td>
                          <td style={{ padding: '10px 14px', color: 'hsl(var(--cyan))', fontWeight: 600 }}>{intv.nwbv}</td>
                          <td style={{ padding: '10px 14px' }}>
                            <span className="badge badge-subtle">{nat.predicted_cdr_label.split('(')[0]}</span>
                          </td>
                          <td style={{ padding: '10px 14px' }}>
                            <span className="badge badge-success">{intv.predicted_cdr_label.split('(')[0]}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="callout callout-info">
                <span>💡</span>
                <div>
                  <strong>Clinical Interpretation:</strong> {results.summary.clinical_significance}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'trajectories' && !results && loading && (
            <div className="text-center py-12 flex-col items-center gap-3">
              <div className="inline-block animate-spin" style={{ fontSize: '2rem' }}>⚙️</div>
              <p className="text-sm font-semibold text-cyan">Running Counterfactual GRU Trajectory Projection...</p>
              <p className="text-xs text-muted">Calculating multi-year ΔMMSE, brain parenchyma preservation, and CDR transition probabilities.</p>
            </div>
          )}

          {activeTab === 'trajectories' && !results && !loading && (
            <div className="text-center py-12 text-muted">
              Click <strong>"▶ Run Simulation"</strong> above to calculate counterfactual trajectory projections.
            </div>
          )}

          {activeTab === 'report' && (
            <div className="animate-in flex-col gap-4">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-muted">Exportable Clinical Markdown Documentation</span>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(reportText || '');
                    alert('Clinical Report copied to clipboard!');
                  }}
                  className="btn btn-sm btn-ghost text-xs"
                >
                  📋 Copy to Clipboard
                </button>
              </div>
              <pre
                style={{
                  background: 'hsl(var(--surface-raised))',
                  padding: 20,
                  borderRadius: 8,
                  fontSize: '0.8rem',
                  lineHeight: 1.5,
                  overflowX: 'auto',
                  border: '1px solid hsl(var(--border-subtle))',
                  whiteSpace: 'pre-wrap'
                }}
              >
                {reportText || 'Click "Generate Full Report" above to compile the full multi-modal clinical diagnostic report.'}
              </pre>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
