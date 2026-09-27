import type { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Overview — Cerebro-X',
  description: 'Cerebro-X research dashboard — model performance, phase status, and key findings.',
};

const METRICS = [
  { label: 'Best Accuracy',     value: '75.0%', sub: 'Temporal GRU (test set)',    trend: '+3.6%', up: true  },
  { label: 'Balanced Accuracy', value: '55.7%', sub: 'vs 53.7% Last-Visit Baseline', trend: '+2.0%', up: true  },
  { label: 'F1 Macro',          value: '55.1%', sub: 'Temporal GRU (4-class CDR)', trend: '+1.1%', up: true  },
  { label: 'Total Subjects',    value: '150',   sub: 'OASIS-2 Longitudinal',        trend: null,    up: false },
];

const MODEL_TABLE = [
  { model: 'Dummy (Majority)',         phase: 'Phase 3', acc: '39.3%', bacc: '25.0%', f1: '14.1%', hw: 'CPU',          best: false },
  { model: 'Random Forest',            phase: 'Phase 3', acc: '64.3%', bacc: '45.7%', f1: '42.6%', hw: 'CPU',          best: false },
  { model: 'Logistic Regression',      phase: 'Phase 3', acc: '67.9%', bacc: '72.0%', f1: '64.0%', hw: 'CPU',          best: false },
  { model: 'Last-Visit Baseline (GRU)',phase: 'Phase 5', acc: '71.4%', bacc: '53.7%', f1: '54.0%', hw: 'Kaggle T4',    best: false },
  { model: 'Temporal GRU (Cerebro-X)',phase: 'Phase 5', acc: '75.0%', bacc: '55.7%', f1: '55.1%', hw: 'Kaggle T4',    best: true  },
];

const PHASES = [
  { id: 1,  label: 'Project Setup & Architecture',      done: true  },
  { id: 2,  label: 'Dataset Curation (OASIS-2)',         done: true  },
  { id: 3,  label: 'Traditional ML Baselines',           done: true  },
  { id: 4,  label: 'Multimodal Scaffold (MRI+Clinical)', done: true  },
  { id: 5,  label: 'Longitudinal Temporal GRU',          done: true  },
  { id: 6,  label: 'Explainability (SHAP + Grad-CAM)',   done: true  },
  { id: 7,  label: 'Final Review & Delivery',            done: true  },
  { id: 8,  label: 'Research API (FastAPI)',              done: true  },
  { id: 9,  label: 'Frontend Dashboard',                 done: true  },
  { id: 10, label: 'Database & Experiment Management',   done: true  },
  { id: 11, label: 'Validation & Robustness (39 tests)', done: true  },
  { id: 12, label: 'Thesis / Paper Preparation',         done: true  },
  { id: 13, label: 'Deployment (Docker)',                done: true  },
  { id: 14, label: 'Final Research Audit',               done: true  },
];

const FINDINGS = [
  {
    icon: '📈',
    title: 'Temporal context matters',
    text: 'The Temporal GRU (75.0%) outperformed the Last-Visit Baseline (71.4%), demonstrating that modelling how cognition changes over time adds measurable predictive power.',
  },
  {
    icon: '🔬',
    title: 'Traditional ML is a strong baseline',
    text: 'Logistic Regression achieved 72.0% balanced accuracy — competitive for a 4-class CDR problem with only 150 patients. It remains our Phase 3 reference benchmark.',
  },
  {
    icon: '⚠️',
    title: 'Random Forest overfits',
    text: 'RF achieved 100% train accuracy but only 45.7% balanced test accuracy — confirming that with small medical datasets, simpler regularised models generalise better.',
  },
  {
    icon: '🧬',
    title: 'SHAP reveals clinical signals',
    text: 'nWBV (Normalised Whole Brain Volume), MMSE, and Age are dominant predictors — fully consistent with established neurological literature on Alzheimer\'s progression.',
  },
];

export default function OverviewPage() {
  const doneCount = PHASES.filter(p => p.done).length;

  return (
    <>
      <div className="page-header">
        <div className="page-header-left">
          <h1 className="page-title">Research Overview</h1>
          <p className="page-subtitle">
            Longitudinal Digital Brain Twin · OASIS-2 · CDR 4-class Prediction
          </p>
        </div>
        <div className="flex gap-2" style={{ alignItems: 'center', flexShrink: 0 }}>
          <span className="badge badge-emerald">✓ {doneCount}/{PHASES.length} Phases Complete</span>
          <Link href="/predict" className="btn btn-accent">
            ⚡ Run Prediction
          </Link>
        </div>
      </div>

      <div className="page-body flex-col gap-8">

        {/* ── Disclaimer ── */}
        <div className="callout callout-warning animate-in" role="alert">
          <span>⚠️</span>
          <div>
            <strong>Research Prototype Only.</strong> All models are trained on OASIS-2 (150 subjects).
            Results are NOT validated for clinical use and must not be interpreted as medical diagnoses.
          </div>
        </div>

        {/* ── Stat Cards ── */}
        <section aria-labelledby="metrics-heading" className="animate-in">
          <h2 id="metrics-heading" className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Key Metrics
          </h2>
          <div className="grid-4">
            {METRICS.map((m, i) => {
              const pastelClass = ['card-pastel-lavender', 'card-pastel-sage', 'card-pastel-peach', 'card-pastel-blue'][i];
              return (
                <div key={m.label} className={`stat-card ${pastelClass} animate-in animate-delay-${i + 1}`}>
                  <p className="stat-label" style={{ color: 'hsl(var(--text-1))', opacity: 0.7 }}>{m.label}</p>
                  <p className="stat-value">{m.value}</p>
                  <p className="stat-sub" style={{ color: 'hsl(var(--text-1))', opacity: 0.6 }}>{m.sub}</p>
                  {m.trend && (
                    <span className={`stat-trend ${m.up ? 'up' : 'down'}`} style={{ marginTop: 8 }}>
                      {m.up ? '↑' : '↓'} {m.trend} vs baseline
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* ── Main grid: model table + phase tracker ── */}
        <div className="grid-2" style={{ gridTemplateColumns: '1fr 320px' }}>

          {/* Model Comparison Table */}
          <section className="animate-in" aria-labelledby="models-heading">
            <div className="flex items-center justify-between" style={{ marginBottom: '12px' }}>
              <h2 id="models-heading" className="text-sm font-bold text-muted" style={{ textTransform: 'uppercase', letterSpacing: '0.07em' }}>
                Model Comparison
              </h2>
              <Link href="/experiments" className="btn btn-ghost btn-sm">View Experiments →</Link>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Acc.</th>
                    <th>Bal. Acc.</th>
                    <th>F1</th>
                    <th>HW</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_TABLE.map((m) => (
                    <tr key={m.model} style={m.best ? { background: 'hsl(var(--accent) / 0.05)' } : {}}>
                      <td>
                        <div className="flex items-center gap-2">
                          {m.best && <span className="badge badge-accent">⭐ Best</span>}
                          <span style={m.best ? { color: 'hsl(var(--accent))', fontWeight: 600 } : {}}>
                            {m.model}
                          </span>
                        </div>
                      </td>
                      <td className="td-highlight">{m.acc}</td>
                      <td className="td-highlight">{m.bacc}</td>
                      <td>{m.f1}</td>
                      <td>
                        <span className="badge badge-neutral">
                          {m.hw}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Phase Tracker */}
          <section className="animate-in" aria-labelledby="phases-heading">
            <h2 id="phases-heading" className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Phase Tracker
            </h2>
            <div className="card" style={{ padding: '16px', maxHeight: '380px', overflowY: 'auto' }}>
              <div className="flex-col gap-2">
                {PHASES.map((phase) => (
                  <div key={phase.id} className="flex items-center gap-3" style={{ fontSize: '0.8125rem' }}>
                    <span style={{
                      width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
                      display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem',
                      background: phase.done ? 'hsl(var(--emerald) / 0.15)' : 'hsl(var(--bg-3))',
                      border: `1px solid ${phase.done ? 'hsl(var(--emerald) / 0.4)' : 'hsl(var(--border))'}`,
                      color: phase.done ? 'hsl(var(--emerald))' : 'hsl(var(--text-3))',
                    }}>
                      {phase.done ? '✓' : phase.id}
                    </span>
                    <span style={{ color: phase.done ? 'hsl(var(--text-1))' : 'hsl(var(--text-3))' }}>
                      {phase.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>

        </div>

        {/* ── Key Findings ── */}
        <section className="animate-in" aria-labelledby="findings-heading">
          <h2 id="findings-heading" className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Key Findings
          </h2>
          <div className="grid-2">
            {FINDINGS.map((f, i) => (
              <div key={f.title} className={`card animate-in animate-delay-${i % 4 + 1}`}>
                <div className="flex items-center gap-3" style={{ marginBottom: '10px' }}>
                  <span style={{ fontSize: '1.5rem' }}>{f.icon}</span>
                  <h3 style={{ fontSize: '0.9375rem', fontWeight: 600 }}>{f.title}</h3>
                </div>
                <p style={{ fontSize: '0.875rem', color: 'hsl(var(--text-2))', lineHeight: 1.65 }}>{f.text}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Quick actions ── */}
        <section className="animate-in" aria-labelledby="actions-heading">
          <h2 id="actions-heading" className="text-sm font-bold text-muted" style={{ marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Quick Actions
          </h2>
          <div className="grid-3">
            {[
              { href: '/predict',        icon: '⚡', title: 'Run a Prediction',      sub: 'Enter patient visits and get CDR forecast' },
              { href: '/explainability', icon: '🔍', title: 'View Explainability',   sub: 'SHAP features & Grad-CAM spatial attention' },
              { href: '/experiments',    icon: '⚗',  title: 'Browse Experiments',    sub: 'All 10 experiment results and metrics' },
            ].map((a) => (
              <Link key={a.href} href={a.href} className="card" style={{ textDecoration: 'none', cursor: 'pointer' }}>
                <span style={{ fontSize: '1.75rem', display: 'block', marginBottom: '10px' }}>{a.icon}</span>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: '5px', color: 'hsl(var(--text-1))' }}>{a.title}</h3>
                <p style={{ fontSize: '0.8125rem', color: 'hsl(var(--text-3))' }}>{a.sub}</p>
              </Link>
            ))}
          </div>
        </section>

      </div>
    </>
  );
}
