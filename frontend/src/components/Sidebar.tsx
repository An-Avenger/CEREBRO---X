'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV_ITEMS = [
  {
    section: 'Research',
    items: [
      { href: '/',              icon: '⬡',  label: 'Overview',      badge: null },
      { href: '/predict',       icon: '⚡', label: 'Predict CDR',   badge: null },
      { href: '/brain-twin',    icon: '🧠', label: 'Brain Twin',    badge: null },
      { href: '/explainability',icon: '🔍', label: 'Explainability',badge: null },
    ],
  },
  {
    section: 'Data',
    items: [
      { href: '/experiments',   icon: '⚗',  label: 'Experiments',   badge: '10' },
      { href: '/history',       icon: '📋',  label: 'Prediction Log',badge: null },
    ],
  },
  {
    section: 'Presentation',
    items: [
      { href: '/demo',          icon: '▶',  label: 'Demo Mode',     badge: 'NEW' },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">🧠</div>
        <div className="sidebar-logo-text">
          <span className="sidebar-logo-name">Cerebro-X</span>
          <span className="sidebar-logo-sub">Digital Brain Twin</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((group) => (
          <div key={group.section}>
            <p className="sidebar-section-label">{group.section}</p>
            {group.items.map((item) => {
              const isActive = item.href === '/'
                ? pathname === '/'
                : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`sidebar-link${isActive ? ' active' : ''}`}
                >
                  <span className="sidebar-link-icon">{item.icon}</span>
                  <span>{item.label}</span>
                  {item.badge && (
                    <span className="sidebar-badge">{item.badge}</span>
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-status">
          <span className="status-dot" />
          <span>API: localhost:8000</span>
        </div>
        <div className="sidebar-status" style={{ marginTop: '6px' }}>
          <span style={{ width: 7 }} />
          <span style={{ fontSize: '0.6875rem', color: 'hsl(var(--text-side-2))' }}>
            OASIS-2 · 150 subjects
          </span>
        </div>
      </div>
    </aside>
  );
}
