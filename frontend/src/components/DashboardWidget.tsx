import React from 'react';

interface DashboardWidgetProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

export function DashboardWidget({ title, children, className = '' }: DashboardWidgetProps) {
  return (
    <div className={`glass-panel flex-col gap-4 animate-fade-in ${className}`}>
      <h2 style={{ fontSize: '1.25rem', color: 'hsl(var(--accent-cyan))' }}>{title}</h2>
      <div className="widget-content flex-col" style={{ flex: 1, minHeight: 0 }}>
        {children}
      </div>
    </div>
  );
}
