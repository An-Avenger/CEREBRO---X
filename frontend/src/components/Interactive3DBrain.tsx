'use client';

import React, { useEffect, useRef, useState } from 'react';

interface Particle {
  x: number;
  y: number;
  z: number;
  baseX: number;
  baseY: number;
  baseZ: number;
  color: string;
  size: number;
  region: 'frontal' | 'temporal' | 'parietal' | 'occipital' | 'hippocampus';
  activity: number;
}

export default function Interactive3DBrain({
  atrophyLevel = 0.35,
  activeRegion = 'temporal',
  onRegionSelect,
}: {
  atrophyLevel?: number;
  activeRegion?: string;
  onRegionSelect?: (region: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rotation, setRotation] = useState({ x: 0.15, y: 0.4 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1.85); // Increased default zoom for clear, centered brain rendering
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);

  const particlesRef = useRef<Particle[]>([]);

  // Initialize Point Cloud for Anatomical Brain Twin
  useEffect(() => {
    const pts: Particle[] = [];
    const count = 1350;

    for (let i = 0; i < count; i++) {
      const hemi = Math.random() > 0.5 ? 1 : -1;
      const u = Math.random() * Math.PI * 2;
      const v = (Math.random() - 0.5) * Math.PI;

      // Realistic anatomical brain ratios
      const a = 90; // Length
      const b = 68; // Height
      const c = 64; // Width

      const noise = (Math.sin(u * 5) * Math.cos(v * 4) + Math.sin(u * 8) * 0.4) * 8;
      const r = 1.0 + (Math.random() * 0.16 - 0.08);

      let x = (c * Math.cos(v) * Math.sin(u) * 0.92 + hemi * 14 + noise * 0.5) * r;
      let y = (b * Math.sin(v) + noise * 0.6) * r;
      let z = (a * Math.cos(v) * Math.cos(u) + noise * 0.8) * r;

      // Anatomical lobe localization
      let reg: Particle['region'] = 'parietal';
      let col = '#38bdf8'; // Electric Cyan

      if (z > 25) {
        reg = 'frontal';
        col = '#a855f7'; // Purple/Violet
      } else if (z < -30) {
        reg = 'occipital';
        col = '#10b981'; // Emerald
      } else if (y < -15 && Math.abs(x) > 28) {
        reg = 'temporal';
        col = '#f97316'; // Coral / Amber
      } else if (Math.abs(x) < 22 && y < -10 && z > -15 && z < 15) {
        reg = 'hippocampus';
        col = '#ef4444'; // Red-Rose
      }

      pts.push({
        x, y, z,
        baseX: x, baseY: y, baseZ: z,
        color: col,
        size: Math.random() * 1.8 + 1.2,
        region: reg,
        activity: Math.random(),
      });
    }

    particlesRef.current = pts;
  }, []);

  // Animation & Rendering Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let t = 0;

    const render = () => {
      t += 0.02;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const cx = canvas.width / 2;
      const cy = canvas.height / 2;

      let curRotY = rotation.y;
      if (autoRotate && !isDragging) {
        curRotY += 0.007;
        setRotation(r => ({ ...r, y: r.y + 0.007 }));
      }

      const cosX = Math.cos(rotation.x);
      const sinX = Math.sin(rotation.x);
      const cosY = Math.cos(curRotY);
      const sinY = Math.sin(curRotY);

      // Depth sort particles
      const transformed = particlesRef.current.map(p => {
        let shrink = 1.0;
        if (p.region === 'temporal' || p.region === 'hippocampus') {
          shrink = 1.0 - (atrophyLevel * 0.28);
        }

        let px = p.baseX * shrink;
        let py = p.baseY * shrink;
        let pz = p.baseZ * shrink;

        const pulse = Math.sin(t * 3 + p.activity * 10) * 1.4;
        px += pulse * 0.3;
        py += pulse * 0.3;

        // 3D Rotation Matrix
        let x1 = px * cosY - pz * sinY;
        let z1 = pz * cosY + px * sinY;

        let y2 = py * cosX - z1 * sinX;
        let z2 = z1 * cosX + py * sinX;

        // Perspective Projection
        const fov = 380;
        const scale = (fov / (fov + z2 + 160)) * zoom;
        const projX = cx + x1 * scale;
        const projY = cy + y2 * scale;

        return {
          ...p,
          projX,
          projY,
          depth: z2,
          scale,
        };
      });

      transformed.sort((a, b) => b.depth - a.depth);

      // Draw Synapses
      ctx.lineWidth = 0.6;
      for (let i = 0; i < transformed.length; i += 7) {
        const p1 = transformed[i];
        if (p1.depth < -60) continue;

        for (let j = i + 1; j < Math.min(i + 7, transformed.length); j++) {
          const p2 = transformed[j];
          const dist = Math.hypot(p1.projX - p2.projX, p1.projY - p2.projY);
          if (dist < 28) {
            const alpha = (1 - dist / 28) * 0.22 * Math.min(p1.scale, 1.2);
            ctx.strokeStyle = p1.region === activeRegion ? `rgba(249, 115, 22, ${alpha * 2})` : `rgba(56, 189, 248, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(p1.projX, p1.projY);
            ctx.lineTo(p2.projX, p2.projY);
            ctx.stroke();
          }
        }
      }

      // Draw Particles
      transformed.forEach(p => {
        const isHovered = hoveredRegion === p.region;
        const isActive = activeRegion === p.region;

        const size = Math.max(0.9, p.size * p.scale * (isActive ? 1.5 : 1));
        const alpha = Math.max(0.25, Math.min(0.95, (p.depth + 120) / 240));

        ctx.beginPath();
        ctx.arc(p.projX, p.projY, size, 0, Math.PI * 2);

        if (isActive) {
          ctx.fillStyle = '#f97316';
          ctx.shadowColor = '#f97316';
          ctx.shadowBlur = 10;
        } else if (isHovered) {
          ctx.fillStyle = '#38bdf8';
          ctx.shadowColor = '#38bdf8';
          ctx.shadowBlur = 8;
        } else {
          ctx.fillStyle = p.color;
          ctx.shadowBlur = 0;
        }

        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.shadowBlur = 0;
      });

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [rotation, zoom, activeRegion, hoveredRegion, autoRotate, isDragging, atrophyLevel]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setAutoRotate(false);
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setRotation(r => ({
      x: Math.max(-1.2, Math.min(1.2, r.x + dy * 0.008)),
      y: r.y + dx * 0.008,
    }));
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="flex flex-col select-none w-full gap-3">
      {/* 3D Holographic Canvas Box with clean border */}
      <div
        className="relative overflow-hidden rounded-xl border border-[hsl(var(--border))]"
        style={{
          background: 'radial-gradient(circle at 50% 50%, #111827 0%, #060910 100%)',
          width: '100%',
          height: '380px',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={(e) => {
          e.preventDefault();
          setZoom(z => Math.max(1.0, Math.min(2.8, z - e.deltaY * 0.0015)));
        }}
      >
        <canvas
          ref={canvasRef}
          width={500}
          height={380}
          style={{ width: '100%', height: '100%', cursor: isDragging ? 'grabbing' : 'grab' }}
        />

        {/* Top-Right Orbit Controls */}
        <div className="absolute top-3 right-3 flex gap-2 z-10">
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`btn btn-xs ${autoRotate ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '3px 9px', fontSize: '0.7rem' }}
          >
            {autoRotate ? '⏸ Auto' : '▶ Orbit'}
          </button>
          <button
            onClick={() => setZoom(1.85)}
            className="btn btn-xs btn-secondary"
            style={{ padding: '3px 9px', fontSize: '0.7rem' }}
          >
            Reset
          </button>
        </div>
      </div>

      {/* Clean Bottom Toolbar - completely placed OUTSIDE the canvas so it never overlaps */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-lg bg-[hsl(var(--bg-2))] border border-[hsl(var(--border-subtle))]">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-bold text-muted mr-1">LOBES:</span>
          {[
            { id: 'temporal', label: 'Temporal / Hippo', color: '#f97316' },
            { id: 'frontal', label: 'Frontal', color: '#a855f7' },
            { id: 'parietal', label: 'Parietal', color: '#38bdf8' },
          ].map(r => (
            <button
              key={r.id}
              onClick={() => onRegionSelect?.(r.id)}
              className={`btn btn-xs ${activeRegion === r.id ? 'btn-primary' : 'btn-ghost'}`}
              style={{ padding: '3px 8px', fontSize: '0.72rem' }}
            >
              <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ background: r.color }} />
              {r.label}
            </button>
          ))}
        </div>

        <span className="text-[11px] text-muted font-mono">
          Drag to Orbit · Scroll to Zoom
        </span>
      </div>
    </div>
  );
}
