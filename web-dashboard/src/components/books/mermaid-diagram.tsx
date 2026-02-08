"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Move,
  RotateCcw,
} from "lucide-react";

interface MermaidDiagramProps {
  chart: string;
}

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3;
const ZOOM_STEP = 0.2;

export function MermaidDiagram({ chart }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string>("");

  // Transform state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          themeVariables: {
            primaryColor: "#e8f0fe",
            primaryTextColor: "#1d1d1f",
            primaryBorderColor: "#007AFF",
            lineColor: "#86868b",
            secondaryColor: "#f0f7ee",
            tertiaryColor: "#fef7e0",
            fontFamily: "IBM Plex Sans, system-ui, sans-serif",
            fontSize: "13px",
            nodeBorder: "#007AFF",
            mainBkg: "#e8f0fe",
            clusterBkg: "#f5f5f7",
            clusterBorder: "#d1d1d6",
            titleColor: "#1d1d1f",
            edgeLabelBackground: "#ffffff",
          },
          flowchart: {
            htmlLabels: true,
            curve: "basis",
            padding: 12,
          },
        });

        const id = `mermaid-${Math.random().toString(36).slice(2, 9)}`;
        const { svg: renderedSvg } = await mermaid.render(id, chart.trim());
        if (!cancelled) {
          setSvg(renderedSvg);
          setError("");
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Diagram error");
        }
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [chart]);

  // Reset when chart changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [chart]);

  const handleZoomIn = useCallback(() => {
    setZoom((z) => Math.min(MAX_ZOOM, z + ZOOM_STEP));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom((z) => Math.max(MIN_ZOOM, z - ZOOM_STEP));
  }, []);

  const handleReset = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const handleFitToView = useCallback(() => {
    if (!containerRef.current || !viewportRef.current) return;
    const viewport = viewportRef.current.getBoundingClientRect();
    const svgEl = containerRef.current.querySelector("svg");
    if (!svgEl) return;

    const svgW = svgEl.getBoundingClientRect().width / zoom;
    const svgH = svgEl.getBoundingClientRect().height / zoom;
    if (svgW === 0 || svgH === 0) return;

    const fitZoom = Math.min(
      (viewport.width - 32) / svgW,
      (viewport.height - 32) / svgH,
      MAX_ZOOM
    );
    setZoom(Math.max(MIN_ZOOM, fitZoom));
    setPan({ x: 0, y: 0 });
  }, [zoom]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
    setZoom((z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z + delta)));
  }, []);

  // Drag to pan
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      setIsDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    },
    [pan]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      setPan({ x: dragStart.current.panX + dx, y: dragStart.current.panY + dy });
    },
    [isDragging]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Touch support for mobile
  const touchStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (e.touches.length === 1) {
        const t = e.touches[0];
        setIsDragging(true);
        touchStart.current = { x: t.clientX, y: t.clientY, panX: pan.x, panY: pan.y };
      }
    },
    [pan]
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      if (!isDragging || e.touches.length !== 1) return;
      const t = e.touches[0];
      const dx = t.clientX - touchStart.current.x;
      const dy = t.clientY - touchStart.current.y;
      setPan({ x: touchStart.current.panX + dx, y: touchStart.current.panY + dy });
    },
    [isDragging]
  );

  const handleTouchEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  if (error) {
    return (
      <div className="my-4 p-4 rounded-xl bg-red-50 border border-red-200 text-sm text-red-700">
        <p className="font-medium mb-1">Diagram Error</p>
        <pre className="text-xs whitespace-pre-wrap">{error}</pre>
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-red-500">Source</summary>
          <pre className="mt-1 text-xs whitespace-pre-wrap text-red-600">{chart}</pre>
        </details>
      </div>
    );
  }

  if (!svg) {
    return (
      <div className="my-4 flex items-center justify-center p-8 rounded-xl bg-slate-50 border border-slate-200">
        <div className="animate-pulse text-sm text-slate-400">Rendering diagram...</div>
      </div>
    );
  }

  const zoomPercent = Math.round(zoom * 100);

  return (
    <div className="my-4 rounded-xl bg-white/60 border border-slate-200 overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-slate-50/80 border-b border-slate-200/60">
        <div className="flex items-center gap-1">
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded-md hover:bg-slate-200/60 text-slate-500 hover:text-slate-700 transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="h-3.5 w-3.5" />
          </button>

          <span className="text-[11px] font-mono text-slate-400 w-10 text-center select-none">
            {zoomPercent}%
          </span>

          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded-md hover:bg-slate-200/60 text-slate-500 hover:text-slate-700 transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="h-3.5 w-3.5" />
          </button>

          <div className="w-px h-4 bg-slate-200 mx-1" />

          <button
            onClick={handleFitToView}
            className="p-1.5 rounded-md hover:bg-slate-200/60 text-slate-500 hover:text-slate-700 transition-colors"
            title="Fit to view"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>

          <button
            onClick={handleReset}
            className="p-1.5 rounded-md hover:bg-slate-200/60 text-slate-500 hover:text-slate-700 transition-colors"
            title="Reset"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <Move className="h-3 w-3" />
          <span>Drag untuk geser, scroll untuk zoom</span>
        </div>
      </div>

      {/* Viewport */}
      <div
        ref={viewportRef}
        className="relative overflow-hidden"
        style={{ height: "clamp(200px, 50vh, 600px)", cursor: isDragging ? "grabbing" : "grab" }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <div
          ref={containerRef}
          className="flex justify-center items-start min-w-full min-h-full p-4 select-none"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "center top",
            transition: isDragging ? "none" : "transform 0.15s ease-out",
          }}
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      </div>
    </div>
  );
}
