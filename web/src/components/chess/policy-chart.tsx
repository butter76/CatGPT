"use client";

import type { PolicyEntry, WDL } from "@/lib/types";
import { uciToAlgebraic } from "@/lib/chess-utils";
import { usePositionStore } from "@/lib/store";

// ─── Policy Bar Chart ─────────────────────────────────────────────

interface PolicyChartProps {
  policy: PolicyEntry[];
  fen: string;
  maxEntries?: number;
}

export function PolicyChart({ policy, fen, maxEntries = 8 }: PolicyChartProps) {
  const { notationFormat } = usePositionStore();
  const sorted = [...policy]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, maxEntries);

  const maxProb = Math.max(...sorted.map((e) => e.probability), 0.01);

  return (
    <div className="space-y-1.5">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Policy Distribution
      </h4>
      {sorted.map((entry) => {
        const label =
          notationFormat === "algebraic"
            ? uciToAlgebraic(fen, entry.move)
            : entry.move;
        const pct = (entry.probability * 100).toFixed(1);
        const barWidth = (entry.probability / maxProb) * 100;

        return (
          <div key={entry.move} className="flex items-center gap-2">
            <span className="w-12 text-right font-mono text-sm font-medium">
              {label}
            </span>
            <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded transition-all duration-300"
                style={{ width: `${barWidth}%` }}
              />
            </div>
            <span className="w-14 text-right text-xs text-muted-foreground font-mono">
              {pct}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── WDL Bar ──────────────────────────────────────────────────────

interface WDLBarProps {
  wdl: WDL;
}

export function WDLBar({ wdl }: WDLBarProps) {
  const winPct = (wdl.win * 100).toFixed(1);
  const drawPct = (wdl.draw * 100).toFixed(1);
  const lossPct = (wdl.loss * 100).toFixed(1);

  return (
    <div className="space-y-1.5">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Win / Draw / Loss
      </h4>
      <div className="flex h-7 rounded overflow-hidden text-xs font-medium">
        {wdl.win > 0.01 && (
          <div
            className="bg-green-500 flex items-center justify-center text-white transition-all duration-300"
            style={{ width: `${wdl.win * 100}%` }}
            title={`Win: ${winPct}%`}
          >
            {wdl.win > 0.08 && `${winPct}%`}
          </div>
        )}
        {wdl.draw > 0.01 && (
          <div
            className="bg-gray-400 flex items-center justify-center text-white transition-all duration-300"
            style={{ width: `${wdl.draw * 100}%` }}
            title={`Draw: ${drawPct}%`}
          >
            {wdl.draw > 0.08 && `${drawPct}%`}
          </div>
        )}
        {wdl.loss > 0.01 && (
          <div
            className="bg-red-500 flex items-center justify-center text-white transition-all duration-300"
            style={{ width: `${wdl.loss * 100}%` }}
            title={`Loss: ${lossPct}%`}
          >
            {wdl.loss > 0.08 && `${lossPct}%`}
          </div>
        )}
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span className="text-green-600">W {winPct}%</span>
        <span>D {drawPct}%</span>
        <span className="text-red-600">L {lossPct}%</span>
      </div>
    </div>
  );
}

// ─── Q Value Display ──────────────────────────────────────────────

interface QValueProps {
  q: number;
  nodes: number;
}

export function QValueDisplay({ q, nodes }: QValueProps) {
  // Map Q from [-1, 1] to a color gradient
  const normalizedQ = (q + 1) / 2; // 0 to 1
  const isPositive = q >= 0;

  return (
    <div className="space-y-1.5">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Evaluation
      </h4>
      <div className="flex items-baseline gap-3">
        <span
          className={`text-3xl font-bold font-mono ${
            isPositive ? "text-green-600" : "text-red-600"
          }`}
        >
          {q >= 0 ? "+" : ""}
          {q.toFixed(3)}
        </span>
        <span className="text-sm text-muted-foreground">
          Q @ {nodes} node{nodes !== 1 ? "s" : ""}
        </span>
      </div>
      {/* Eval bar */}
      <div className="h-2 bg-red-400 rounded overflow-hidden">
        <div
          className="h-full bg-green-500 transition-all duration-500"
          style={{ width: `${normalizedQ * 100}%` }}
        />
      </div>
    </div>
  );
}
