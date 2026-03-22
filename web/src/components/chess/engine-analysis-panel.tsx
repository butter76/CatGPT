"use client";

import { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { useEngineAnalysis } from "@/lib/use-engine-analysis";
import { usePositionStore } from "@/lib/store";
import { uciToAlgebraic } from "@/lib/chess-utils";
import type { EngineInfoLine, EngineKind, CatGPTSearchStats } from "@/lib/types";
import {
  Play,
  Square,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Save,
} from "lucide-react";

interface EngineAnalysisPanelProps {
  fen: string;
  /** If provided, the final analysis will be auto-saved to this position */
  positionId?: string;
  /** Callback when analysis is saved */
  onSaved?: () => void;
}

export function EngineAnalysisPanel({
  fen,
  positionId,
  onSaved,
}: EngineAnalysisPanelProps) {
  const {
    running,
    engine,
    depthHistory,
    latestInfo,
    bestMove,
    saved,
    error,
    catgptStats,
    catgptHistory,
    startAnalysis,
    stopAnalysis,
    reset,
  } = useEngineAnalysis();

  const [selectedEngine, setSelectedEngine] = useState<EngineKind>("catgpt");
  const [nodes, setNodes] = useState(400);
  const [availableEngines, setAvailableEngines] = useState<string[]>([]);

  const isCatGPTEngine = (eng: EngineKind) => eng === "catgpt" || eng === "catgpt_mcts";

  const handleEngineChange = (eng: EngineKind) => {
    setSelectedEngine(eng);
    setNodes(eng === "stockfish" ? 500000 : isCatGPTEngine(eng) ? 400 : 5000);
  };

  // Fetch available engines
  useEffect(() => {
    fetch("/api/analyze/engines")
      .then((r) => r.json())
      .then((data) => setAvailableEngines(data.engines || []))
      .catch(() => setAvailableEngines([]));
  }, []);

  const handleStart = () => {
    startAnalysis({
      fen,
      engine: selectedEngine,
      nodes,
      positionId,
    });
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center justify-between">
          <span>🔧 Engine Analysis</span>
          {running && (
            <Badge variant="outline" className="border-blue-500 text-blue-600 animate-pulse">
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              Running
            </Badge>
          )}
          {!running && bestMove && (
            <Badge variant="outline" className="border-green-500 text-green-600">
              <CheckCircle2 className="w-3 h-3 mr-1" />
              Complete
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Controls */}
        {!running && (
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label className="text-xs">Engine</Label>
                <Select
                  value={selectedEngine}
                  onValueChange={(v) => handleEngineChange(v as EngineKind)}
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem
                      value="catgpt"
                      disabled={!availableEngines.includes("catgpt")}
                    >
                      🐱 CatGPT (Fractional)
                    </SelectItem>
                    <SelectItem
                      value="catgpt_mcts"
                      disabled={!availableEngines.includes("catgpt_mcts")}
                    >
                      🐱 CatGPT (MCTS)
                    </SelectItem>
                    <SelectItem
                      value="stockfish"
                      disabled={!availableEngines.includes("stockfish")}
                    >
                      Stockfish 18
                    </SelectItem>
                    <SelectItem
                      value="leela"
                      disabled={!availableEngines.includes("leela")}
                    >
                      Leela Chess Zero
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label className="text-xs">
                  {isCatGPTEngine(selectedEngine) ? "GPU Evals" : "Nodes"}
                </Label>
                <Input
                  type="number"
                  value={nodes}
                  onChange={(e) => setNodes(parseInt(e.target.value) || 1)}
                  className="h-8 text-xs font-mono"
                  min={isCatGPTEngine(selectedEngine) ? 10 : 100}
                  max={isCatGPTEngine(selectedEngine) ? 10000 : 100000000}
                  step={isCatGPTEngine(selectedEngine) ? 100 : 100000}
                />
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={handleStart} className="flex-1" size="sm">
                <Play className="w-4 h-4 mr-1" />
                Analyze
              </Button>
              {(bestMove || catgptStats) && (
                <Button onClick={reset} variant="outline" size="sm">
                  Reset
                </Button>
              )}
            </div>
          </div>
        )}

        {running && (
          <Button onClick={stopAnalysis} variant="destructive" size="sm" className="w-full">
            <Square className="w-4 h-4 mr-1" />
            Stop
          </Button>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 text-sm text-red-500">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* ── CatGPT Analysis Display ── */}
        {catgptStats && (engine === "catgpt" || engine === "catgpt_mcts") && (
          <>
            <Separator />
            <CatGPTStatsDisplay
              stats={catgptStats}
              history={catgptHistory}
              bestMove={bestMove}
              fen={fen}
            />
          </>
        )}

        {/* ── UCI Engine Display (Stockfish / Leela) ── */}
        {latestInfo && engine !== "catgpt" && engine !== "catgpt_mcts" && (
          <>
            <Separator />
            <UCIStatsDisplay
              latestInfo={latestInfo}
              depthHistory={depthHistory}
              bestMove={bestMove}
              fen={fen}
            />
          </>
        )}

        {/* Saved indicator */}
        {saved && (
          <div className="flex items-center gap-1.5 text-xs text-green-600">
            <Save className="w-3 h-3" />
            Analysis saved to database
          </div>
        )}

        {/* Empty state */}
        {!running && !latestInfo && !catgptStats && !error && (
          <p className="text-xs text-muted-foreground text-center py-2">
            {availableEngines.length === 0
              ? "No engines available on this server"
              : "Select an engine and click Analyze to start"}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// ─── CatGPT Stats Display ─────────────────────────────────────────

function CatGPTStatsDisplay({
  stats,
  history,
  bestMove,
  fen,
}: {
  stats: CatGPTSearchStats;
  history: CatGPTSearchStats[];
  bestMove: string | null;
  fen: string;
}) {
  const { notationFormat } = usePositionStore();
  // null = show latest (live); number = pinned to a history entry
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  // The snapshot we're displaying: either the pinned one or the latest
  const displayStats = selectedIdx !== null ? history[selectedIdx] : stats;
  const isPinned = selectedIdx !== null;

  const cpDisplay = useMemo(() => {
    const v = displayStats.cp / 100;
    return v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
  }, [displayStats.cp]);

  const evalBarPct = Math.max(2, Math.min(98, 50 + displayStats.cp / 10));

  // Top policy entries sorted by weight
  const topPolicy = useMemo(
    () =>
      [...displayStats.policy]
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 10),
    [displayStats.policy]
  );
  const maxWeight = topPolicy.length > 0 ? topPolicy[0].weight : 1;

  const typeLabel = (t: string) =>
    t === "root_eval" ? "root" : t === "search_update" ? "update" : "done";

  return (
    <div className="space-y-3">
      {/* Eval + meta */}
      <div className="flex items-baseline justify-between">
        <span
          className={`text-2xl font-bold font-mono ${
            displayStats.cp >= 0 ? "text-green-600" : "text-red-600"
          }`}
        >
          {cpDisplay}
        </span>
        <span className="text-xs text-muted-foreground">
          {displayStats.nodes} evals • iter {displayStats.iteration}
          {displayStats.type === "search_complete" && " • done"}
        </span>
      </div>

      {/* Eval bar */}
      <div className="h-3 bg-gray-800 rounded overflow-hidden">
        <div
          className="h-full bg-white rounded transition-all duration-300 ease-out"
          style={{ width: `${evalBarPct}%` }}
        />
      </div>

      {/* Best move */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Best:</span>
        <Badge className="font-mono">
          {fmtMove(isPinned ? displayStats.bestMove : (bestMove ?? displayStats.bestMove))}
        </Badge>
        <Badge variant="outline" className="text-xs font-mono">
          {isPinned
            ? `${typeLabel(displayStats.type)} #${selectedIdx! + 1}`
            : displayStats.type === "root_eval"
            ? "NN prior"
            : displayStats.type === "search_update"
            ? "searching…"
            : "final"}
        </Badge>
        {isPinned && (
          <Button
            variant="ghost"
            size="sm"
            className="h-5 px-1.5 text-[10px] text-muted-foreground"
            onClick={() => setSelectedIdx(null)}
          >
            ✕ live
          </Button>
        )}
      </div>

      {/* PV line */}
      {displayStats.pv && displayStats.pv.length > 0 && (
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground">
            Principal Variation:
          </span>
          <div className="text-xs font-mono text-muted-foreground break-all">
            {displayStats.pv
              .slice(0, 40)
              .map((m, i) => (
                <span key={i}>
                  {i > 0 && " "}
                  <span
                    className={
                      i === 0 ? "text-foreground font-medium" : ""
                    }
                  >
                    {fmtMove(m)}
                  </span>
                </span>
              ))}
            {displayStats.pv.length > 40 && " ..."}
          </div>
        </div>
      )}

      <Separator />

      {/* Modified Policy Distribution */}
      <CatGPTPolicyChart
        policy={topPolicy}
        bestMove={displayStats.bestMove}
        maxWeight={maxWeight}
        fen={fen}
      />

      <Separator />

      {/* DistQ Mini-Histogram */}
      <DistQHistogram distQ={displayStats.distQ} />

      {/* Search history table */}
      {history.length > 1 && (
        <>
          <Separator />
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
              Search History
              <span className="ml-1 font-normal normal-case">(click to inspect)</span>
            </span>
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full text-xs font-mono">
                <thead className="text-muted-foreground sticky top-0 bg-card">
                  <tr>
                    <th className="text-left py-0.5 pr-2">Type</th>
                    <th className="text-right py-0.5 pr-2">Eval</th>
                    <th className="text-left py-0.5 pr-2">Best</th>
                    <th className="text-right py-0.5">Evals</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((s, i) => {
                    const prevBest = i > 0 ? history[i - 1].bestMove : null;
                    const changed = prevBest !== null && s.bestMove !== prevBest;
                    const isSelected = selectedIdx === i;
                    return (
                      <tr
                        key={i}
                        className={`cursor-pointer hover:bg-muted/50 ${
                          isSelected
                            ? "bg-muted ring-1 ring-blue-500/40 text-foreground font-medium"
                            : changed
                            ? "text-amber-500 font-medium"
                            : i === history.length - 1
                            ? "text-foreground font-medium"
                            : "text-muted-foreground"
                        }`}
                        onClick={() =>
                          setSelectedIdx(isSelected ? null : i)
                        }
                      >
                        <td className="py-0.5 pr-2 text-xs">
                          {typeLabel(s.type)}
                        </td>
                        <td
                          className={`text-right py-0.5 pr-2 ${
                            s.cp >= 0 ? "text-green-600" : "text-red-600"
                          }`}
                        >
                          {(s.cp / 100).toFixed(2)}
                        </td>
                        <td className="py-0.5 pr-2">
                          <span className={changed ? "underline" : ""}>
                            {fmtMove(s.bestMove)}
                          </span>
                        </td>
                        <td className="text-right py-0.5">{s.nodes}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── DistQ Mini-Histogram ─────────────────────────────────────────

// ─── Shared CatGPT Policy Chart (with Q values) ──────────────────

function CatGPTPolicyChart({
  policy,
  bestMove,
  maxWeight,
  fen,
}: {
  policy: CatGPTSearchStats["policy"];
  bestMove: string;
  maxWeight: number;
  fen: string;
}) {
  const { notationFormat } = usePositionStore();
  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  const fmtQ = (q: number) => {
    const cp = 100.7066 * Math.tan(q * 1.5637541897);
    const cpStr = cp >= 0 ? `+${(cp / 100).toFixed(2)}` : (cp / 100).toFixed(2);
    return `${q >= 0 ? "+" : ""}${q.toFixed(3)} (${cpStr})`;
  };

  return (
    <div className="space-y-1.5">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Modified Policy
      </h4>
      {policy.map((entry) => {
        const label = fmtMove(entry.move);
        const pct = (entry.weight * 100).toFixed(1);
        const barWidth = (entry.weight / maxWeight) * 100;
        const isBest = entry.move === bestMove;

        return (
          <div key={entry.move} className="space-y-0">
            <div className="flex items-center gap-2">
              <span
                className={`w-12 text-right font-mono text-sm ${
                  isBest ? "font-bold text-foreground" : "font-medium"
                }`}
              >
                {label}
              </span>
              <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
                <div
                  className={`h-full rounded transition-all duration-300 ${
                    isBest ? "bg-amber-500" : "bg-blue-500"
                  }`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <span className="w-14 text-right text-xs text-muted-foreground font-mono">
                {pct}%
              </span>
            </div>
            {entry.q != null && (
              <div className="flex items-center gap-2 ml-14">
                <span
                  className={`text-[10px] font-mono ${
                    entry.q >= 0 ? "text-green-600" : "text-red-600"
                  }`}
                >
                  Q {fmtQ(entry.q)}
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── DistQ Mini-Histogram ─────────────────────────────────────────

function DistQHistogram({ distQ }: { distQ: number[] }) {
  const maxProb = useMemo(
    () => Math.max(...distQ, 0.001),
    [distQ]
  );

  // Show a compact 81-bar histogram
  return (
    <div className="space-y-1.5">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Value Distribution (distQ)
      </h4>
      <div className="flex items-end gap-px h-16 bg-muted/30 rounded p-1">
        {distQ.map((prob, i) => {
          const height = (prob / maxProb) * 100;
          // Color: red (loss) → gray (draw) → green (win)
          const t = i / Math.max(distQ.length - 1, 1); // 0 to 1
          const r = Math.round(239 * (1 - t) + 34 * t);
          const g = Math.round(68 * (1 - t) + 197 * t);
          const b = Math.round(68 * (1 - t) + 94 * t);
          return (
            <div
              key={i}
              className="flex-1 rounded-t transition-all duration-300"
              style={{
                height: `${Math.max(height, 1)}%`,
                backgroundColor: `rgb(${r},${g},${b})`,
                opacity: prob > 0.001 ? 1 : 0.2,
              }}
              title={`Bin ${i}: ${(prob * 100).toFixed(1)}%`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
        <span>Loss</span>
        <span>Draw</span>
        <span>Win</span>
      </div>
    </div>
  );
}

// ─── UCI Stats Display (Stockfish / Leela) ────────────────────────

function UCIStatsDisplay({
  latestInfo,
  depthHistory,
  bestMove,
  fen,
}: {
  latestInfo: EngineInfoLine;
  depthHistory: EngineInfoLine[];
  bestMove: string | null;
  fen: string;
}) {
  const { notationFormat } = usePositionStore();
  // null = show latest (live); number = pinned to a history entry
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  const fmtEval = (info: EngineInfoLine) => {
    if (info.score.type === "mate") return `M${info.score.value}`;
    const cp = info.score.value / 100;
    return cp >= 0 ? `+${cp.toFixed(2)}` : cp.toFixed(2);
  };

  // The info line we're displaying
  const displayInfo = selectedIdx !== null ? depthHistory[selectedIdx] : latestInfo;
  const isPinned = selectedIdx !== null;

  const evalBarPct = displayInfo.score.type === "mate"
    ? displayInfo.score.value > 0 ? 95 : 5
    : Math.max(2, Math.min(98, 50 + displayInfo.score.value / 10));

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <span
          className={`text-2xl font-bold font-mono ${
            displayInfo.score.value >= 0
              ? "text-green-600"
              : "text-red-600"
          }`}
        >
          {fmtEval(displayInfo)}
        </span>
        <span className="text-xs text-muted-foreground">
          depth {displayInfo.depth}
          {displayInfo.seldepth ? `/${displayInfo.seldepth}` : ""} •{" "}
          {(displayInfo.nodes / 1000).toFixed(0)}k nodes
          {displayInfo.nps
            ? ` • ${(displayInfo.nps / 1000).toFixed(0)}k nps`
            : ""}
        </span>
      </div>

      {/* Eval bar */}
      <div className="h-3 bg-gray-800 rounded overflow-hidden">
        <div
          className="h-full bg-white rounded transition-all duration-300 ease-out"
          style={{ width: `${evalBarPct}%` }}
        />
      </div>

      {/* Best move */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Best:</span>
        <Badge className="font-mono">
          {fmtMove(
            isPinned
              ? displayInfo.pv[0] ?? "-"
              : bestMove ?? displayInfo.pv[0] ?? "-"
          )}
        </Badge>
        {isPinned && (
          <>
            <Badge variant="outline" className="text-xs font-mono">
              depth {displayInfo.depth}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              className="h-5 px-1.5 text-[10px] text-muted-foreground"
              onClick={() => setSelectedIdx(null)}
            >
              ✕ live
            </Button>
          </>
        )}
      </div>

      {/* PV line */}
      {displayInfo.pv.length > 0 && (
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground">
            Principal Variation:
          </span>
          <div className="text-xs font-mono text-muted-foreground break-all">
            {displayInfo.pv
              .slice(0, 12)
              .map((m, i) => (
                <span key={i}>
                  {i > 0 && " "}
                  <span
                    className={
                      i === 0 ? "text-foreground font-medium" : ""
                    }
                  >
                    {m}
                  </span>
                </span>
              ))}
            {displayInfo.pv.length > 12 && " ..."}
          </div>
        </div>
      )}

      {/* WDL */}
      {displayInfo.wdl && (
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground">WDL:</span>
          <div className="flex h-5 rounded overflow-hidden text-[10px] font-medium">
            <div
              className="bg-green-500 flex items-center justify-center text-white"
              style={{ width: `${displayInfo.wdl.win / 10}%` }}
            >
              {displayInfo.wdl.win > 80 &&
                `${(displayInfo.wdl.win / 10).toFixed(0)}%`}
            </div>
            <div
              className="bg-gray-400 flex items-center justify-center text-white"
              style={{ width: `${displayInfo.wdl.draw / 10}%` }}
            >
              {displayInfo.wdl.draw > 80 &&
                `${(displayInfo.wdl.draw / 10).toFixed(0)}%`}
            </div>
            <div
              className="bg-red-500 flex items-center justify-center text-white"
              style={{ width: `${displayInfo.wdl.loss / 10}%` }}
            >
              {displayInfo.wdl.loss > 80 &&
                `${(displayInfo.wdl.loss / 10).toFixed(0)}%`}
            </div>
          </div>
        </div>
      )}

      {/* Depth history table */}
      {depthHistory.length > 0 && (
        <>
          <Separator />
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
              Depth History
              <span className="ml-1 font-normal normal-case">(click to inspect)</span>
            </span>
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full text-xs font-mono">
                <thead className="text-muted-foreground sticky top-0 bg-card">
                  <tr>
                    <th className="text-left py-0.5 pr-2">D</th>
                    <th className="text-right py-0.5 pr-2">Eval</th>
                    <th className="text-left py-0.5 pr-2">Best</th>
                    <th className="text-right py-0.5">Nodes</th>
                  </tr>
                </thead>
                <tbody>
                  {depthHistory.map((info, i) => {
                    const prevBest = i > 0 ? depthHistory[i - 1].pv[0] : null;
                    const bestChanged = prevBest !== null && info.pv[0] !== prevBest;
                    const isSelected = selectedIdx === i;
                    return (
                      <tr
                        key={i}
                        className={`cursor-pointer hover:bg-muted/50 ${
                          isSelected
                            ? "bg-muted ring-1 ring-blue-500/40 text-foreground font-medium"
                            : bestChanged
                            ? "text-amber-500 font-medium"
                            : i === depthHistory.length - 1
                            ? "text-foreground font-medium"
                            : "text-muted-foreground"
                        }`}
                        onClick={() =>
                          setSelectedIdx(isSelected ? null : i)
                        }
                      >
                        <td className="py-0.5 pr-2">{info.depth}</td>
                        <td
                          className={`text-right py-0.5 pr-2 ${
                            info.score.value >= 0
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {info.score.type === "mate"
                            ? `M${info.score.value}`
                            : (info.score.value / 100).toFixed(2)}
                        </td>
                        <td className="py-0.5 pr-2">
                          <span className={bestChanged ? "underline" : ""}>
                            {info.pv[0] ? fmtMove(info.pv[0]) : "-"}
                          </span>
                        </td>
                        <td className="text-right py-0.5">
                          {info.nodes >= 1000
                            ? `${(info.nodes / 1000).toFixed(0)}k`
                            : info.nodes}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
