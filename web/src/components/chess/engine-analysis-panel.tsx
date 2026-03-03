"use client";

import { useState, useEffect } from "react";
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
import type { EngineInfoLine, EngineKind } from "@/lib/types";
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
    startAnalysis,
    stopAnalysis,
    reset,
  } = useEngineAnalysis();

  const { notationFormat } = usePositionStore();
  const [selectedEngine, setSelectedEngine] = useState<EngineKind>("stockfish");
  const [nodes, setNodes] = useState(500000);
  const [availableEngines, setAvailableEngines] = useState<string[]>([]);

  // Default nodes: 500k for Stockfish (CPU-cheap), 5k for GPU engines
  const handleEngineChange = (eng: EngineKind) => {
    setSelectedEngine(eng);
    setNodes(eng === "stockfish" ? 500000 : 5000);
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

  // Format a move for display
  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  // Format eval
  const fmtEval = (info: EngineInfoLine) => {
    if (info.score.type === "mate") {
      return `M${info.score.value}`;
    }
    const cp = info.score.value / 100;
    return cp >= 0 ? `+${cp.toFixed(2)}` : cp.toFixed(2);
  };

  // Eval bar position (0 = black winning, 100 = white winning)
  const evalBarPct = latestInfo
    ? latestInfo.score.type === "mate"
      ? latestInfo.score.value > 0
        ? 95
        : 5
      : Math.max(2, Math.min(98, 50 + latestInfo.score.value / 10))
    : 50;

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
                <Label className="text-xs">Nodes</Label>
                <Input
                  type="number"
                  value={nodes}
                  onChange={(e) => setNodes(parseInt(e.target.value) || 1)}
                  className="h-8 text-xs font-mono"
                  min={100}
                  max={100000000}
                  step={100000}
                />
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={handleStart} className="flex-1" size="sm">
                <Play className="w-4 h-4 mr-1" />
                Analyze
              </Button>
              {bestMove && (
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

        {/* Live eval */}
        {latestInfo && (
          <>
            <Separator />
            {/* Eval display */}
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span
                  className={`text-2xl font-bold font-mono ${
                    latestInfo.score.value >= 0
                      ? "text-green-600"
                      : "text-red-600"
                  }`}
                >
                  {fmtEval(latestInfo)}
                </span>
                <span className="text-xs text-muted-foreground">
                  depth {latestInfo.depth}
                  {latestInfo.seldepth ? `/${latestInfo.seldepth}` : ""} •{" "}
                  {(latestInfo.nodes / 1000).toFixed(0)}k nodes
                  {latestInfo.nps
                    ? ` • ${(latestInfo.nps / 1000).toFixed(0)}k nps`
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
              {bestMove && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">Best:</span>
                  <Badge className="font-mono">{fmtMove(bestMove)}</Badge>
                </div>
              )}

              {/* PV line */}
              {latestInfo.pv.length > 0 && (
                <div className="space-y-1">
                  <span className="text-xs text-muted-foreground">
                    Principal Variation:
                  </span>
                  <div className="text-xs font-mono text-muted-foreground break-all">
                    {latestInfo.pv
                      .slice(0, 12)
                      .map((m, i) => (
                        <span key={i}>
                          {i > 0 && " "}
                          <span
                            className={
                              i === 0
                                ? "text-foreground font-medium"
                                : ""
                            }
                          >
                            {notationFormat === "algebraic"
                              ? m // PV moves need the full position to convert; show UCI
                              : m}
                          </span>
                        </span>
                      ))}
                    {latestInfo.pv.length > 12 && " ..."}
                  </div>
                </div>
              )}

              {/* WDL (lc0) */}
              {latestInfo.wdl && (
                <div className="space-y-1">
                  <span className="text-xs text-muted-foreground">WDL:</span>
                  <div className="flex h-5 rounded overflow-hidden text-[10px] font-medium">
                    <div
                      className="bg-green-500 flex items-center justify-center text-white"
                      style={{ width: `${latestInfo.wdl.win / 10}%` }}
                    >
                      {latestInfo.wdl.win > 80 &&
                        `${(latestInfo.wdl.win / 10).toFixed(0)}%`}
                    </div>
                    <div
                      className="bg-gray-400 flex items-center justify-center text-white"
                      style={{ width: `${latestInfo.wdl.draw / 10}%` }}
                    >
                      {latestInfo.wdl.draw > 80 &&
                        `${(latestInfo.wdl.draw / 10).toFixed(0)}%`}
                    </div>
                    <div
                      className="bg-red-500 flex items-center justify-center text-white"
                      style={{ width: `${latestInfo.wdl.loss / 10}%` }}
                    >
                      {latestInfo.wdl.loss > 80 &&
                        `${(latestInfo.wdl.loss / 10).toFixed(0)}%`}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Depth history table */}
            {depthHistory.length > 0 && (
              <>
                <Separator />
                <div className="space-y-1">
                  <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
                    Depth History
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
                        {depthHistory.map((info, i) => (
                          <tr
                            key={i}
                            className={
                              i === depthHistory.length - 1
                                ? "text-foreground font-medium"
                                : "text-muted-foreground"
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
                              {info.pv[0] ? fmtMove(info.pv[0]) : "-"}
                            </td>
                            <td className="text-right py-0.5">
                              {info.nodes >= 1000
                                ? `${(info.nodes / 1000).toFixed(0)}k`
                                : info.nodes}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {/* Saved indicator */}
            {saved && (
              <div className="flex items-center gap-1.5 text-xs text-green-600">
                <Save className="w-3 h-3" />
                Analysis saved to database
              </div>
            )}
          </>
        )}

        {/* Empty state */}
        {!running && !latestInfo && !error && (
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
