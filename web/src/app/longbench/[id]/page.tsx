"use client";

import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CompactBoard } from "@/components/chess/analysis-board";
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  Hourglass,
  Zap,
  Castle,
} from "lucide-react";
import type {
  BenchmarkRun,
  BenchmarkRunDetail,
  BenchmarkRunStatus,
  BenchmarkStatsSample,
  Outcome,
  UCIMove,
} from "@/lib/types";
import {
  extractGroundTruth,
  isEventCorrect,
  type LongBenchGroundTruth,
} from "@/lib/longbench-score";

// ─── Shared helpers ───────────────────────────────────────────────

function statusBadge(status: BenchmarkRunStatus) {
  switch (status) {
    case "pending":
      return (
        <Badge variant="outline" className="border-yellow-500 text-yellow-600">
          <Hourglass className="w-3 h-3 mr-1" /> pending
        </Badge>
      );
    case "running":
      return (
        <Badge variant="outline" className="border-blue-500 text-blue-600">
          <Loader2 className="w-3 h-3 mr-1 animate-spin" /> running
        </Badge>
      );
    case "completed":
      return (
        <Badge variant="outline" className="border-green-500 text-green-600">
          <CheckCircle2 className="w-3 h-3 mr-1" /> completed
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="outline" className="border-red-500 text-red-600">
          <XCircle className="w-3 h-3 mr-1" /> failed
        </Badge>
      );
    case "cancelled":
      return (
        <Badge variant="outline" className="border-gray-400 text-gray-500">
          cancelled
        </Badge>
      );
  }
}

// ─── Live SSE state ───────────────────────────────────────────────

interface LivePositionSummary {
  id: string;
  name: string;
  type: "SHARP" | "FORTRESS";
  fen: string;
  expectedOutcome: Outcome | null;
  correctMoves: UCIMove[];
}

interface LivePositionState {
  statsHistory: BenchmarkStatsSample[];
  currentCorrect: boolean | null;
  done: boolean;
  skipped: boolean;
  skippedReason?: string;
  score: number | null;
  stableNodes: number | null;
  failed: boolean;
  finalCp: number | null;
  finalBestMove: string | null;
  totalNodes: number | null;
}

function emptyLiveState(): LivePositionState {
  return {
    statsHistory: [],
    currentCorrect: null,
    done: false,
    skipped: false,
    score: null,
    stableNodes: null,
    failed: false,
    finalCp: null,
    finalBestMove: null,
    totalNodes: null,
  };
}

// ─── Page ─────────────────────────────────────────────────────────

export default function LongBenchRunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const runId = Number(id);
  const searchParams = useSearchParams();

  const [detail, setDetail] = useState<BenchmarkRunDetail | null>(null);
  const [loading, setLoading] = useState(true);

  // Live execution state (populated while the SSE stream is open).
  const [live, setLive] = useState<null | {
    positions: LivePositionSummary[];
    currentIndex: number;
    byPosition: Record<string, LivePositionState>;
    finalAggregate: number | null;
    streamError: string | null;
    streamDone: boolean;
  }>(null);
  const esRef = useRef<EventSource | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    fetch(`/api/longbench/runs/${runId}`)
      .then((r) => r.json())
      .then((d: BenchmarkRunDetail) => setDetail(d))
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => {
    load();
  }, [load]);

  const startStream = useCallback(() => {
    if (esRef.current) return;
    setLive({
      positions: [],
      currentIndex: -1,
      byPosition: {},
      finalAggregate: null,
      streamError: null,
      streamDone: false,
    });
    const es = new EventSource(`/api/longbench/runs/${runId}/execute`);
    esRef.current = es;

    es.addEventListener("run_started", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        positions: LivePositionSummary[];
      };
      setLive((prev) =>
        prev
          ? {
              ...prev,
              positions: data.positions,
              byPosition: Object.fromEntries(
                data.positions.map((p) => [p.id, emptyLiveState()])
              ),
            }
          : prev
      );
    });

    es.addEventListener("position_started", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        positionId: string;
        index: number;
      };
      setLive((prev) =>
        prev ? { ...prev, currentIndex: data.index } : prev
      );
    });

    es.addEventListener("stats", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        positionId: string;
        nodes: number;
        cp: number;
        bestMove: string;
        correct: boolean;
      };
      setLive((prev) => {
        if (!prev) return prev;
        const cur = prev.byPosition[data.positionId] ?? emptyLiveState();
        const nextHistory = cur.statsHistory;
        nextHistory.push({
          nodes: data.nodes,
          cp: data.cp,
          bestMove: data.bestMove,
        });
        return {
          ...prev,
          byPosition: {
            ...prev.byPosition,
            [data.positionId]: {
              ...cur,
              statsHistory: nextHistory,
              currentCorrect: data.correct,
            },
          },
        };
      });
    });

    es.addEventListener("position_done", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        positionId: string;
        skipped?: boolean;
        reason?: string;
        score?: number | null;
        stableNodes?: number | null;
        failed?: boolean;
        finalCp?: number | null;
        finalBestMove?: string | null;
        totalNodes?: number | null;
      };
      setLive((prev) => {
        if (!prev) return prev;
        const cur = prev.byPosition[data.positionId] ?? emptyLiveState();
        return {
          ...prev,
          byPosition: {
            ...prev.byPosition,
            [data.positionId]: {
              ...cur,
              done: true,
              skipped: !!data.skipped,
              skippedReason: data.reason,
              score: data.score ?? null,
              stableNodes: data.stableNodes ?? null,
              failed: !!data.failed,
              finalCp: data.finalCp ?? null,
              finalBestMove: data.finalBestMove ?? null,
              totalNodes: data.totalNodes ?? null,
            },
          },
        };
      });
    });

    es.addEventListener("run_complete", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        aggregateScore: number;
      };
      setLive((prev) =>
        prev ? { ...prev, finalAggregate: data.aggregateScore } : prev
      );
    });

    es.addEventListener("error", (ev) => {
      const raw = (ev as MessageEvent).data;
      const message = raw
        ? (() => {
            try {
              return JSON.parse(raw).message ?? String(raw);
            } catch {
              return String(raw);
            }
          })()
        : "Connection error";
      setLive((prev) =>
        prev ? { ...prev, streamError: message } : prev
      );
    });

    es.addEventListener("done", () => {
      setLive((prev) => (prev ? { ...prev, streamDone: true } : prev));
      es.close();
      esRef.current = null;
      // Refresh the canonical detail from the DB.
      load();
    });
  }, [runId, load]);

  // Auto-start if requested via query string (from the "Start new run" flow).
  useEffect(() => {
    if (searchParams.get("autostart") === "1" && detail?.status === "pending") {
      startStream();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detail?.status]);

  useEffect(() => {
    return () => {
      esRef.current?.close();
      esRef.current = null;
    };
  }, []);

  if (loading && !detail) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="text-center py-16">
        <p className="text-lg text-muted-foreground">Run not found</p>
        <Button variant="ghost" asChild className="mt-4">
          <Link href="/longbench">
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to LongBench
          </Link>
        </Button>
      </div>
    );
  }

  const canStart = detail.status === "pending" && !live;
  const isStreaming = !!live && !live.streamDone;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/longbench">
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Link>
          </Button>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Gauge className="w-6 h-6 text-indigo-500" />
            LongBench run #{detail.id}
          </h1>
          <p className="text-sm text-muted-foreground font-mono break-all">
            {detail.engine}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw
              className={`w-4 h-4 mr-1.5 ${loading ? "animate-spin" : ""}`}
            />
            Refresh
          </Button>
          {canStart && (
            <Button size="sm" onClick={startStream}>
              <Play className="w-4 h-4 mr-1.5" />
              Start
            </Button>
          )}
        </div>
      </div>

      <RunSummaryCard run={detail} live={live} />

      {live && (
        <LiveProgressCard live={live} isStreaming={isStreaming} />
      )}

      <ResultsTable detail={detail} live={live} />
    </div>
  );
}

// ─── Run summary card ─────────────────────────────────────────────

function RunSummaryCard({
  run,
  live,
}: {
  run: BenchmarkRun;
  live: {
    positions: LivePositionSummary[];
    byPosition: Record<string, LivePositionState>;
    finalAggregate: number | null;
  } | null;
}) {
  // Live aggregate = mean of scored (non-skipped, done) positions from live state.
  const liveAggregate = useMemo(() => {
    if (!live) return null;
    const scores = Object.values(live.byPosition)
      .filter((s) => s.done && !s.skipped && s.score != null)
      .map((s) => s.score as number);
    if (scores.length === 0) return null;
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  }, [live]);

  const aggregate =
    live?.finalAggregate ?? liveAggregate ?? run.aggregateScore ?? null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center justify-between">
          <span>Run metadata</span>
          {statusBadge(run.status)}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
          <div>
            <dt className="text-xs text-muted-foreground">Max nodes</dt>
            <dd className="font-mono">{run.maxNodes.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Positions</dt>
            <dd className="font-mono">{run.positionCount ?? "–"}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Aggregate score</dt>
            <dd className="font-mono text-lg font-semibold">
              {aggregate != null ? aggregate.toFixed(4) : "–"}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">
              {run.status === "completed" ? "Finished" : "Started"}
            </dt>
            <dd className="text-xs text-muted-foreground">
              {run.finishedAt
                ? new Date(run.finishedAt).toLocaleString()
                : run.startedAt
                ? new Date(run.startedAt).toLocaleString()
                : new Date(run.createdAt).toLocaleString()}
            </dd>
          </div>
        </dl>
        {run.errorMessage && (
          <p className="mt-3 text-sm text-red-500 font-mono break-all">
            {run.errorMessage}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Live progress card ───────────────────────────────────────────

function LiveProgressCard({
  live,
  isStreaming,
}: {
  live: {
    positions: LivePositionSummary[];
    currentIndex: number;
    byPosition: Record<string, LivePositionState>;
    streamError: string | null;
    streamDone: boolean;
  };
  isStreaming: boolean;
}) {
  const total = live.positions.length;
  const doneCount = Object.values(live.byPosition).filter((s) => s.done).length;
  const pct = total > 0 ? (doneCount / total) * 100 : 0;

  const current =
    live.currentIndex >= 0 && live.currentIndex < live.positions.length
      ? live.positions[live.currentIndex]
      : null;
  const currentState = current ? live.byPosition[current.id] : null;
  const lastSample =
    currentState?.statsHistory[currentState.statsHistory.length - 1];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          {isStreaming ? (
            <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          )}
          Live progress
          <span className="text-xs font-normal text-muted-foreground ml-auto">
            {doneCount}/{total} positions
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overall progress bar */}
        <div className="h-2 w-full rounded bg-muted overflow-hidden">
          <div
            className="h-full bg-indigo-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>

        {current && currentState && (
          <div className="grid grid-cols-1 sm:grid-cols-[140px_1fr] gap-4">
            <div className="shrink-0">
              <CompactBoard fen={current.fen} width={140} />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className={
                    current.type === "SHARP"
                      ? "border-amber-500 text-amber-600"
                      : "border-blue-500 text-blue-600"
                  }
                >
                  {current.type === "SHARP" ? (
                    <Zap className="w-3 h-3 mr-0.5" />
                  ) : (
                    <Castle className="w-3 h-3 mr-0.5" />
                  )}
                  {current.type}
                </Badge>
                <span className="font-semibold">{current.name}</span>
              </div>
              {lastSample ? (
                <div className="text-sm space-y-0.5 font-mono">
                  <div>
                    <span className="text-muted-foreground">Nodes: </span>
                    {lastSample.nodes.toLocaleString()}
                  </div>
                  <div>
                    <span className="text-muted-foreground">CP: </span>
                    <span
                      className={
                        lastSample.cp >= 0 ? "text-green-600" : "text-red-600"
                      }
                    >
                      {(lastSample.cp / 100).toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Best: </span>
                    {lastSample.bestMove}
                  </div>
                  <div>
                    <span className="text-muted-foreground">Status: </span>
                    {currentState.currentCorrect === true ? (
                      <span className="text-green-600">correct</span>
                    ) : currentState.currentCorrect === false ? (
                      <span className="text-red-600">incorrect</span>
                    ) : (
                      <span className="text-muted-foreground">…</span>
                    )}
                  </div>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">Spinning up engine…</p>
              )}
            </div>
          </div>
        )}

        {live.streamError && (
          <p className="text-sm text-red-500">{live.streamError}</p>
        )}
        {live.streamDone && !live.streamError && (
          <p className="text-sm text-green-600">Run stream finished.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ─── Results table ────────────────────────────────────────────────

type MergedResult = {
  positionId: string;
  name: string;
  type: "SHARP" | "FORTRESS";
  fen: string;
  skipped: boolean;
  failed: boolean;
  score: number | null;
  stableNodes: number | null;
  finalCp: number | null;
  finalBestMove: string | null;
  totalNodes: number | null;
  statsHistory: BenchmarkStatsSample[];
  truth: LongBenchGroundTruth | null;
  done: boolean;
};

function mergeResults(
  detail: BenchmarkRunDetail,
  live: {
    positions: LivePositionSummary[];
    byPosition: Record<string, LivePositionState>;
  } | null
): MergedResult[] {
  const merged = new Map<string, MergedResult>();

  // Seed from DB.
  for (const r of detail.results) {
    const truth = extractGroundTruth({
      type: r.position.type,
      expectedOutcome: r.position.expectedOutcome,
      moveAnnotations: r.position.moveAnnotations,
    });
    merged.set(r.positionId, {
      positionId: r.positionId,
      name: r.position.name,
      type: r.position.type,
      fen: r.position.fen,
      skipped: r.score == null && !r.failed && r.stableNodes == null && r.totalNodes == null,
      failed: r.failed,
      score: r.score,
      stableNodes: r.stableNodes,
      finalCp: r.finalCp,
      finalBestMove: r.finalBestMove,
      totalNodes: r.totalNodes,
      statsHistory: r.statsHistory,
      truth,
      done: true,
    });
  }

  // Overlay live state (live is more up-to-date for the current run).
  if (live) {
    for (const pos of live.positions) {
      // Derive truth from the live summary — lets the Target column render
      // immediately, without waiting for a DB refresh.
      const liveTruth = extractGroundTruth({
        type: pos.type,
        expectedOutcome: pos.expectedOutcome ?? undefined,
        moveAnnotations: pos.correctMoves.map((move) => ({
          move,
          annotation: "correct",
        })),
      });
      const s = live.byPosition[pos.id];
      const existing = merged.get(pos.id);
      if (!s) {
        if (!existing) {
          merged.set(pos.id, {
            positionId: pos.id,
            name: pos.name,
            type: pos.type,
            fen: pos.fen,
            skipped: false,
            failed: false,
            score: null,
            stableNodes: null,
            finalCp: null,
            finalBestMove: null,
            totalNodes: null,
            statsHistory: [],
            truth: liveTruth,
            done: false,
          });
        } else if (!existing.truth) {
          merged.set(pos.id, { ...existing, truth: liveTruth });
        }
        continue;
      }
      merged.set(pos.id, {
        positionId: pos.id,
        name: pos.name,
        type: pos.type,
        fen: pos.fen,
        skipped: s.skipped,
        failed: s.failed,
        score: s.score ?? existing?.score ?? null,
        stableNodes: s.stableNodes ?? existing?.stableNodes ?? null,
        finalCp: s.finalCp ?? existing?.finalCp ?? null,
        finalBestMove: s.finalBestMove ?? existing?.finalBestMove ?? null,
        totalNodes: s.totalNodes ?? existing?.totalNodes ?? null,
        statsHistory: s.statsHistory.length > 0 ? s.statsHistory : existing?.statsHistory ?? [],
        truth: existing?.truth ?? liveTruth,
        done: s.done,
      });
    }
  }

  return [...merged.values()].sort((a, b) => {
    // Undone last, then by score ascending (lower is better).
    if (a.done !== b.done) return a.done ? -1 : 1;
    if (a.skipped !== b.skipped) return a.skipped ? 1 : -1;
    const sa = a.score ?? Infinity;
    const sb = b.score ?? Infinity;
    return sa - sb;
  });
}

function ResultsTable({
  detail,
  live,
}: {
  detail: BenchmarkRunDetail;
  live: {
    positions: LivePositionSummary[];
    byPosition: Record<string, LivePositionState>;
  } | null;
}) {
  const rows = useMemo(() => mergeResults(detail, live), [detail, live]);
  const [openId, setOpenId] = useState<string | null>(null);

  if (rows.length === 0) {
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-muted-foreground">
          No results yet.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Per-position results</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="text-left py-2 px-4">Position</th>
                <th className="text-left py-2 px-4">Type</th>
                <th className="text-left py-2 px-4">Target</th>
                <th className="text-right py-2 px-4">Final CP</th>
                <th className="text-left py-2 px-4">Final Best</th>
                <th className="text-right py-2 px-4">Stable nodes</th>
                <th className="text-right py-2 px-4">Score</th>
                <th className="text-left py-2 px-4">Result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <ResultRow
                  key={r.positionId}
                  row={r}
                  open={openId === r.positionId}
                  onToggle={() =>
                    setOpenId((prev) =>
                      prev === r.positionId ? null : r.positionId
                    )
                  }
                />
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function ResultRow({
  row,
  open,
  onToggle,
}: {
  row: MergedResult;
  open: boolean;
  onToggle: () => void;
}) {
  const target =
    row.type === "FORTRESS"
      ? row.truth?.kind === "fortress"
        ? row.truth.expected
        : "–"
      : row.truth?.kind === "sharp"
      ? [...row.truth.correctMoves].join(", ")
      : "–";

  return (
    <>
      <tr
        className="border-b last:border-b-0 hover:bg-muted/30 cursor-pointer transition-colors"
        onClick={onToggle}
      >
        <td className="py-2 px-4">
          <Link
            href={`/positions/${row.positionId}`}
            onClick={(e) => e.stopPropagation()}
            className="hover:underline text-foreground font-medium"
          >
            {row.name}
          </Link>
        </td>
        <td className="py-2 px-4">
          <Badge
            variant="outline"
            className={`text-[10px] ${
              row.type === "SHARP"
                ? "border-amber-500 text-amber-600"
                : "border-blue-500 text-blue-600"
            }`}
          >
            {row.type}
          </Badge>
        </td>
        <td className="py-2 px-4 font-mono text-xs text-muted-foreground max-w-[160px] truncate">
          {target}
        </td>
        <td className="py-2 px-4 text-right font-mono">
          {row.finalCp != null ? (row.finalCp / 100).toFixed(2) : "–"}
        </td>
        <td className="py-2 px-4 font-mono text-xs">
          {row.finalBestMove ?? "–"}
        </td>
        <td className="py-2 px-4 text-right font-mono">
          {row.stableNodes != null ? row.stableNodes.toLocaleString() : "–"}
        </td>
        <td className="py-2 px-4 text-right font-mono font-semibold">
          {row.score != null ? row.score.toFixed(3) : "–"}
        </td>
        <td className="py-2 px-4">
          {row.skipped ? (
            <span className="text-xs text-muted-foreground italic">skipped</span>
          ) : !row.done ? (
            <span className="text-xs text-blue-500">running…</span>
          ) : row.failed ? (
            <Badge variant="outline" className="border-red-500 text-red-600 text-[10px]">
              <XCircle className="w-3 h-3 mr-1" /> failed
            </Badge>
          ) : (
            <Badge
              variant="outline"
              className="border-green-500 text-green-600 text-[10px]"
            >
              <CheckCircle2 className="w-3 h-3 mr-1" /> ok
            </Badge>
          )}
        </td>
      </tr>
      {open && (
        <tr className="border-b last:border-b-0 bg-muted/20">
          <td colSpan={8} className="p-4">
            <PositionDetail row={row} />
          </td>
        </tr>
      )}
    </>
  );
}

// ─── Per-position chart ───────────────────────────────────────────

function PositionDetail({ row }: { row: MergedResult }) {
  // Recompute correctness per sample using the shared classifier.
  const truth = row.truth;
  const events = row.statsHistory;

  return (
    <div className="grid grid-cols-1 md:grid-cols-[160px_1fr] gap-4">
      <div className="shrink-0">
        <CompactBoard fen={row.fen} width={160} />
      </div>
      <div className="space-y-2 min-w-0">
        <p className="text-xs text-muted-foreground">
          {events.length} search events recorded.
          {row.stableNodes != null && (
            <>
              {" "}
              Stabilized at{" "}
              <span className="font-mono text-foreground">
                {row.stableNodes.toLocaleString()}
              </span>{" "}
              nodes.
            </>
          )}
        </p>
        {events.length > 0 ? (
          <ScoreChart events={events} truth={truth} stableNodes={row.stableNodes} />
        ) : (
          <p className="text-xs text-muted-foreground italic">
            No event samples available.
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * A simple inline SVG chart plotting CP over log(nodes), shading regions where
 * the event was correct according to the ground truth.
 */
function ScoreChart({
  events,
  truth,
  stableNodes,
}: {
  events: BenchmarkStatsSample[];
  truth: LongBenchGroundTruth | null;
  stableNodes: number | null;
}) {
  const W = 640;
  const H = 160;
  const PAD_L = 40;
  const PAD_R = 8;
  const PAD_T = 8;
  const PAD_B = 24;

  if (events.length === 0) return null;

  const logN = events.map((e) => Math.log(Math.max(1, e.nodes)));
  const minLog = logN[0];
  const maxLog = logN[logN.length - 1];
  const logSpan = Math.max(1e-6, maxLog - minLog);

  const cps = events.map((e) => Math.max(-600, Math.min(600, e.cp)));
  const cpMin = Math.min(-150, ...cps);
  const cpMax = Math.max(150, ...cps);
  const cpSpan = Math.max(1, cpMax - cpMin);

  const x = (logv: number) =>
    PAD_L + ((logv - minLog) / logSpan) * (W - PAD_L - PAD_R);
  const y = (cp: number) =>
    PAD_T + (1 - (cp - cpMin) / cpSpan) * (H - PAD_T - PAD_B);

  // Precompute correctness per event for shading.
  const correct = events.map((e) => (truth ? isEventCorrect(e, truth) : false));

  // Build contiguous "correct" segments for shading.
  const segments: Array<[number, number]> = [];
  let segStart: number | null = null;
  for (let i = 0; i < events.length; i++) {
    if (correct[i] && segStart == null) segStart = i;
    else if (!correct[i] && segStart != null) {
      segments.push([segStart, i - 1]);
      segStart = null;
    }
  }
  if (segStart != null) segments.push([segStart, events.length - 1]);

  // Polyline for CP.
  const path = events
    .map((e, i) => `${i === 0 ? "M" : "L"} ${x(logN[i])} ${y(cps[i])}`)
    .join(" ");

  const stableX =
    stableNodes != null
      ? x(Math.log(Math.max(1, stableNodes)))
      : null;

  // CP threshold guides.
  const yZero = y(0);
  const y100 = y(100);
  const yNeg100 = y(-100);

  // Axis tick labels at integer log values (1, 10, 100, ...).
  const ticks: { x: number; label: string }[] = [];
  const startTick = Math.ceil(minLog / Math.log(10)) - 1;
  const endTick = Math.floor(maxLog / Math.log(10)) + 1;
  for (let k = startTick; k <= endTick; k++) {
    const val = Math.pow(10, k);
    const lv = Math.log(val);
    if (lv < minLog - 0.01 || lv > maxLog + 0.01) continue;
    const xv = x(lv);
    let label: string;
    if (val >= 1_000_000) label = `${(val / 1_000_000).toFixed(0)}M`;
    else if (val >= 1_000) label = `${(val / 1_000).toFixed(0)}k`;
    else label = `${val.toFixed(0)}`;
    ticks.push({ x: xv, label });
  }

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-full h-auto bg-muted/30 rounded"
    >
      {/* Correct shading */}
      {segments.map(([s, e], i) => {
        const xs = x(logN[s]);
        const xe = x(logN[e]);
        return (
          <rect
            key={i}
            x={xs}
            y={PAD_T}
            width={Math.max(1, xe - xs)}
            height={H - PAD_T - PAD_B}
            className="fill-green-500/15"
          />
        );
      })}
      {/* Threshold lines */}
      <line
        x1={PAD_L}
        x2={W - PAD_R}
        y1={yZero}
        y2={yZero}
        className="stroke-muted-foreground/40"
        strokeDasharray="2 3"
      />
      <line
        x1={PAD_L}
        x2={W - PAD_R}
        y1={y100}
        y2={y100}
        className="stroke-green-500/40"
        strokeDasharray="2 3"
      />
      <line
        x1={PAD_L}
        x2={W - PAD_R}
        y1={yNeg100}
        y2={yNeg100}
        className="stroke-red-500/40"
        strokeDasharray="2 3"
      />
      {/* Stable boundary */}
      {stableX != null && (
        <line
          x1={stableX}
          x2={stableX}
          y1={PAD_T}
          y2={H - PAD_B}
          className="stroke-indigo-500"
          strokeWidth={1.5}
        />
      )}
      {/* CP polyline */}
      <path d={path} className="fill-none stroke-blue-500" strokeWidth={1.5} />
      {/* Axes */}
      <line
        x1={PAD_L}
        x2={W - PAD_R}
        y1={H - PAD_B}
        y2={H - PAD_B}
        className="stroke-muted-foreground/60"
      />
      <line
        x1={PAD_L}
        x2={PAD_L}
        y1={PAD_T}
        y2={H - PAD_B}
        className="stroke-muted-foreground/60"
      />
      {/* Y labels */}
      <text
        x={PAD_L - 4}
        y={y100 + 3}
        className="fill-muted-foreground text-[9px]"
        textAnchor="end"
      >
        +1.00
      </text>
      <text
        x={PAD_L - 4}
        y={yZero + 3}
        className="fill-muted-foreground text-[9px]"
        textAnchor="end"
      >
        0.00
      </text>
      <text
        x={PAD_L - 4}
        y={yNeg100 + 3}
        className="fill-muted-foreground text-[9px]"
        textAnchor="end"
      >
        -1.00
      </text>
      {/* X tick labels */}
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={t.x}
            x2={t.x}
            y1={H - PAD_B}
            y2={H - PAD_B + 3}
            className="stroke-muted-foreground/60"
          />
          <text
            x={t.x}
            y={H - PAD_B + 13}
            className="fill-muted-foreground text-[9px]"
            textAnchor="middle"
          >
            {t.label}
          </text>
        </g>
      ))}
      <text
        x={(W + PAD_L) / 2}
        y={H - 4}
        className="fill-muted-foreground text-[9px]"
        textAnchor="middle"
      >
        nodes (log scale)
      </text>
    </svg>
  );
}
