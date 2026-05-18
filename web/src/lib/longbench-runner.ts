/**
 * LongBenchRunner — process-wide singleton that owns the execution loop for
 * LongBench runs, completely decoupled from any HTTP / SSE connection.
 *
 * Goals:
 *   - Survives browser refreshes and SSE disconnects. The executor keeps
 *     running; a new SSE connection rehydrates from the in-memory snapshot.
 *   - Does NOT need to survive Next.js restarts. On import we sweep any
 *     orphaned `running` rows in the DB to `failed`.
 *
 * Lifecycle of a run:
 *   1. An API route creates a DB row (status=pending) and calls
 *      `longBenchRunner.start(runId)`.
 *   2. `start` spins up an async executor that iterates LongBench positions,
 *      spawns `catgpt_search` sequentially, persists each position result,
 *      and pushes events into an in-memory ring of subscribers.
 *   3. Any number of SSE routes can `subscribe(runId, ...)` at any time.
 *      On subscribe the subscriber receives a single `snapshot` event with
 *      the full current state, then streams incremental events.
 *   4. When the executor terminates, `finalSnapshot` is retained briefly so
 *      late-joining clients still see the result in the stream; the DB row
 *      remains authoritative for long-term persistence.
 */

import type { ChildProcess } from "child_process";
import {
  failBenchmarkRun,
  completeBenchmarkRun,
  getBenchmarkRun,
  getLongBenchPositions,
  markBenchmarkRunRunning,
  markStaleRunsFailed,
  upsertBenchmarkPositionResult,
} from "@/db/queries";
import { runCatGPTAnalysis } from "@/lib/catgpt-engine";
import {
  extractGroundTruth,
  isEventCorrect,
  scorePosition,
  type LongBenchGroundTruth,
} from "@/lib/longbench-score";
import type {
  BenchmarkStatsSample,
  Outcome,
  UCIMove,
} from "@/lib/types";

// ─── Types ────────────────────────────────────────────────────────

export interface RunnerPositionSummary {
  id: string;
  name: string;
  type: "SHARP" | "FORTRESS";
  fen: string;
  expectedOutcome: Outcome | null;
  correctMoves: UCIMove[];
  blunderMoves: UCIMove[];
}

export interface RunnerPositionState {
  done: boolean;
  skipped: boolean;
  skippedReason?: string;
  score: number | null;
  stableNodes: number | null;
  failed: boolean;
  finalCp: number | null;
  finalBestMove: string | null;
  totalNodes: number | null;
  /** Latest-ish stats samples — capped at MAX_RECENT_STATS_PER_POSITION. */
  recentStats: BenchmarkStatsSample[];
  currentCorrect: boolean | null;
}

export interface RunnerSnapshot {
  runId: number;
  status: "pending" | "running" | "completed" | "failed";
  error: string | null;
  engine: string;
  maxNodes: number;
  positions: RunnerPositionSummary[];
  /** Index of the position currently being searched, or -1 before any start. */
  currentIndex: number;
  perPosition: Record<string, RunnerPositionState>;
  aggregateScore: number | null;
  startedAt: string | null;
  finishedAt: string | null;
}

export type RunnerEvent =
  | { type: "run_started"; data: Pick<RunnerSnapshot, "engine" | "maxNodes" | "positions"> }
  | { type: "position_started"; data: { positionId: string; index: number } }
  | {
      type: "stats";
      data: {
        positionId: string;
        nodes: number;
        cp: number;
        bestMove: string;
        correct: boolean;
      };
    }
  | {
      type: "position_done";
      data: {
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
    }
  | { type: "run_complete"; data: { aggregateScore: number } }
  | { type: "error"; data: { message: string; positionId?: string } };

export type Subscriber = (event: RunnerEvent) => void;

// ─── Tunables ────────────────────────────────────────────────────

/** Maximum number of recent stats samples to keep in memory per position. */
const MAX_RECENT_STATS_PER_POSITION = 2000;

/** Keep a finished run's snapshot in memory this long for late subscribers. */
const FINISHED_RETENTION_MS = 60 * 60 * 1000; // 1 hour

// ─── Runner ──────────────────────────────────────────────────────

interface ActiveRun {
  snapshot: RunnerSnapshot;
  subscribers: Set<Subscriber>;
  /** Pointer to the live child process so we can cancel mid-position. */
  currentChild: ChildProcess | null;
  aborted: boolean;
  finishedTimer: NodeJS.Timeout | null;
}

class LongBenchRunner {
  private runs = new Map<number, ActiveRun>();
  private sweepDone = false;

  /** Called lazily on first use; also safe to invoke explicitly. */
  async initStartupSweep(): Promise<void> {
    if (this.sweepDone) return;
    this.sweepDone = true;
    try {
      await markStaleRunsFailed();
    } catch (err) {
      console.error("[longbench] startup sweep failed:", err);
    }
  }

  getSnapshot(runId: number): RunnerSnapshot | null {
    return this.runs.get(runId)?.snapshot ?? null;
  }

  /**
   * Subscribe to the run. Returns an `unsubscribe` function and the current
   * snapshot (so callers can emit a single `snapshot` event up front). If the
   * run has already ended and been evicted, returns `{ snapshot: null }` and
   * the listener is never invoked.
   */
  subscribe(
    runId: number,
    listener: Subscriber
  ): { snapshot: RunnerSnapshot | null; unsubscribe: () => void } {
    const active = this.runs.get(runId);
    if (!active) return { snapshot: null, unsubscribe: () => {} };
    active.subscribers.add(listener);
    return {
      snapshot: active.snapshot,
      unsubscribe: () => {
        active.subscribers.delete(listener);
      },
    };
  }

  /**
   * Kick off the executor for a pending run. Fire-and-forget.
   *
   * Idempotent: calling `start` for a runId that is already active does
   * nothing (returns existing snapshot). Returns `null` if the run row is
   * missing or not in `pending` state.
   */
  async start(runId: number): Promise<RunnerSnapshot | null> {
    await this.initStartupSweep();

    const existing = this.runs.get(runId);
    if (existing) return existing.snapshot;

    const runRow = await getBenchmarkRun(runId);
    if (!runRow) return null;
    if (runRow.status !== "pending") return null;

    const snapshot: RunnerSnapshot = {
      runId,
      status: "pending",
      error: null,
      engine: runRow.engine,
      maxNodes: runRow.maxNodes,
      positions: [],
      currentIndex: -1,
      perPosition: {},
      aggregateScore: null,
      startedAt: null,
      finishedAt: null,
    };

    const active: ActiveRun = {
      snapshot,
      subscribers: new Set(),
      currentChild: null,
      aborted: false,
      finishedTimer: null,
    };
    this.runs.set(runId, active);

    // Fire-and-forget: run the executor on its own microtask.
    void this.executeRun(active).catch((err) => {
      console.error(`[longbench] run ${runId} crashed:`, err);
    });

    return snapshot;
  }

  /** Request abort. The current position's engine is killed; loop exits. */
  abort(runId: number): boolean {
    const active = this.runs.get(runId);
    if (!active) return false;
    active.aborted = true;
    if (active.currentChild) {
      try {
        active.currentChild.kill("SIGKILL");
      } catch {
        // ignore
      }
    }
    return true;
  }

  // ─── internals ──────────────────────────────────────────────────

  private emit(active: ActiveRun, event: RunnerEvent) {
    for (const sub of active.subscribers) {
      try {
        sub(event);
      } catch (err) {
        console.error("[longbench] subscriber threw:", err);
      }
    }
  }

  private async executeRun(active: ActiveRun) {
    const { snapshot } = active;
    const runId = snapshot.runId;

    try {
      const allPositions = await getLongBenchPositions();
      if (allPositions.length === 0) {
        snapshot.status = "failed";
        snapshot.error = "No positions flagged as LongBench";
        snapshot.finishedAt = new Date().toISOString();
        await failBenchmarkRun(runId, snapshot.error);
        this.emit(active, {
          type: "error",
          data: { message: snapshot.error },
        });
        return;
      }

      await markBenchmarkRunRunning(runId, allPositions.length);
      snapshot.status = "running";
      snapshot.startedAt = new Date().toISOString();

      const positions: RunnerPositionSummary[] = allPositions.map((p) => ({
        id: p.id,
        name: p.name,
        type: p.type,
        fen: p.fen,
        expectedOutcome: p.expectedOutcome ?? null,
        correctMoves:
          p.moveAnnotations
            ?.filter((a) => a.annotation === "correct")
            .map((a) => a.move) ?? [],
        blunderMoves:
          p.moveAnnotations
            ?.filter((a) => a.annotation === "blunder")
            .map((a) => a.move) ?? [],
      }));

      snapshot.positions = positions;
      for (const pos of positions) {
        snapshot.perPosition[pos.id] = {
          done: false,
          skipped: false,
          score: null,
          stableNodes: null,
          failed: false,
          finalCp: null,
          finalBestMove: null,
          totalNodes: null,
          recentStats: [],
          currentCorrect: null,
        };
      }

      this.emit(active, {
        type: "run_started",
        data: {
          engine: snapshot.engine,
          maxNodes: snapshot.maxNodes,
          positions,
        },
      });

      const scoredScores: number[] = [];

      for (let i = 0; i < positions.length; i++) {
        if (active.aborted) break;
        const pos = positions[i];
        snapshot.currentIndex = i;

        const truth: LongBenchGroundTruth | null = extractGroundTruth({
          type: pos.type,
          expectedOutcome: pos.expectedOutcome ?? undefined,
          moveAnnotations: [
            ...pos.correctMoves.map((move) => ({
              move,
              annotation: "correct" as const,
            })),
            ...pos.blunderMoves.map((move) => ({
              move,
              annotation: "blunder" as const,
            })),
          ],
        });

        this.emit(active, {
          type: "position_started",
          data: { positionId: pos.id, index: i },
        });

        const posState = snapshot.perPosition[pos.id];

        if (!truth) {
          posState.done = true;
          posState.skipped = true;
          posState.skippedReason =
            pos.type === "FORTRESS"
              ? "missing expectedOutcome"
              : "no move annotations";
          await upsertBenchmarkPositionResult(runId, pos.id, {
            score: null,
            stableNodes: null,
            failed: false,
            finalCp: null,
            finalBestMove: null,
            totalNodes: null,
            statsHistory: [],
          });
          this.emit(active, {
            type: "position_done",
            data: {
              positionId: pos.id,
              skipped: true,
              reason: posState.skippedReason,
            },
          });
          continue;
        }

        // Spawn the engine and accumulate events.
        const statsHistory: BenchmarkStatsSample[] = [];
        let finalCp: number | null = null;
        let finalBestMove: string | null = null;
        let totalNodes: number | null = null;
        let engineError: string | null = null;

        try {
          const iter = runCatGPTAnalysis({
            fen: pos.fen,
            nodes: snapshot.maxNodes,
            onSpawn: (child) => {
              active.currentChild = child;
            },
          });
          for await (const event of iter) {
            if (active.aborted) break;
            if (event.type === "stats") {
              const s = event.data;
              const sample: BenchmarkStatsSample = {
                nodes: s.nodes,
                cp: s.cp,
                bestMove: s.bestMove,
              };
              statsHistory.push(sample);
              finalCp = s.cp;
              finalBestMove = s.bestMove;
              totalNodes = s.nodes;
              const correct = isEventCorrect(sample, truth);
              posState.currentCorrect = correct;
              posState.recentStats.push(sample);
              if (posState.recentStats.length > MAX_RECENT_STATS_PER_POSITION) {
                posState.recentStats.splice(
                  0,
                  posState.recentStats.length - MAX_RECENT_STATS_PER_POSITION
                );
              }
              this.emit(active, {
                type: "stats",
                data: {
                  positionId: pos.id,
                  nodes: s.nodes,
                  cp: s.cp,
                  bestMove: s.bestMove,
                  correct,
                },
              });
            } else if (event.type === "bestmove") {
              finalBestMove = event.data.bestMove;
            } else if (event.type === "error") {
              engineError = event.data.message;
            }
          }
        } catch (err) {
          engineError = err instanceof Error ? err.message : String(err);
        } finally {
          active.currentChild = null;
        }

        if (active.aborted) break;

        if (engineError) {
          snapshot.status = "failed";
          snapshot.error = `Engine error on ${pos.name} (${pos.id}): ${engineError}`;
          snapshot.finishedAt = new Date().toISOString();
          await failBenchmarkRun(runId, snapshot.error);
          this.emit(active, {
            type: "error",
            data: { message: engineError, positionId: pos.id },
          });
          return;
        }

        const result = scorePosition(statsHistory, truth, snapshot.maxNodes);

        await upsertBenchmarkPositionResult(runId, pos.id, {
          score: result.score,
          stableNodes: result.stableNodes,
          failed: result.failed,
          finalCp,
          finalBestMove,
          totalNodes,
          statsHistory,
        });

        posState.done = true;
        posState.score = result.score;
        posState.stableNodes = result.stableNodes;
        posState.failed = result.failed;
        posState.finalCp = finalCp;
        posState.finalBestMove = finalBestMove;
        posState.totalNodes = totalNodes;

        scoredScores.push(result.score);

        this.emit(active, {
          type: "position_done",
          data: {
            positionId: pos.id,
            score: result.score,
            stableNodes: result.stableNodes,
            failed: result.failed,
            finalCp,
            finalBestMove,
            totalNodes,
          },
        });
      }

      if (active.aborted) {
        snapshot.status = "failed";
        snapshot.error = "Aborted by user";
        snapshot.finishedAt = new Date().toISOString();
        await failBenchmarkRun(runId, snapshot.error);
        this.emit(active, {
          type: "error",
          data: { message: snapshot.error },
        });
        return;
      }

      const aggregate =
        scoredScores.length > 0
          ? scoredScores.reduce((a, b) => a + b, 0) / scoredScores.length
          : 0;

      await completeBenchmarkRun(runId, aggregate);
      snapshot.status = "completed";
      snapshot.aggregateScore = aggregate;
      snapshot.finishedAt = new Date().toISOString();

      this.emit(active, {
        type: "run_complete",
        data: { aggregateScore: aggregate },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      snapshot.status = "failed";
      snapshot.error = message;
      snapshot.finishedAt = new Date().toISOString();
      await failBenchmarkRun(runId, message).catch(() => {});
      this.emit(active, { type: "error", data: { message } });
    } finally {
      // Keep the snapshot around for late subscribers, then evict.
      active.finishedTimer = setTimeout(() => {
        this.runs.delete(runId);
      }, FINISHED_RETENTION_MS);
    }
  }
}

// ─── Module-level singleton (survives module reloads in dev) ─────

declare global {
  // eslint-disable-next-line no-var
  var __longBenchRunner: LongBenchRunner | undefined;
}

export const longBenchRunner: LongBenchRunner =
  globalThis.__longBenchRunner ??
  (globalThis.__longBenchRunner = new LongBenchRunner());
