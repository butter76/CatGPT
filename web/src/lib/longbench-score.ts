/**
 * LongBench scoring helper.
 *
 * Given an ordered sequence of search events (with `nodes`, `cp`, `bestMove`)
 * and a position's ground truth (FORTRESS expected outcome or SHARP correct
 * move set), compute:
 *   - stableNodes: the smallest node count from which the prediction is
 *     correct for the rest of the search (i.e. it never flickers back to
 *     incorrect afterwards). If the very last event is incorrect, no such
 *     boundary exists.
 *   - score: `log(stableNodes)` if stable, else `log(10 * maxNodes)` (failure).
 *
 * Natural log is used; base is irrelevant for averaging / comparisons.
 */

import type {
    BenchmarkStatsSample,
    Outcome,
    Position,
    SharpMoveAnnotation,
    UCIMove,
  } from "./types";

  /** Threshold (in centipawns) that separates draw from decisive. */
  export const CP_DECISIVE_THRESHOLD = 100;

  /** Classify a centipawn evaluation from the side-to-move's perspective. */
  export function cpToOutcome(cp: number): "win" | "loss" | "draw" {
    if (cp > CP_DECISIVE_THRESHOLD) return "win";
    if (cp < -CP_DECISIVE_THRESHOLD) return "loss";
    return "draw";
  }

  export type LongBenchGroundTruth =
    | { kind: "fortress"; expected: Exclude<Outcome, "unknown"> }
    /** SHARP position with an explicit set of correct moves (bestMove ∈ set). */
    | { kind: "sharp_correct"; correctMoves: Set<UCIMove> }
    /** SHARP position annotated only with blunders (bestMove ∉ set). */
    | { kind: "sharp_avoid"; blunderMoves: Set<UCIMove> };

  /**
   * Extract the ground truth a position needs to be scoreable on LongBench.
   *
   * Returns null for positions that are not scoreable:
   *   - FORTRESS without an `expectedOutcome` (or `unknown`).
   *   - SHARP without any move annotations at all.
   *
   * SHARP precedence: if any `correct` annotations exist they define the truth
   * (correct iff bestMove ∈ that set). Otherwise, a non-empty `blunder` set
   * defines the truth as "correct iff bestMove is NOT in the blunder set".
   */
  export function extractGroundTruth(
    position: Pick<Position, "type" | "expectedOutcome" | "moveAnnotations">
  ): LongBenchGroundTruth | null {
    if (position.type === "FORTRESS") {
      const expected = position.expectedOutcome;
      if (!expected || expected === "unknown") return null;
      return { kind: "fortress", expected };
    }
    const anns = position.moveAnnotations ?? [];
    const correct = anns.filter(
      (a: SharpMoveAnnotation) => a.annotation === "correct"
    );
    if (correct.length > 0) {
      return {
        kind: "sharp_correct",
        correctMoves: new Set(correct.map((a) => a.move)),
      };
    }
    const blunders = anns.filter(
      (a: SharpMoveAnnotation) => a.annotation === "blunder"
    );
    if (blunders.length > 0) {
      return {
        kind: "sharp_avoid",
        blunderMoves: new Set(blunders.map((a) => a.move)),
      };
    }
    return null;
  }

  /** Test a single event against ground truth. */
  export function isEventCorrect(
    event: Pick<BenchmarkStatsSample, "cp" | "bestMove">,
    truth: LongBenchGroundTruth
  ): boolean {
    if (truth.kind === "fortress") {
      return cpToOutcome(event.cp) === truth.expected;
    }
    if (truth.kind === "sharp_correct") {
      return truth.correctMoves.has(event.bestMove);
    }
    return !truth.blunderMoves.has(event.bestMove);
  }

  export interface LongBenchScoreResult {
    /** Node count at which the prediction became permanently correct, or null. */
    stableNodes: number | null;
    /** Score used for averaging (natural log). */
    score: number;
    /** True iff the final event is not correct (so `log(10 * maxNodes)` was used). */
    failed: boolean;
  }

  /** Penalty factor: failed positions score `log(FAILURE_MULTIPLIER * maxNodes)`. */
  export const FAILURE_MULTIPLIER = 10;

  /**
   * Compute the LongBench score for a single position given its event stream.
   *
   * Events must be sorted by `nodes` ascending. Duplicate node counts are
   * tolerated and treated as later events (the later one wins on the transition
   * logic).
   */
  export function scorePosition(
    events: Pick<BenchmarkStatsSample, "nodes" | "cp" | "bestMove">[],
    truth: LongBenchGroundTruth,
    maxNodes: number
  ): LongBenchScoreResult {
    const failScore = Math.log(FAILURE_MULTIPLIER * maxNodes);

    if (events.length === 0) {
      return { stableNodes: null, score: failScore, failed: true };
    }

    // If final event is not correct there is no stable suffix.
    const lastCorrect = isEventCorrect(events[events.length - 1], truth);
    if (!lastCorrect) {
      return { stableNodes: null, score: failScore, failed: true };
    }

    // Walk backward to find the first event of the maximal all-correct suffix.
    let boundary = events.length - 1;
    for (let i = events.length - 2; i >= 0; i--) {
      if (!isEventCorrect(events[i], truth)) break;
      boundary = i;
    }

    const stableNodes = Math.max(1, events[boundary].nodes);
    return {
      stableNodes,
      score: Math.log(stableNodes),
      failed: false,
    };
  }
