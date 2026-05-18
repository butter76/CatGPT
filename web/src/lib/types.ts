// ─── Position Types ───────────────────────────────────────────────

export type PositionType = "SHARP" | "FORTRESS";

export type Outcome = "win" | "loss" | "draw" | "unknown";

export type BlunderTag = "catgpt" | "stockfish" | "leela";

/** A single move in UCI format, e.g. "e2e4", "e7e5" */
export type UCIMove = string;

/** A square on the board, e.g. "e2", "d7" */
export type Square = string;

// ─── Analysis ─────────────────────────────────────────────────────

/** Policy entry: move → probability */
export interface PolicyEntry {
  move: UCIMove;
  probability: number;
}

/** Win/Draw/Loss distribution */
export interface WDL {
  win: number;
  draw: number;
  loss: number;
}

/** Analysis from our own network */
export interface NetworkAnalysis {
  policy: PolicyEntry[];
  wdl: WDL;
  bestQ: number; // best Q value (expected value from search)
  nodes: number;
  timestamp: string;
}

/** A single parsed UCI info line from a running engine */
export interface EngineInfoLine {
  depth: number;
  seldepth?: number;
  score: { type: "cp" | "mate"; value: number };
  wdl?: { win: number; draw: number; loss: number }; // per-mille (lc0)
  nodes: number;
  nps?: number;
  time?: number; // ms
  pv: UCIMove[];
  multipv?: number;
}

/** Final stored analysis from an engine (Leela / Stockfish / CatGPT) */
export interface EngineAnalysis {
  id?: number;
  engine: "leela" | "stockfish" | "catgpt";
  bestMove: UCIMove;
  evaluation: number; // centipawns (or mate value)
  depth: number;
  nodes: number;
  pv: UCIMove[]; // principal variation
  wdl?: { win: number; draw: number; loss: number };
  /** Full per-depth info history from the search (UCI engines) */
  depthHistory?: EngineInfoLine[];
  /** Full search stats history (CatGPT only) */
  catgptHistory?: CatGPTSearchStats[];
  timestamp: string;
}

export type EngineKind = "leela" | "stockfish" | "catgpt";

// ─── CatGPT Search Stats ─────────────────────────────────────────

/**
 * A single move at the root, with its allocation-derived weight and
 * (when available) per-child Q value. Emitted by the LKS-backed
 * `catgpt_search` binary.
 */
export interface CatGPTSearchPolicyEntry {
  move: UCIMove;
  /** Softmax-normalized exp(log_alloc) over all legal moves at the root (sums to ~1). */
  weight: number;
  /**
   * Q from the root side-to-move's perspective in [-1, 1]. Only present
   * for children that have been expanded (TT-hit, position-only terminal,
   * or path-dependent draw); never-evaluated children omit this.
   */
  q?: number;
}

/** Stats emitted by the LKS search binary (`catgpt_search`) during analysis. */
export interface CatGPTSearchStats {
  type: "root_eval" | "search_update" | "search_complete";
  bestMove: UCIMove;
  cp: number;
  nodes: number;
  /**
   * Centi-depth: `round(max_depth() * 100)` from the LKS search. LKS has
   * no global iteration counter — this is the closest analog and grows
   * by ~20 per ID step (default delta_depth=0.2).
   */
  iteration: number;
  /** Per-move allocation-derived weights + per-child Q (when expanded). */
  policy: CatGPTSearchPolicyEntry[];
  /** Greedy best-child PV walk. Absent only when the root has no TT entry yet. */
  pv?: UCIMove[];
}

// ─── Sharp Position ───────────────────────────────────────────────

export interface SharpMoveAnnotation {
  move: UCIMove;
  /** "blunder" = this move is a mistake; "correct" = must be played */
  annotation: "blunder" | "correct";
}

// ─── Position ─────────────────────────────────────────────────────

export interface Position {
  id: string;
  name: string;
  description?: string;
  type: PositionType;
  fen: string;

  // SHARP-specific
  moveAnnotations?: SharpMoveAnnotation[];

  // FORTRESS-specific
  expectedOutcome?: Outcome;

  // Blunder tagging (CatGPT/Stockfish/Leela blunder)
  blunderTag?: BlunderTag;

  // LongBench membership
  longBench: boolean;

  // Analyses
  networkAnalysis?: NetworkAnalysis;
  engineAnalyses?: EngineAnalysis[];

  createdAt: string;
  updatedAt: string;
}

// ─── LongBench ────────────────────────────────────────────────────

export type BenchmarkRunStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

/** Compact per-event sample stored in a benchmark position result. */
export interface BenchmarkStatsSample {
  nodes: number;
  cp: number;
  bestMove: UCIMove;
}

export interface BenchmarkRun {
  id: number;
  engine: string;
  maxNodes: number;
  status: BenchmarkRunStatus;
  aggregateScore: number | null;
  positionCount: number | null;
  errorMessage: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface BenchmarkPositionResult {
  id: number;
  runId: number;
  positionId: string;
  score: number | null;
  stableNodes: number | null;
  failed: boolean;
  finalCp: number | null;
  finalBestMove: string | null;
  totalNodes: number | null;
  statsHistory: BenchmarkStatsSample[];
  createdAt: string;
}

/** Run detail: the run row plus per-position results joined with position metadata. */
export interface BenchmarkRunDetail extends BenchmarkRun {
  results: (BenchmarkPositionResult & {
    position: Pick<
      Position,
      "id" | "name" | "type" | "fen" | "expectedOutcome" | "moveAnnotations"
    >;
  })[];
}

// ─── Notation helpers ─────────────────────────────────────────────

export type NotationFormat = "uci" | "algebraic";
