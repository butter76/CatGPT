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
  engine: "leela" | "stockfish" | "catgpt" | "catgpt_mcts";
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

export type EngineKind = "leela" | "stockfish" | "catgpt" | "catgpt_mcts";

// ─── CatGPT Search Stats ─────────────────────────────────────────

/** A single move with its modified policy weight and Q value from the MCTS search */
export interface CatGPTSearchPolicyEntry {
  move: UCIMove;
  weight: number;
  /** Q value from parent's perspective [-1, 1]. Only present for expanded children. */
  q?: number;
}

/** Stats emitted by the CatGPT search binary during analysis */
export interface CatGPTSearchStats {
  type: "root_eval" | "search_update" | "search_complete";
  bestMove: UCIMove;
  cp: number;
  nodes: number;
  iteration: number;
  /** 81-bin distributional Q from the root's perspective */
  distQ: number[];
  /** Modified policy weights: allocation/N_adjusted for expanded, raw prior for unexpanded */
  policy: CatGPTSearchPolicyEntry[];
  /** Principal variation (best line by allocation-based selection). Absent for root_eval. */
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
