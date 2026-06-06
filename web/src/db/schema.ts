import {
  pgTable,
  pgEnum,
  text,
  serial,
  integer,
  doublePrecision,
  timestamp,
  unique,
  jsonb,
  boolean,
} from "drizzle-orm/pg-core";

// ─── Enums ────────────────────────────────────────────────────────

export const positionTypeEnum = pgEnum("position_type", ["SHARP", "FORTRESS"]);
export const outcomeEnum = pgEnum("outcome", ["win", "loss", "draw", "unknown"]);
export const moveAnnotationKindEnum = pgEnum("move_annotation_kind", [
  "blunder",
  "correct",
]);
export const engineKindEnum = pgEnum("engine_kind", ["leela", "stockfish", "catgpt"]);
export const blunderTagEnum = pgEnum("blunder_tag", ["catgpt", "stockfish", "leela"]);
export const benchmarkRunStatusEnum = pgEnum("benchmark_run_status", [
  "pending",
  "running",
  "completed",
  "failed",
  "cancelled",
]);

// ─── Tables ───────────────────────────────────────────────────────

export const positions = pgTable("positions", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  description: text("description"),
  type: positionTypeEnum("type").notNull(),
  fen: text("fen").notNull(),
  expectedOutcome: outcomeEnum("expected_outcome"),
  blunderTag: blunderTagEnum("blunder_tag"),
  longBench: boolean("long_bench").notNull().default(false),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const moveAnnotations = pgTable(
  "move_annotations",
  {
    id: serial("id").primaryKey(),
    positionId: text("position_id")
      .notNull()
      .references(() => positions.id, { onDelete: "cascade" }),
    move: text("move").notNull(),
    annotation: moveAnnotationKindEnum("annotation").notNull(),
  },
  (table) => [unique().on(table.positionId, table.move)]
);

export const networkAnalyses = pgTable("network_analyses", {
  id: serial("id").primaryKey(),
  positionId: text("position_id")
    .notNull()
    .references(() => positions.id, { onDelete: "cascade" }),
  bestQ: doublePrecision("best_q").notNull(),
  wdlWin: doublePrecision("wdl_win").notNull(),
  wdlDraw: doublePrecision("wdl_draw").notNull(),
  wdlLoss: doublePrecision("wdl_loss").notNull(),
  nodes: integer("nodes").notNull().default(1),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const policyEntries = pgTable("policy_entries", {
  id: serial("id").primaryKey(),
  analysisId: integer("analysis_id")
    .notNull()
    .references(() => networkAnalyses.id, { onDelete: "cascade" }),
  move: text("move").notNull(),
  probability: doublePrecision("probability").notNull(),
});

export const engineAnalyses = pgTable("engine_analyses", {
  id: serial("id").primaryKey(),
  positionId: text("position_id")
    .notNull()
    .references(() => positions.id, { onDelete: "cascade" }),
  engine: engineKindEnum("engine").notNull(),
  bestMove: text("best_move").notNull(),
  evaluation: doublePrecision("evaluation").notNull(),
  depth: integer("depth").notNull(),
  nodes: integer("nodes").notNull(),
  pv: text("pv").array().notNull().default([]),
  depthHistory: jsonb("depth_history").notNull().default([]),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

// ─── LongBench ────────────────────────────────────────────────────

export const benchmarkRuns = pgTable("benchmark_runs", {
  id: serial("id").primaryKey(),
  engine: text("engine").notNull(),
  maxNodes: integer("max_nodes").notNull(),
  status: benchmarkRunStatusEnum("status").notNull().default("pending"),
  aggregateScore: doublePrecision("aggregate_score"),
  positionCount: integer("position_count"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  startedAt: timestamp("started_at", { withTimezone: true }),
  finishedAt: timestamp("finished_at", { withTimezone: true }),
});

export const benchmarkPositionResults = pgTable(
  "benchmark_position_results",
  {
    id: serial("id").primaryKey(),
    runId: integer("run_id")
      .notNull()
      .references(() => benchmarkRuns.id, { onDelete: "cascade" }),
    positionId: text("position_id")
      .notNull()
      .references(() => positions.id, { onDelete: "cascade" }),
    score: doublePrecision("score"),
    stableNodes: integer("stable_nodes"),
    failed: boolean("failed").notNull().default(false),
    finalCp: doublePrecision("final_cp"),
    finalBestMove: text("final_best_move"),
    totalNodes: integer("total_nodes"),
    statsHistory: jsonb("stats_history").notNull().default([]),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [unique().on(table.runId, table.positionId)]
);

// ─── Tournaments ──────────────────────────────────────────────────

export const tournamentStatusEnum = pgEnum("tournament_status", [
  "pending",
  "running",
  "completed",
  "failed",
  "cancelled",
]);
export const gameStatusEnum = pgEnum("game_status", [
  "pending",
  "in_progress",
  "completed",
]);
export const gameResultEnum = pgEnum("game_result", [
  "white_win",
  "black_win",
  "draw",
]);
export const gameSideEnum = pgEnum("game_side", ["white", "black"]);
export const uciLogEngineEnum = pgEnum("uci_log_engine", [
  "white",
  "black",
  "combined",
]);

export const tournaments = pgTable("tournaments", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  whiteLabel: text("white_label").notNull(),
  blackLabel: text("black_label").notNull(),
  /** Engine config { name, command, options?, initStrings? } for the first engine. */
  whiteConfig: jsonb("white_config").notNull(),
  blackConfig: jsonb("black_config").notNull(),
  /** cutechess tc string, e.g. "900+5". */
  timeControl: text("time_control").notNull(),
  totalGames: integer("total_games").notNull(),
  concurrency: integer("concurrency").notNull().default(1),
  openingBook: text("opening_book"),
  drawMoveNumber: integer("draw_move_number").notNull().default(1),
  drawMoveCount: integer("draw_move_count").notNull().default(7),
  drawScoreCp: integer("draw_score_cp").notNull().default(25),
  tbPath: text("tb_path"),
  status: tournamentStatusEnum("status").notNull().default("pending"),
  scoreWhite: integer("score_white").notNull().default(0),
  scoreBlack: integer("score_black").notNull().default(0),
  scoreDraw: integer("score_draw").notNull().default(0),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  startedAt: timestamp("started_at", { withTimezone: true }),
  finishedAt: timestamp("finished_at", { withTimezone: true }),
});

export const tournamentGames = pgTable(
  "tournament_games",
  {
    id: serial("id").primaryKey(),
    tournamentId: integer("tournament_id")
      .notNull()
      .references(() => tournaments.id, { onDelete: "cascade" }),
    gameNumber: integer("game_number").notNull(),
    whiteEngine: text("white_engine").notNull(),
    blackEngine: text("black_engine").notNull(),
    status: gameStatusEnum("status").notNull().default("pending"),
    result: gameResultEnum("result"),
    termination: text("termination"),
    pgn: text("pgn"),
    openingFen: text("opening_fen"),
    finalFen: text("final_fen"),
    plyCount: integer("ply_count").notNull().default(0),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    finishedAt: timestamp("finished_at", { withTimezone: true }),
  },
  (table) => [unique().on(table.tournamentId, table.gameNumber)]
);

export const gameMoves = pgTable(
  "game_moves",
  {
    id: serial("id").primaryKey(),
    gameId: integer("game_id")
      .notNull()
      .references(() => tournamentGames.id, { onDelete: "cascade" }),
    ply: integer("ply").notNull(),
    mover: gameSideEnum("mover").notNull(),
    san: text("san").notNull(),
    uci: text("uci").notNull(),
    fenAfter: text("fen_after").notNull(),
    evalCp: integer("eval_cp"),
    depth: integer("depth"),
    timeMs: integer("time_ms"),
    // Clock remaining (ms) for each side as reported by cutechess in the `go`
    // command immediately *before* this ply was played (null for book moves /
    // non-clock time controls).
    whiteClockMs: integer("white_clock_ms"),
    blackClockMs: integer("black_clock_ms"),
  },
  (table) => [unique().on(table.gameId, table.ply)]
);

export const gameUciLogs = pgTable("game_uci_logs", {
  id: serial("id").primaryKey(),
  gameId: integer("game_id")
    .notNull()
    .references(() => tournamentGames.id, { onDelete: "cascade" }),
  engine: uciLogEngineEnum("engine").notNull().default("combined"),
  content: text("content").notNull(),
});
