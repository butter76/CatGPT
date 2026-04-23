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
export const engineKindEnum = pgEnum("engine_kind", ["leela", "stockfish", "catgpt", "catgpt_mcts"]);
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
