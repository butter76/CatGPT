CREATE TYPE "public"."benchmark_run_status" AS ENUM('pending', 'running', 'completed', 'failed', 'cancelled');--> statement-breakpoint
CREATE TYPE "public"."blunder_tag" AS ENUM('catgpt', 'stockfish', 'leela');--> statement-breakpoint
CREATE TYPE "public"."engine_kind" AS ENUM('leela', 'stockfish', 'catgpt');--> statement-breakpoint
CREATE TYPE "public"."game_result" AS ENUM('white_win', 'black_win', 'draw');--> statement-breakpoint
CREATE TYPE "public"."game_side" AS ENUM('white', 'black');--> statement-breakpoint
CREATE TYPE "public"."game_status" AS ENUM('pending', 'in_progress', 'completed');--> statement-breakpoint
CREATE TYPE "public"."move_annotation_kind" AS ENUM('blunder', 'correct');--> statement-breakpoint
CREATE TYPE "public"."outcome" AS ENUM('win', 'loss', 'draw', 'unknown');--> statement-breakpoint
CREATE TYPE "public"."position_type" AS ENUM('SHARP', 'FORTRESS');--> statement-breakpoint
CREATE TYPE "public"."tournament_status" AS ENUM('pending', 'running', 'completed', 'failed', 'cancelled');--> statement-breakpoint
CREATE TYPE "public"."uci_log_engine" AS ENUM('white', 'black', 'combined');--> statement-breakpoint
CREATE TABLE "benchmark_position_results" (
	"id" serial PRIMARY KEY NOT NULL,
	"run_id" integer NOT NULL,
	"position_id" text NOT NULL,
	"score" double precision,
	"stable_nodes" integer,
	"failed" boolean DEFAULT false NOT NULL,
	"final_cp" double precision,
	"final_best_move" text,
	"total_nodes" integer,
	"stats_history" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "benchmark_position_results_run_id_position_id_unique" UNIQUE("run_id","position_id")
);
--> statement-breakpoint
CREATE TABLE "benchmark_runs" (
	"id" serial PRIMARY KEY NOT NULL,
	"engine" text NOT NULL,
	"max_nodes" integer NOT NULL,
	"status" "benchmark_run_status" DEFAULT 'pending' NOT NULL,
	"aggregate_score" double precision,
	"position_count" integer,
	"error_message" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"started_at" timestamp with time zone,
	"finished_at" timestamp with time zone
);
--> statement-breakpoint
CREATE TABLE "engine_analyses" (
	"id" serial PRIMARY KEY NOT NULL,
	"position_id" text NOT NULL,
	"engine" "engine_kind" NOT NULL,
	"best_move" text NOT NULL,
	"evaluation" double precision NOT NULL,
	"depth" integer NOT NULL,
	"nodes" integer NOT NULL,
	"pv" text[] DEFAULT '{}' NOT NULL,
	"depth_history" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "game_moves" (
	"id" serial PRIMARY KEY NOT NULL,
	"game_id" integer NOT NULL,
	"ply" integer NOT NULL,
	"mover" "game_side" NOT NULL,
	"san" text NOT NULL,
	"uci" text NOT NULL,
	"fen_after" text NOT NULL,
	"eval_cp" integer,
	"depth" integer,
	"time_ms" integer,
	"white_clock_ms" integer,
	"black_clock_ms" integer,
	CONSTRAINT "game_moves_game_id_ply_unique" UNIQUE("game_id","ply")
);
--> statement-breakpoint
CREATE TABLE "game_uci_logs" (
	"id" serial PRIMARY KEY NOT NULL,
	"game_id" integer NOT NULL,
	"engine" "uci_log_engine" DEFAULT 'combined' NOT NULL,
	"content" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "move_annotations" (
	"id" serial PRIMARY KEY NOT NULL,
	"position_id" text NOT NULL,
	"move" text NOT NULL,
	"annotation" "move_annotation_kind" NOT NULL,
	CONSTRAINT "move_annotations_position_id_move_unique" UNIQUE("position_id","move")
);
--> statement-breakpoint
CREATE TABLE "network_analyses" (
	"id" serial PRIMARY KEY NOT NULL,
	"position_id" text NOT NULL,
	"best_q" double precision NOT NULL,
	"wdl_win" double precision NOT NULL,
	"wdl_draw" double precision NOT NULL,
	"wdl_loss" double precision NOT NULL,
	"nodes" integer DEFAULT 1 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "policy_entries" (
	"id" serial PRIMARY KEY NOT NULL,
	"analysis_id" integer NOT NULL,
	"move" text NOT NULL,
	"probability" double precision NOT NULL
);
--> statement-breakpoint
CREATE TABLE "positions" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"description" text,
	"type" "position_type" NOT NULL,
	"fen" text NOT NULL,
	"expected_outcome" "outcome",
	"blunder_tag" "blunder_tag",
	"long_bench" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "tournament_games" (
	"id" serial PRIMARY KEY NOT NULL,
	"tournament_id" integer NOT NULL,
	"game_number" integer NOT NULL,
	"white_engine" text NOT NULL,
	"black_engine" text NOT NULL,
	"status" "game_status" DEFAULT 'pending' NOT NULL,
	"result" "game_result",
	"termination" text,
	"pgn" text,
	"opening_fen" text,
	"final_fen" text,
	"ply_count" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"finished_at" timestamp with time zone,
	CONSTRAINT "tournament_games_tournament_id_game_number_unique" UNIQUE("tournament_id","game_number")
);
--> statement-breakpoint
CREATE TABLE "tournaments" (
	"id" serial PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"white_label" text NOT NULL,
	"black_label" text NOT NULL,
	"white_config" jsonb NOT NULL,
	"black_config" jsonb NOT NULL,
	"time_control" text NOT NULL,
	"total_games" integer NOT NULL,
	"concurrency" integer DEFAULT 1 NOT NULL,
	"opening_book" text,
	"draw_move_number" integer DEFAULT 1 NOT NULL,
	"draw_move_count" integer DEFAULT 7 NOT NULL,
	"draw_score_cp" integer DEFAULT 25 NOT NULL,
	"tb_path" text,
	"status" "tournament_status" DEFAULT 'pending' NOT NULL,
	"score_white" integer DEFAULT 0 NOT NULL,
	"score_black" integer DEFAULT 0 NOT NULL,
	"score_draw" integer DEFAULT 0 NOT NULL,
	"error_message" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"started_at" timestamp with time zone,
	"finished_at" timestamp with time zone
);
--> statement-breakpoint
ALTER TABLE "benchmark_position_results" ADD CONSTRAINT "benchmark_position_results_run_id_benchmark_runs_id_fk" FOREIGN KEY ("run_id") REFERENCES "public"."benchmark_runs"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "benchmark_position_results" ADD CONSTRAINT "benchmark_position_results_position_id_positions_id_fk" FOREIGN KEY ("position_id") REFERENCES "public"."positions"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "engine_analyses" ADD CONSTRAINT "engine_analyses_position_id_positions_id_fk" FOREIGN KEY ("position_id") REFERENCES "public"."positions"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "game_moves" ADD CONSTRAINT "game_moves_game_id_tournament_games_id_fk" FOREIGN KEY ("game_id") REFERENCES "public"."tournament_games"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "game_uci_logs" ADD CONSTRAINT "game_uci_logs_game_id_tournament_games_id_fk" FOREIGN KEY ("game_id") REFERENCES "public"."tournament_games"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "move_annotations" ADD CONSTRAINT "move_annotations_position_id_positions_id_fk" FOREIGN KEY ("position_id") REFERENCES "public"."positions"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "network_analyses" ADD CONSTRAINT "network_analyses_position_id_positions_id_fk" FOREIGN KEY ("position_id") REFERENCES "public"."positions"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "policy_entries" ADD CONSTRAINT "policy_entries_analysis_id_network_analyses_id_fk" FOREIGN KEY ("analysis_id") REFERENCES "public"."network_analyses"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "tournament_games" ADD CONSTRAINT "tournament_games_tournament_id_tournaments_id_fk" FOREIGN KEY ("tournament_id") REFERENCES "public"."tournaments"("id") ON DELETE cascade ON UPDATE no action;