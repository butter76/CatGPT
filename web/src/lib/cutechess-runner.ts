/**
 * CutechessRunner — process-wide singleton that owns a running cutechess-cli
 * match, decoupled from any HTTP / SSE connection (mirrors LongBenchRunner).
 *
 *   - Survives browser refreshes and SSE disconnects; the match keeps running
 *     and a new SSE connection rehydrates from the in-memory snapshot.
 *   - Does NOT survive Next.js restarts. On import we sweep orphaned `running`
 *     tournaments (and `in_progress` games) in the DB.
 *
 * Lifecycle:
 *   1. An API route creates a `tournaments` row (status=pending) and calls
 *      `cutechessRunner.start(id)`.
 *   2. `start` writes engines.json + a work dir, spawns cutechess-cli, and
 *      parses its merged stdout/stderr. cutechess `-debug` gives full UCI I/O
 *      (stored per game for debugging) and lets us reconstruct moves live; the
 *      `-pgnout` file gives authoritative per-move eval/depth/time on finish.
 *   3. Any number of SSE routes can `subscribe(id, ...)`.
 */

import { spawn, type ChildProcess } from "child_process";
import os from "os";
import path from "path";
import {
  completeTournament,
  failTournament,
  finalizeTournamentGame,
  getTournamentRow,
  markStaleTournamentsFailed,
  markTournamentRunning,
  startTournamentGame,
  updateTournamentScores,
} from "@/db/queries";
import {
  CUTECHESS_CLI_PATH,
  LiveGameTracker,
  PGN_FILENAME,
  buildCutechessArgs,
  parseDebugLine,
  parseFinishedGame,
  parseGoClocks,
  parsePositionCommand,
  parseScoreLine,
  parseStartedGame,
  readPgnGameAtIndex,
  reconstructGame,
  scoreStrToResult,
  writeEnginesJson,
  type GoClocks,
  type PositionCommand,
} from "@/lib/cutechess";
import type {
  GameMove,
  GameResult,
  GameSide,
  TournamentGameSummary,
  TournamentStatus,
} from "@/lib/types";

// ─── Snapshot / event types ───────────────────────────────────────

export interface LiveGame {
  gameId: number;
  gameNumber: number;
  white: string;
  black: string;
  openingFen: string;
  moves: GameMove[];
}

export interface TournamentSnapshot {
  tournamentId: number;
  status: TournamentStatus;
  error: string | null;
  totalGames: number;
  whiteLabel: string;
  blackLabel: string;
  scoreFirst: number;
  scoreSecond: number;
  scoreDraws: number;
  /** Completed games accumulated this session. */
  games: TournamentGameSummary[];
  /** In-progress games being reconstructed live. */
  liveGames: LiveGame[];
  startedAt: string | null;
  finishedAt: string | null;
}

export type TournamentEvent =
  | { type: "run_started"; data: { totalGames: number; whiteLabel: string; blackLabel: string } }
  | { type: "game_started"; data: LiveGame }
  | { type: "move"; data: { gameId: number; gameNumber: number; move: GameMove } }
  | { type: "game_finished"; data: { game: TournamentGameSummary } }
  | { type: "score"; data: { first: number; second: number; draws: number } }
  | { type: "run_complete"; data: { first: number; second: number; draws: number } }
  | { type: "error"; data: { message: string } };

export type Subscriber = (event: TournamentEvent) => void;

// ─── Tunables ─────────────────────────────────────────────────────

const FINISHED_RETENTION_MS = 60 * 60 * 1000; // 1 hour

// ─── Internals ────────────────────────────────────────────────────

interface InternalLiveGame extends LiveGame {
  tracker: LiveGameTracker;
  logLines: string[];
}

interface ActiveTournament {
  snapshot: TournamentSnapshot;
  subscribers: Set<Subscriber>;
  child: ChildProcess | null;
  aborted: boolean;
  finishedTimer: NodeJS.Timeout | null;
  workDir: string;
  pgnPath: string;
  // Parsing state
  liveByNumber: Map<number, InternalLiveGame>;
  /** Latest info per engine name: { evalCp, depth }. */
  lastInfo: Map<string, { evalCp: number | null; depth: number | null }>;
  /** Clock state from the most recent `go` command, per engine name. */
  pendingGo: Map<string, GoClocks>;
  currentGameNumber: number | null;
  preamble: string[];
  finishedCount: number;
  queue: Promise<void>;
}

class CutechessRunner {
  private runs = new Map<number, ActiveTournament>();
  private sweepDone = false;

  async initStartupSweep(): Promise<void> {
    if (this.sweepDone) return;
    this.sweepDone = true;
    try {
      await markStaleTournamentsFailed();
    } catch (err) {
      console.error("[tournament] startup sweep failed:", err);
    }
  }

  getSnapshot(id: number): TournamentSnapshot | null {
    return this.runs.get(id)?.snapshot ?? null;
  }

  subscribe(
    id: number,
    listener: Subscriber
  ): { snapshot: TournamentSnapshot | null; unsubscribe: () => void } {
    const active = this.runs.get(id);
    if (!active) return { snapshot: null, unsubscribe: () => {} };
    active.subscribers.add(listener);
    return {
      snapshot: active.snapshot,
      unsubscribe: () => active.subscribers.delete(listener),
    };
  }

  abort(id: number): boolean {
    const active = this.runs.get(id);
    if (!active) return false;
    active.aborted = true;
    if (active.child) {
      try {
        active.child.kill("SIGKILL");
      } catch {
        // ignore
      }
    }
    return true;
  }

  async start(id: number): Promise<TournamentSnapshot | null> {
    // Reserve this id SYNCHRONOUSLY before any `await`. `start` is called both
    // by POST /api/tournaments and by the /stream route the client opens right
    // after; without a synchronous guard the two calls race past the existence
    // check (which used to sit behind awaits) and spawn two cutechess-cli
    // processes for the same tournament — doubling games onto one games.pgn.
    const existing = this.runs.get(id);
    if (existing) return existing.snapshot;

    const workDir = path.join(os.tmpdir(), "catgpt-tournaments", String(id));
    const pgnPath = path.join(workDir, PGN_FILENAME);

    const snapshot: TournamentSnapshot = {
      tournamentId: id,
      status: "pending",
      error: null,
      totalGames: 0,
      whiteLabel: "",
      blackLabel: "",
      scoreFirst: 0,
      scoreSecond: 0,
      scoreDraws: 0,
      games: [],
      liveGames: [],
      startedAt: null,
      finishedAt: null,
    };

    const active: ActiveTournament = {
      snapshot,
      subscribers: new Set(),
      child: null,
      aborted: false,
      finishedTimer: null,
      workDir,
      pgnPath,
      liveByNumber: new Map(),
      lastInfo: new Map(),
      pendingGo: new Map(),
      currentGameNumber: null,
      preamble: [],
      finishedCount: 0,
      queue: Promise.resolve(),
    };
    this.runs.set(id, active);

    try {
      await this.initStartupSweep();

      const row = await getTournamentRow(id);
      if (!row || row.status !== "pending") {
        this.runs.delete(id);
        return null;
      }

      snapshot.totalGames = row.totalGames;
      snapshot.whiteLabel = row.whiteLabel;
      snapshot.blackLabel = row.blackLabel;

      void this.execute(active, row.id).catch((err) => {
        console.error(`[tournament] run ${id} crashed:`, err);
      });

      return snapshot;
    } catch (err) {
      this.runs.delete(id);
      throw err;
    }
  }

  // ─── internals ───────────────────────────────────────────────────

  private emit(active: ActiveTournament, event: TournamentEvent) {
    for (const sub of active.subscribers) {
      try {
        sub(event);
      } catch (err) {
        console.error("[tournament] subscriber threw:", err);
      }
    }
  }

  private async execute(active: ActiveTournament, id: number) {
    const { snapshot } = active;
    try {
      const row = await getTournamentRow(id);
      if (!row) throw new Error("Tournament row vanished");

      await writeEnginesJson(active.workDir, [row.whiteConfig, row.blackConfig]);
      const args = buildCutechessArgs(active.workDir, {
        whiteConfig: row.whiteConfig,
        blackConfig: row.blackConfig,
        timeControl: row.timeControl,
        totalGames: row.totalGames,
        concurrency: row.concurrency,
        openingBook: row.openingBook,
        tbPath: row.tbPath,
        drawMoveNumber: row.drawMoveNumber,
        drawMoveCount: row.drawMoveCount,
        drawScoreCp: row.drawScoreCp,
      });

      await markTournamentRunning(id);
      snapshot.status = "running";
      snapshot.startedAt = new Date().toISOString();
      this.emit(active, {
        type: "run_started",
        data: {
          totalGames: row.totalGames,
          whiteLabel: row.whiteLabel,
          blackLabel: row.blackLabel,
        },
      });

      let child: ChildProcess;
      try {
        child = spawn(CUTECHESS_CLI_PATH, args, {
          cwd: active.workDir,
          env: process.env,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (err) {
        throw new Error(`Failed to spawn cutechess-cli: ${err}`);
      }
      active.child = child;

      this.wireStream(active, id, child);

      const exitCode: number = await new Promise((resolve) => {
        child.on("close", (code) => resolve(code ?? 0));
        child.on("error", () => resolve(-1));
      });

      // Drain the parse queue so all pending finalize/move writes complete.
      await active.queue;

      if (active.aborted) {
        snapshot.status = "cancelled";
        snapshot.error = "Aborted by user";
        snapshot.finishedAt = new Date().toISOString();
        await failTournament(id, "Aborted by user").catch(() => {});
        this.emit(active, { type: "error", data: { message: "Aborted by user" } });
        return;
      }

      if (exitCode < 0) {
        throw new Error("cutechess-cli failed to start (check CUTECHESS_CLI_PATH)");
      }

      const scores = {
        white: snapshot.scoreFirst,
        black: snapshot.scoreSecond,
        draw: snapshot.scoreDraws,
      };
      await completeTournament(id, scores);
      snapshot.status = "completed";
      snapshot.finishedAt = new Date().toISOString();
      this.emit(active, {
        type: "run_complete",
        data: {
          first: snapshot.scoreFirst,
          second: snapshot.scoreSecond,
          draws: snapshot.scoreDraws,
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      snapshot.status = "failed";
      snapshot.error = message;
      snapshot.finishedAt = new Date().toISOString();
      await failTournament(id, message).catch(() => {});
      this.emit(active, { type: "error", data: { message } });
    } finally {
      active.finishedTimer = setTimeout(() => {
        this.runs.delete(id);
      }, FINISHED_RETENTION_MS);
    }
  }

  private wireStream(active: ActiveTournament, id: number, child: ChildProcess) {
    const handle = (line: string) => {
      active.queue = active.queue
        .then(() => this.handleLine(active, id, line))
        .catch((err) => {
          console.error("[tournament] line handler error:", err);
        });
    };

    const attach = (stream: NodeJS.ReadableStream | null) => {
      if (!stream) return;
      let partial = "";
      stream.on("data", (chunk: Buffer) => {
        const text = partial + chunk.toString();
        const lines = text.split("\n");
        partial = lines.pop() || "";
        for (const l of lines) {
          const t = l.replace(/\r$/, "");
          if (t.length > 0) handle(t);
        }
      });
      stream.on("end", () => {
        if (partial.trim()) handle(partial);
        partial = "";
      });
    };

    attach(child.stdout);
    attach(child.stderr);
  }

  private async handleLine(active: ActiveTournament, id: number, line: string) {
    const { snapshot } = active;

    // 1. Debug (UCI I/O) lines — capture for logs + live reconstruction.
    const dbg = parseDebugLine(line);
    if (dbg) {
      // Attribute raw line to the current game log buffer.
      if (active.currentGameNumber != null) {
        const lg = active.liveByNumber.get(active.currentGameNumber);
        if (lg) lg.logLines.push(line);
      } else {
        active.preamble.push(line);
      }

      if (dbg.dir === "<") {
        // info line: track latest eval/depth for this engine.
        const info = parseUciInfo(dbg.payload);
        if (info) {
          active.lastInfo.set(dbg.engineName, info);
        }
        // bestmove line: apply to the matching live game.
        const bm = /^bestmove\s+(\S+)/.exec(dbg.payload);
        if (bm && bm[1] !== "(none)" && bm[1] !== "0000") {
          this.applyLiveMove(active, dbg.engineName, bm[1]);
        }
      } else {
        // GUI→engine `position` command: seed any opening-book plies cutechess
        // played internally (they never arrive as `bestmove` lines).
        const pos = parsePositionCommand(dbg.payload);
        if (pos) {
          this.applyPositionMoves(active, dbg.engineName, pos);
        }
        // GUI→engine `go wtime … btime …`: stash the clock state so the
        // ensuing `bestmove` can be tagged with the time each side had left.
        const clocks = parseGoClocks(dbg.payload);
        if (clocks) active.pendingGo.set(dbg.engineName, clocks);
      }
      return;
    }

    // 2. Started game.
    const started = parseStartedGame(line);
    if (started) {
      await this.onStarted(active, id, started);
      return;
    }

    // 3. Finished game.
    const finished = parseFinishedGame(line);
    if (finished) {
      await this.onFinished(active, finished);
      return;
    }

    // 4. Score update.
    const score = parseScoreLine(line);
    if (score) {
      snapshot.scoreFirst = score.first;
      snapshot.scoreSecond = score.second;
      snapshot.scoreDraws = score.draws;
      await updateTournamentScores(id, {
        white: score.first,
        black: score.second,
        draw: score.draws,
      }).catch(() => {});
      this.emit(active, { type: "score", data: score });
      return;
    }
  }

  private async onStarted(
    active: ActiveTournament,
    id: number,
    started: ReturnType<typeof parseStartedGame> & object
  ) {
    const openingFen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const gameId = await startTournamentGame({
      tournamentId: id,
      gameNumber: started.gameNumber,
      whiteEngine: started.white,
      blackEngine: started.black,
      openingFen,
    });

    const internal: InternalLiveGame = {
      gameId,
      gameNumber: started.gameNumber,
      white: started.white,
      black: started.black,
      openingFen,
      moves: [],
      tracker: new LiveGameTracker(openingFen),
      logLines: active.currentGameNumber == null ? [...active.preamble] : [],
    };
    active.preamble = [];
    active.liveByNumber.set(started.gameNumber, internal);
    active.currentGameNumber = started.gameNumber;

    this.refreshLiveSnapshot(active);
    this.emit(active, {
      type: "game_started",
      data: toLiveGame(internal),
    });
  }

  /**
   * Seed opening-book plies onto a live game from a UCI `position` command.
   *
   * cutechess plays the opening book itself and only hands the engine a
   * `position startpos moves <book…>` command before its first real search, so
   * those plies never appear as `bestmove` lines. We fast-forward the live
   * tracker to that move list and emit the missing moves so the viewer sees the
   * book played out on the board (instead of the engine's first move looking
   * illegal on an empty starting position).
   */
  private applyPositionMoves(
    active: ActiveTournament,
    engineName: string,
    pos: PositionCommand
  ) {
    // The engine receiving a `position` command is the side to move after the
    // listed plies; match the live game where that side is `engineName`.
    const baseSide: GameSide =
      pos.baseFen.split(" ")[1] === "b" ? "black" : "white";
    const toMove: GameSide =
      pos.moves.length % 2 === 0
        ? baseSide
        : baseSide === "white"
          ? "black"
          : "white";

    for (const lg of active.liveByNumber.values()) {
      const mover = toMove === "white" ? lg.white : lg.black;
      if (mover !== engineName) continue;
      const added = lg.tracker.syncToMoves(pos.baseFen, pos.moves);
      if (added.length === 0) return;
      // A custom base (e.g. EPD book) re-roots the tracker; reflect it live.
      if (lg.openingFen !== lg.tracker.openingFen) {
        lg.openingFen = lg.tracker.openingFen;
      }
      for (const move of added) {
        lg.moves.push(move);
        this.emit(active, {
          type: "move",
          data: { gameId: lg.gameId, gameNumber: lg.gameNumber, move },
        });
      }
      this.refreshLiveSnapshot(active);
      return;
    }
  }

  private applyLiveMove(
    active: ActiveTournament,
    engineName: string,
    uci: string
  ) {
    // Find the live game where `engineName` is the side to move.
    for (const lg of active.liveByNumber.values()) {
      const side: GameSide = lg.tracker.currentFen.split(" ")[1] === "w" ? "white" : "black";
      const mover = side === "white" ? lg.white : lg.black;
      if (mover !== engineName) continue;
      const info = active.lastInfo.get(engineName) ?? { evalCp: null, depth: null };
      const clocks = active.pendingGo.get(engineName) ?? null;
      active.pendingGo.delete(engineName);
      const move = lg.tracker.applyUci(uci, info.evalCp, info.depth, clocks);
      if (!move) return;
      lg.moves.push(move);
      this.refreshLiveSnapshot(active);
      this.emit(active, {
        type: "move",
        data: { gameId: lg.gameId, gameNumber: lg.gameNumber, move },
      });
      return;
    }
  }

  private async onFinished(
    active: ActiveTournament,
    finished: ReturnType<typeof parseFinishedGame> & object
  ) {
    const internal = active.liveByNumber.get(finished.gameNumber);
    const index = active.finishedCount;
    active.finishedCount += 1;

    const parsed = await readPgnGameAtIndex(active.pgnPath, index);

    let moves: GameMove[];
    let finalFen: string;
    let pgnText: string | null = null;
    let result: GameResult | null = scoreStrToResult(finished.scoreStr);

    if (parsed) {
      const recon = reconstructGame(parsed);
      moves = recon.moves;
      finalFen = recon.finalFen;
      pgnText = rebuildPgnText(parsed);
      if (parsed.result) {
        result = scoreStrToResult(parsed.result) ?? result;
      }
      // The PGN carries no clock data; overlay the per-ply clocks captured
      // live from cutechess's `go` commands (matched by ply).
      if (internal) {
        const clockByPly = new Map(
          internal.moves.map((m) => [m.ply, m])
        );
        for (const m of moves) {
          const live = clockByPly.get(m.ply);
          if (live) {
            m.whiteClockMs = live.whiteClockMs;
            m.blackClockMs = live.blackClockMs;
          }
        }
      }
    } else if (internal) {
      moves = internal.moves;
      finalFen = internal.tracker.currentFen;
    } else {
      moves = [];
      finalFen = "";
    }

    const logContent = internal ? internal.logLines.join("\n") : "";

    if (internal) {
      await finalizeTournamentGame({
        gameId: internal.gameId,
        result,
        termination: finished.reason,
        pgn: pgnText,
        finalFen,
        plyCount: moves.length,
        moves,
        uciLogs: logContent ? [{ engine: "combined", content: logContent }] : [],
      }).catch((err) => console.error("[tournament] finalize failed:", err));

      const summary: TournamentGameSummary = {
        id: internal.gameId,
        tournamentId: active.snapshot.tournamentId,
        gameNumber: internal.gameNumber,
        whiteEngine: internal.white,
        blackEngine: internal.black,
        status: "completed",
        result,
        termination: finished.reason,
        plyCount: moves.length,
        createdAt: new Date().toISOString(),
        finishedAt: new Date().toISOString(),
      };
      active.snapshot.games.push(summary);
      active.liveByNumber.delete(finished.gameNumber);
      if (active.currentGameNumber === finished.gameNumber) {
        active.currentGameNumber = null;
      }
      this.refreshLiveSnapshot(active);
      this.emit(active, { type: "game_finished", data: { game: summary } });
    }
  }

  private refreshLiveSnapshot(active: ActiveTournament) {
    active.snapshot.liveGames = [...active.liveByNumber.values()].map(toLiveGame);
  }
}

// ─── Helpers ──────────────────────────────────────────────────────

function toLiveGame(g: InternalLiveGame): LiveGame {
  return {
    gameId: g.gameId,
    gameNumber: g.gameNumber,
    white: g.white,
    black: g.black,
    openingFen: g.openingFen,
    moves: g.moves,
  };
}

/** Parse `score cp <n>` / `score mate <n>` and `depth <n>` from an info line. */
function parseUciInfo(
  payload: string
): { evalCp: number | null; depth: number | null } | null {
  if (!payload.startsWith("info")) return null;
  let evalCp: number | null = null;
  let depth: number | null = null;
  const cp = /\bscore cp (-?\d+)/.exec(payload);
  if (cp) evalCp = Number(cp[1]);
  const mate = /\bscore mate (-?\d+)/.exec(payload);
  if (mate) evalCp = (Number(mate[1]) >= 0 ? 1 : -1) * 100000;
  const d = /\bdepth (\d+)/.exec(payload);
  if (d) depth = Number(d[1]);
  if (evalCp === null && depth === null) return null;
  return { evalCp, depth };
}

/** Re-serialize a parsed game into a clean PGN string for storage. */
function rebuildPgnText(parsed: {
  headers: Record<string, string>;
  moves: { san: string; comment: string | null }[];
  result: string | null;
}): string {
  const headerLines = Object.entries(parsed.headers).map(
    ([k, v]) => `[${k} "${v}"]`
  );
  const tokens: string[] = [];
  let moveNo = 1;
  // Determine starting side from FEN header (default white).
  const fen = parsed.headers["FEN"];
  let whiteToMove = fen ? fen.split(" ")[1] !== "b" : true;
  for (const { san, comment } of parsed.moves) {
    if (whiteToMove) {
      tokens.push(`${moveNo}.`);
    } else if (tokens.length === 0) {
      tokens.push(`${moveNo}...`);
    }
    tokens.push(san);
    if (comment) tokens.push(`{${comment}}`);
    if (!whiteToMove) moveNo += 1;
    whiteToMove = !whiteToMove;
  }
  if (parsed.result) tokens.push(parsed.result);
  return `${headerLines.join("\n")}\n\n${tokens.join(" ")}`.trim();
}

// ─── Module-level singleton ───────────────────────────────────────

declare global {
  // eslint-disable-next-line no-var
  var __cutechessRunner: CutechessRunner | undefined;
}

export const cutechessRunner: CutechessRunner =
  globalThis.__cutechessRunner ??
  (globalThis.__cutechessRunner = new CutechessRunner());
