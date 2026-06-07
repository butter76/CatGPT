/**
 * cutechess helpers — engine resolution, engines.json generation, CLI argument
 * construction, and pure parsers for cutechess-cli stdout/stderr + PGN output.
 *
 * Server-only (uses fs/path + chess.js).
 */

import { mkdir, writeFile, readFile, chmod } from "fs/promises";
import { execSync } from "child_process";
import path from "path";
import { Chess, type Square as ChessSquare } from "chess.js";
import type { EngineConfig, GameMove, GameResult, GameSide } from "./types";

// ─── Binary / path resolution ─────────────────────────────────────

const HOME = process.env.HOME ?? "";

export const CUTECHESS_CLI_PATH =
  process.env.CUTECHESS_CLI_PATH || `${HOME}/cutechess/cutechess-cli`;

export const LKS_UCI_PATH =
  process.env.LKS_UCI_PATH || `${HOME}/CatGPT/cpp/build/bin/lks_uci`;

export const LKS_NETWORK_PATH =
  process.env.LKS_NETWORK_PATH ||
  process.env.CATGPT_ENGINE_PATH ||
  `${HOME}/CatGPT/S4.network`;

export const STOCKFISH_PATH =
  process.env.STOCKFISH_PATH || `${HOME}/Stockfish/src/stockfish`;

// Lc0 built with the ONNX→TensorRT backend (see build/onnxtrt). Distinct from
// LC0_PATH (the plain CUDA build used for position analysis).
export const LC0_TRT_PATH =
  process.env.LC0_TRT_PATH || `${HOME}/lc0/build/onnxtrt/lc0`;

export const LC0_WEIGHTS_PATH =
  process.env.LC0_WEIGHTS_PATH || `${HOME}/lc0/build/release/BT4-1740.pb.gz`;

// onnxruntime-gpu (TensorRT EP) + TensorRT + cuDNN + CUDA runtime libs. lc0
// cannot pick these up via UCI, so they must be exported for the engine
// process; we bake them into the launch command via `env`.
//
// /usr/local/lib64 is first because the onnx-trt binary was compiled with
// gcc 14 and needs its newer libstdc++ (CXXABI_1.3.15) — the system default
// (gcc 13) is too old and makes lc0 fail to start with a CXXABI error.
export const LC0_LD_LIBRARY_PATH =
  process.env.LC0_LD_LIBRARY_PATH ||
  [
    "/usr/local/lib64",
    `${HOME}/onnxruntime-linux-x64-gpu-1.23.2/lib`,
    `${HOME}/TensorRT-10.16.1.11/lib`,
    `${HOME}/CatGPT/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib`,
    "/usr/local/cuda/lib64",
  ].join(":");

export const SYZYGY_HOME =
  process.env.SYZYGY_HOME || process.env.LKS_SYZYGY_PATH || "";

function isExecutable(p: string): boolean {
  if (!p) return false;
  try {
    execSync(`test -x "${p}"`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function dirExists(p: string): boolean {
  if (!p) return false;
  try {
    execSync(`test -d "${p}"`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

export interface TournamentEnvAvailability {
  cutechess: boolean;
  catgpt: boolean;
  stockfish: boolean;
  lc0: boolean;
  syzygy: boolean;
  defaults: {
    catgptCommand: string;
    stockfishCommand: string;
    lc0Command: string;
    syzygyPath: string;
    cutechessPath: string;
  };
}

export function getTournamentEnvAvailability(): TournamentEnvAvailability {
  return {
    cutechess: isExecutable(CUTECHESS_CLI_PATH),
    catgpt: isExecutable(LKS_UCI_PATH),
    stockfish: isExecutable(STOCKFISH_PATH),
    lc0: isExecutable(LC0_TRT_PATH),
    syzygy: dirExists(SYZYGY_HOME),
    defaults: {
      catgptCommand: `${LKS_UCI_PATH} ${LKS_NETWORK_PATH}`,
      stockfishCommand: STOCKFISH_PATH,
      lc0Command: defaultLc0Config().command,
      syzygyPath: SYZYGY_HOME,
      cutechessPath: CUTECHESS_CLI_PATH,
    },
  };
}

// ─── Default engine configs (CatGPT lks_uci, Stockfish, Lc0) ──────

export function defaultCatgptConfig(): EngineConfig {
  return {
    name: "CatGPT",
    command: `${LKS_UCI_PATH} ${LKS_NETWORK_PATH}`,
    options: [],
    initStrings: [],
  };
}

export function defaultStockfishConfig(): EngineConfig {
  return {
    name: "Stockfish",
    command: STOCKFISH_PATH,
    options: [
      { name: "Threads", value: "8" },
      { name: "Hash", value: "8192" },
    ],
    initStrings: [],
  };
}

export function defaultLc0Config(): EngineConfig {
  // Everything goes in the launch command as CLI flags rather than UCI
  // options, for two reasons:
  //   1. LD_LIBRARY_PATH can't be set over UCI, so we wrap lc0 in `env`.
  //   2. `--backend-opts=gpu=0,fp16=true,batch=112` contains commas; the UI's
  //      option parser splits on commas, which would mangle it into three
  //      bogus options. Keeping it as a flag avoids that entirely.
  return {
    name: "Lc0",
    command:
      `env LD_LIBRARY_PATH=${LC0_LD_LIBRARY_PATH} ${LC0_TRT_PATH} ` +
      `--backend=onnx-trt --weights=${LC0_WEIGHTS_PATH} ` +
      `--minibatch-size=112 --backend-opts=gpu=0,fp16=true,batch=112`,
    options: [],
    initStrings: [],
  };
}

export function defaultEngineConfigs(): {
  catgpt: EngineConfig;
  stockfish: EngineConfig;
  lc0: EngineConfig;
} {
  return {
    catgpt: defaultCatgptConfig(),
    stockfish: defaultStockfishConfig(),
    lc0: defaultLc0Config(),
  };
}

// ─── engines.json generation ──────────────────────────────────────

interface CutechessEngineEntry {
  name: string;
  command: string;
  protocol: "uci";
  workingDirectory?: string;
  options?: { name: string; value: string }[];
  initStrings?: string[];
}

function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_") || "engine";
}

/**
 * Write an engines.json (the cutechess "engines.jsons") into `dir`. The two
 * configs MUST have distinct `name`s; those names are referenced via
 * `-engine conf=<name>` in the CLI args.
 *
 * cutechess launches the engine `command` as a single executable (it does not
 * split arguments, and engines.json has no `arguments` field). To support
 * commands like `lks_uci /path/to.network`, we write a tiny wrapper script per
 * engine that `exec`s the full command, and point `command` at the wrapper.
 */
export async function writeEnginesJson(
  dir: string,
  configs: EngineConfig[]
): Promise<string> {
  await mkdir(dir, { recursive: true });

  const entries: CutechessEngineEntry[] = [];
  for (const cfg of configs) {
    const wrapperPath = path.join(dir, `engine_${sanitizeName(cfg.name)}.sh`);
    const script = `#!/bin/sh\nexec ${cfg.command} "$@"\n`;
    await writeFile(wrapperPath, script, { mode: 0o755 });
    await chmod(wrapperPath, 0o755);

    const entry: CutechessEngineEntry = {
      name: cfg.name,
      command: wrapperPath,
      protocol: "uci",
    };
    if (cfg.options && cfg.options.length > 0) entry.options = cfg.options;
    if (cfg.initStrings && cfg.initStrings.length > 0)
      entry.initStrings = cfg.initStrings;
    entries.push(entry);
  }

  const filePath = path.join(dir, "engines.json");
  await writeFile(filePath, JSON.stringify(entries, null, 2), "utf-8");
  return filePath;
}

// ─── CLI argument construction ────────────────────────────────────

export interface CutechessRunConfig {
  whiteConfig: EngineConfig;
  blackConfig: EngineConfig;
  /** cutechess tc string, e.g. "900+5". */
  timeControl: string;
  totalGames: number;
  concurrency: number;
  openingBook: string | null;
  tbPath: string | null;
  drawMoveNumber: number;
  drawMoveCount: number;
  drawScoreCp: number;
}

export const PGN_FILENAME = "games.pgn";

export function buildCutechessArgs(
  workDir: string,
  cfg: CutechessRunConfig
): string[] {
  // Per-engine time control: cutechess applies `-each` after per-engine
  // options, so the tc must live inside each `-engine` list (not `-each`) for
  // per-side overrides to take effect.
  const whiteTc = cfg.whiteConfig.timeControl?.trim() || cfg.timeControl;
  const blackTc = cfg.blackConfig.timeControl?.trim() || cfg.timeControl;
  const args: string[] = [
    "-engine",
    `conf=${cfg.whiteConfig.name}`,
    `tc=${whiteTc}`,
    "-engine",
    `conf=${cfg.blackConfig.name}`,
    `tc=${blackTc}`,
    "-games",
    String(cfg.totalGames),
    "-rounds",
    "1",
    "-concurrency",
    String(Math.max(1, cfg.concurrency)),
    "-ratinginterval",
    "1",
    "-recover",
    // Play each opening twice with swapped sides so both engines get to
    // play it as white and as black (see cutechess `-repeat`). Without this,
    // cutechess draws a fresh opening for every game (openingRepetitions=1).
    "-repeat",
    "2",
  ];

  if (cfg.openingBook) {
    const format = cfg.openingBook.toLowerCase().endsWith(".epd")
      ? "epd"
      : "pgn";
    args.push(
      "-openings",
      `file=${cfg.openingBook}`,
      `format=${format}`,
      "order=random"
    );
  }

  if (cfg.tbPath) {
    args.push("-tb", cfg.tbPath);
  }

  // Draw adjudication: both engines' |score| <= drawScoreCp for drawMoveCount
  // consecutive moves, starting from move drawMoveNumber.
  args.push(
    "-draw",
    `movenumber=${cfg.drawMoveNumber}`,
    `movecount=${cfg.drawMoveCount}`,
    `score=${cfg.drawScoreCp}`
  );

  args.push("-pgnout", path.join(workDir, PGN_FILENAME));
  args.push("-debug");

  return args;
}

// ─── Output line parsers ──────────────────────────────────────────

export interface DebugLine {
  ms: number;
  /** ">" = GUI→engine, "<" = engine→GUI. */
  dir: ">" | "<";
  engineName: string;
  index: number;
  payload: string;
}

const DEBUG_RE = /^(\d+)\s+([<>])([^(]+)\((\d+)\):\s?(.*)$/;

export function parseDebugLine(line: string): DebugLine | null {
  const m = DEBUG_RE.exec(line);
  if (!m) return null;
  return {
    ms: Number(m[1]),
    dir: m[2] as ">" | "<",
    engineName: m[3].trim(),
    index: Number(m[4]),
    payload: m[5],
  };
}

export interface GoClocks {
  whiteClockMs: number | null;
  blackClockMs: number | null;
}

/**
 * Parse the clock state from a UCI `go` command, e.g.
 * `go wtime 900000 btime 894200 winc 5000 binc 5000`. cutechess sends the
 * authoritative remaining time (including its own lag accounting) to each
 * engine before every search. Returns null if the payload isn't a clock-based
 * `go` (e.g. fixed depth/nodes/movetime).
 */
export function parseGoClocks(payload: string): GoClocks | null {
  if (!/^go\b/.test(payload.trim())) return null;
  const wt = /\bwtime\s+(-?\d+)/.exec(payload);
  const bt = /\bbtime\s+(-?\d+)/.exec(payload);
  if (!wt && !bt) return null;
  return {
    whiteClockMs: wt ? Number(wt[1]) : null,
    blackClockMs: bt ? Number(bt[1]) : null,
  };
}

export interface StartedGame {
  gameNumber: number;
  total: number;
  white: string;
  black: string;
}

const STARTED_RE = /^Started game (\d+) of (\d+) \((.+) vs (.+)\)\s*$/;

export function parseStartedGame(line: string): StartedGame | null {
  const m = STARTED_RE.exec(line);
  if (!m) return null;
  return {
    gameNumber: Number(m[1]),
    total: Number(m[2]),
    white: m[3].trim(),
    black: m[4].trim(),
  };
}

export interface FinishedGame {
  gameNumber: number;
  scoreStr: string;
  reason: string;
}

const FINISHED_RE = /^Finished game (\d+) \(.+\): (\S+)\s*\{(.+)\}\s*$/;

export function parseFinishedGame(line: string): FinishedGame | null {
  const m = FINISHED_RE.exec(line);
  if (!m) return null;
  return {
    gameNumber: Number(m[1]),
    scoreStr: m[2],
    reason: m[3].trim(),
  };
}

export interface ScoreLine {
  /** Wins for the first engine (== whiteLabel / whiteConfig). */
  first: number;
  /** Wins for the second engine (== blackLabel / blackConfig). */
  second: number;
  draws: number;
}

const SCORE_RE = /^Score of .+:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)\s/;

export function parseScoreLine(line: string): ScoreLine | null {
  const m = SCORE_RE.exec(line);
  if (!m) return null;
  return { first: Number(m[1]), second: Number(m[2]), draws: Number(m[3]) };
}

export function scoreStrToResult(scoreStr: string): GameResult | null {
  if (scoreStr === "1-0") return "white_win";
  if (scoreStr === "0-1") return "black_win";
  if (scoreStr === "1/2-1/2") return "draw";
  return null;
}

// ─── PGN parsing ──────────────────────────────────────────────────

const STANDARD_START =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export interface ParsedPgnGame {
  headers: Record<string, string>;
  /** Movetext entries: SAN token + its trailing {comment} (if any). */
  moves: { san: string; comment: string | null }[];
  result: string | null;
}

/** Split a multi-game PGN file into individual game blocks. */
export function splitPgnGames(pgn: string): string[] {
  const games: string[] = [];
  const lines = pgn.split(/\r?\n/);
  let current: string[] = [];
  let inMoves = false;
  for (const line of lines) {
    const isHeader = line.startsWith("[");
    // A new game begins when we see a header line after having collected
    // movetext for the previous game.
    if (isHeader && inMoves && current.some((l) => l.trim())) {
      games.push(current.join("\n").trim());
      current = [];
      inMoves = false;
    }
    if (!isHeader && line.trim()) inMoves = true;
    current.push(line);
  }
  if (current.some((l) => l.trim())) games.push(current.join("\n").trim());
  return games.filter((g) => g.length > 0);
}

export function parsePgnGame(block: string): ParsedPgnGame {
  const headers: Record<string, string> = {};
  const movetextLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    const hm = /^\[(\w+)\s+"(.*)"\]\s*$/.exec(line.trim());
    if (hm) {
      headers[hm[1]] = hm[2];
    } else if (line.trim()) {
      movetextLines.push(line);
    }
  }

  const movetext = movetextLines.join(" ");
  const moves: { san: string; comment: string | null }[] = [];
  let result: string | null = null;

  let i = 0;
  const n = movetext.length;
  let pendingSan: string | null = null;
  const flush = (comment: string | null) => {
    if (pendingSan) {
      moves.push({ san: pendingSan, comment });
      pendingSan = null;
    }
  };

  while (i < n) {
    const ch = movetext[i];
    if (ch === "{") {
      const end = movetext.indexOf("}", i);
      const comment = end === -1 ? movetext.slice(i + 1) : movetext.slice(i + 1, end);
      flush(comment.trim());
      i = end === -1 ? n : end + 1;
      continue;
    }
    if (ch === "(") {
      // Skip recursive variations entirely (cutechess doesn't emit these,
      // but be defensive).
      let depth = 1;
      i++;
      while (i < n && depth > 0) {
        if (movetext[i] === "(") depth++;
        else if (movetext[i] === ")") depth--;
        i++;
      }
      continue;
    }
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    // Read a token up to whitespace / brace / paren.
    let j = i;
    while (j < n && !/[\s{}()]/.test(movetext[j])) j++;
    const token = movetext.slice(i, j);
    i = j;

    if (token === "1-0" || token === "0-1" || token === "1/2-1/2" || token === "*") {
      flush(null);
      result = token;
      continue;
    }
    // Move number tokens like "1." or "1..." or "12.".
    if (/^\d+\.+$/.test(token)) continue;
    // Strip a leading move number glued to a move, e.g. "1.e4".
    const stripped = token.replace(/^\d+\.+/, "");
    if (!stripped) continue;
    // A new move token: flush the previous (no comment seen).
    flush(null);
    pendingSan = stripped;
  }
  flush(null);

  return { headers, moves, result };
}

export interface ParsedMoveComment {
  evalCp: number | null;
  depth: number | null;
  timeMs: number | null;
}

const COMMENT_RE = /([+-]?(?:M\d+|\d+(?:\.\d+)?))\/(\d+)\s+([\d.]+)s/;

export function parseMoveComment(comment: string | null): ParsedMoveComment {
  if (!comment) return { evalCp: null, depth: null, timeMs: null };
  const m = COMMENT_RE.exec(comment);
  if (!m) return { evalCp: null, depth: null, timeMs: null };

  const evalToken = m[1];
  let evalCp: number | null = null;
  const mateMatch = /^([+-]?)M(\d+)$/.exec(evalToken);
  if (mateMatch) {
    const sign = mateMatch[1] === "-" ? -1 : 1;
    evalCp = sign * 100000;
  } else {
    const pawns = Number(evalToken);
    if (Number.isFinite(pawns)) evalCp = Math.round(pawns * 100);
  }

  const depth = Number(m[2]);
  const timeMs = Math.round(Number(m[3]) * 1000);
  return {
    evalCp,
    depth: Number.isFinite(depth) ? depth : null,
    timeMs: Number.isFinite(timeMs) ? timeMs : null,
  };
}

export interface ReconstructedGame {
  openingFen: string;
  finalFen: string;
  moves: GameMove[];
}

/**
 * Replay the parsed SAN movetext from the (optional) opening FEN to derive
 * per-ply UCI, FEN, mover side, and attach the eval/depth/time from comments.
 */
export function reconstructGame(parsed: ParsedPgnGame): ReconstructedGame {
  const openingFen = parsed.headers["FEN"] || STANDARD_START;
  const moves: GameMove[] = [];
  let finalFen = openingFen;

  try {
    const chess = new Chess(openingFen);
    let ply = 0;
    for (const { san, comment } of parsed.moves) {
      const mover: GameSide = chess.turn() === "w" ? "white" : "black";
      let move;
      try {
        move = chess.move(san);
      } catch {
        break;
      }
      if (!move) break;
      ply += 1;
      const { evalCp, depth, timeMs } = parseMoveComment(comment);
      moves.push({
        ply,
        mover,
        san: move.san,
        uci: move.from + move.to + (move.promotion ?? ""),
        fenAfter: chess.fen(),
        evalCp,
        depth,
        timeMs,
        whiteClockMs: null,
        blackClockMs: null,
      });
      finalFen = chess.fen();
    }
  } catch {
    // openingFen invalid — return what we have.
  }

  return { openingFen, finalFen, moves };
}

/** Read + parse the game at `index` (0-based) from a cutechess PGN file. */
export async function readPgnGameAtIndex(
  pgnPath: string,
  index: number
): Promise<ParsedPgnGame | null> {
  let raw: string;
  try {
    raw = await readFile(pgnPath, "utf-8");
  } catch {
    return null;
  }
  const blocks = splitPgnGames(raw);
  if (index < 0 || index >= blocks.length) return null;
  return parsePgnGame(blocks[index]);
}

// ─── Live move reconstruction (from the debug stream) ──────────────

export interface PositionCommand {
  /** Resolved base FEN (`startpos` → the standard start position). */
  baseFen: string;
  /** UCI moves applied on top of `baseFen`. */
  moves: string[];
}

/**
 * Parse a UCI `position` command (GUI→engine). cutechess sends the engine the
 * full move list — including any opening-book plies it played internally —
 * right before each search, so this is how we recover the book moves that never
 * appear as `bestmove` lines.
 */
export function parsePositionCommand(payload: string): PositionCommand | null {
  const trimmed = payload.trim();
  if (!trimmed.startsWith("position")) return null;
  const rest = trimmed.slice("position".length).trim();

  let baseFen: string;
  let remainder: string;
  if (rest.startsWith("startpos")) {
    baseFen = STANDARD_START;
    remainder = rest.slice("startpos".length).trim();
  } else if (rest.startsWith("fen")) {
    const afterFen = rest.slice("fen".length).trim();
    const movesIdx = afterFen.indexOf("moves");
    if (movesIdx === -1) {
      baseFen = afterFen.trim();
      remainder = "";
    } else {
      baseFen = afterFen.slice(0, movesIdx).trim();
      remainder = afterFen.slice(movesIdx).trim();
    }
  } else {
    return null;
  }

  let moves: string[] = [];
  if (remainder.startsWith("moves")) {
    moves = remainder
      .slice("moves".length)
      .trim()
      .split(/\s+/)
      .filter(Boolean);
  }
  return { baseFen, moves };
}

/**
 * Incrementally reconstructs a game from UCI moves observed on the debug
 * stream, so the UI can show the board updating move-by-move before the PGN
 * is written. One instance per in-progress game.
 */
export class LiveGameTracker {
  private chess: Chess;
  private ply = 0;
  public openingFen: string;

  constructor(openingFen: string) {
    this.openingFen = openingFen;
    this.chess = new Chess(openingFen || STANDARD_START);
  }

  get currentFen(): string {
    return this.chess.fen();
  }

  /** Number of plies applied so far. */
  get plyCount(): number {
    return this.ply;
  }

  /**
   * Fast-forward to the move list of a UCI `position` command, applying any
   * plies not yet seen. This is how opening-book moves — played by cutechess
   * before either engine searches, so they never arrive as `bestmove` lines —
   * get onto the live board. Returns only the newly applied moves (book plies
   * have no eval/depth).
   *
   * If nothing has been applied yet and `baseFen` differs from the opening FEN
   * (e.g. an EPD opening book starting from a custom position), the tracker
   * re-roots to `baseFen`.
   */
  syncToMoves(baseFen: string, uciMoves: string[]): GameMove[] {
    if (this.ply === 0 && baseFen && baseFen !== this.openingFen) {
      try {
        this.chess = new Chess(baseFen);
        this.openingFen = baseFen;
      } catch {
        return [];
      }
    }
    const added: GameMove[] = [];
    for (let i = this.ply; i < uciMoves.length; i++) {
      const move = this.applyUci(uciMoves[i], null, null);
      if (!move) break;
      added.push(move);
    }
    return added;
  }

  /**
   * Apply a UCI move. Returns the derived move (or null if illegal). `evalCp`
   * / `depth` come from the engine's most recent `info` line; `clocks` come
   * from the `go` command that triggered this move (clock state before it).
   */
  applyUci(
    uci: string,
    evalCp: number | null,
    depth: number | null,
    clocks: GoClocks | null = null
  ): GameMove | null {
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promotion = uci.length > 4 ? uci[4] : undefined;
    const mover: GameSide = this.chess.turn() === "w" ? "white" : "black";
    let move;
    try {
      move = this.chess.move({
        from: from as ChessSquare,
        to: to as ChessSquare,
        promotion: promotion as "q" | "r" | "b" | "n" | undefined,
      });
    } catch {
      return null;
    }
    if (!move) return null;
    this.ply += 1;
    return {
      ply: this.ply,
      mover,
      san: move.san,
      uci: move.from + move.to + (move.promotion ?? ""),
      fenAfter: this.chess.fen(),
      evalCp,
      depth,
      timeMs: null,
      whiteClockMs: clocks?.whiteClockMs ?? null,
      blackClockMs: clocks?.blackClockMs ?? null,
    };
  }
}
