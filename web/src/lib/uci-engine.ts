/**
 * UCI Engine Manager — spawns and communicates with UCI chess engines.
 * Server-only module (uses child_process).
 */

import { spawn, type ChildProcess } from "child_process";
import path from "path";
import type { EngineInfoLine } from "./types";

// ─── Engine binary paths ──────────────────────────────────────────

const ENGINE_PATHS: Record<string, string> = {
  stockfish: process.env.STOCKFISH_PATH || `${process.env.HOME}/Stockfish/src/stockfish`,
  leela: process.env.LC0_PATH || "/usr/local/bin/lc0",
};

/** Check which engines are available on this system */
export function getAvailableEngines(): string[] {
  const { execSync } = require("child_process");
  const available: string[] = [];
  for (const [name, binaryPath] of Object.entries(ENGINE_PATHS)) {
    try {
      execSync(`test -x "${binaryPath}"`, { stdio: "ignore" });
      available.push(name);
    } catch {
      // not available
    }
  }
  // CatGPT (LKS-backed `catgpt_search` binary).
  const { isCatGPTAvailable } = require("./catgpt-engine");
  if (isCatGPTAvailable()) {
    available.push("catgpt");
  }
  return available;
}

// ─── Info line parser ─────────────────────────────────────────────

/**
 * Parse a UCI `info` line into a structured object.
 * Example: "info depth 12 seldepth 18 multipv 1 score cp 35 nodes 12345 nps 1234567 time 10 pv e2e4 e7e5 g1f3"
 */
export function parseInfoLine(line: string): EngineInfoLine | null {
  if (!line.startsWith("info depth")) return null;
  // Skip currmove/currmovenumber lines (partial search updates without score/pv)
  if (line.includes("currmove")) return null;

  const tokens = line.split(/\s+/);
  const result: EngineInfoLine = {
    depth: 0,
    score: { type: "cp", value: 0 },
    nodes: 0,
    pv: [],
  };

  let i = 1; // skip "info"
  while (i < tokens.length) {
    switch (tokens[i]) {
      case "depth":
        result.depth = parseInt(tokens[++i]);
        break;
      case "seldepth":
        result.seldepth = parseInt(tokens[++i]);
        break;
      case "multipv":
        result.multipv = parseInt(tokens[++i]);
        break;
      case "score":
        i++;
        if (tokens[i] === "cp") {
          result.score = { type: "cp", value: parseInt(tokens[++i]) };
        } else if (tokens[i] === "mate") {
          result.score = { type: "mate", value: parseInt(tokens[++i]) };
        }
        break;
      case "wdl":
        result.wdl = {
          win: parseInt(tokens[++i]),
          draw: parseInt(tokens[++i]),
          loss: parseInt(tokens[++i]),
        };
        break;
      case "nodes":
        result.nodes = parseInt(tokens[++i]);
        break;
      case "nps":
        result.nps = parseInt(tokens[++i]);
        break;
      case "time":
        result.time = parseInt(tokens[++i]);
        break;
      case "pv":
        // Everything after "pv" is the principal variation
        result.pv = tokens.slice(i + 1);
        i = tokens.length; // break the loop
        break;
      // Skip: hashfull, tbhits, upperbound, lowerbound, string, etc.
      default:
        break;
    }
    i++;
  }

  // Only return if we got a valid depth
  if (result.depth > 0) return result;
  return null;
}

/**
 * Parse a `bestmove` line.
 * Example: "bestmove e2e4 ponder e7e5"
 */
export function parseBestMove(line: string): { bestMove: string; ponder?: string } | null {
  if (!line.startsWith("bestmove")) return null;
  const tokens = line.split(/\s+/);
  return {
    bestMove: tokens[1],
    ponder: tokens[3], // may be undefined
  };
}

// ─── Engine session ───────────────────────────────────────────────

export interface EngineAnalysisRequest {
  engine: "stockfish" | "leela";
  fen: string;
  nodes: number;
  /** Optional UCI options to set, e.g. { Threads: "4", Hash: "256" } */
  options?: Record<string, string>;
}

export interface EngineEvent {
  type: "info" | "bestmove" | "error" | "done";
  data: EngineInfoLine | { bestMove: string; ponder?: string } | { message: string } | Record<string, never>;
}

/**
 * Run a UCI engine analysis.
 * Returns an async generator that yields events as they happen.
 */
export async function* runEngineAnalysis(
  request: EngineAnalysisRequest
): AsyncGenerator<EngineEvent> {
  const enginePath = ENGINE_PATHS[request.engine];
  if (!enginePath) {
    yield { type: "error", data: { message: `Unknown engine: ${request.engine}` } };
    return;
  }

  // For lc0, set the cwd to the binary's directory so that WeightsFile
  // autodiscovery finds the .pb file sitting next to the binary.
  const engineDir = path.dirname(enginePath);

  let proc: ChildProcess;
  try {
    proc = spawn(enginePath, [], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: engineDir,
    });
  } catch (err) {
    yield { type: "error", data: { message: `Failed to spawn engine: ${err}` } };
    return;
  }

  // Set up line buffering on stdout
  const lineBuffer: string[] = [];
  let resolveNextLine: ((line: string) => void) | null = null;
  let done = false;

  function pushLine(line: string) {
    if (resolveNextLine) {
      const resolve = resolveNextLine;
      resolveNextLine = null;
      resolve(line);
    } else {
      lineBuffer.push(line);
    }
  }

  function nextLine(): Promise<string> {
    if (lineBuffer.length > 0) {
      return Promise.resolve(lineBuffer.shift()!);
    }
    if (done) return Promise.resolve("");
    return new Promise((resolve) => {
      resolveNextLine = resolve;
    });
  }

  let partial = "";
  proc.stdout!.on("data", (chunk: Buffer) => {
    const text = partial + chunk.toString();
    const lines = text.split("\n");
    partial = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) pushLine(line.trim());
    }
  });

  proc.stderr!.on("data", () => {
    // lc0 outputs GPU/backend info to stderr; ignore
  });

  proc.on("close", () => {
    done = true;
    if (partial.trim()) pushLine(partial.trim());
    pushLine(""); // sentinel
  });

  proc.on("error", (err) => {
    done = true;
    pushLine(`error ${err.message}`);
    pushLine(""); // sentinel
  });

  // Helper to send a command
  function send(cmd: string) {
    proc.stdin!.write(cmd + "\n");
  }

  // Helper to wait for a specific response
  async function waitFor(prefix: string, timeoutMs = 10000): Promise<string> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      const line = await Promise.race([
        nextLine(),
        new Promise<string>((resolve) => setTimeout(() => resolve("__timeout__"), remaining)),
      ]);
      if (line === "__timeout__" || line === "") break;
      if (line.startsWith(prefix)) return line;
    }
    throw new Error(`Timeout waiting for "${prefix}"`);
  }

  try {
    // UCI handshake
    send("uci");
    await waitFor("uciok", 15000);

    // Set options if provided
    if (request.options) {
      for (const [key, value] of Object.entries(request.options)) {
        send(`setoption name ${key} value ${value}`);
      }
    }

    // For lc0, enable WDL display
    if (request.engine === "leela") {
      send("setoption name UCI_ShowWDL value true");
    }

    // For Stockfish, use 8 threads and 8 GB of hash by default.
    if (request.engine === "stockfish") {
      send("setoption name Threads value 8");
      send("setoption name Hash value 8192");
    }

    send("isready");
    // lc0 first-time readyok can take a while (CUDA init, weight loading)
    await waitFor("readyok", 30000);

    // Set position
    send(`position fen ${request.fen}`);

    // Go!
    send(`go nodes ${request.nodes}`);

    // Read lines until bestmove
    while (!done) {
      const line = await Promise.race([
        nextLine(),
        new Promise<string>((resolve) =>
          setTimeout(() => resolve("__timeout__"), 1800000)
        ),
      ]);

      if (line === "__timeout__" || line === "") break;

      // Parse info lines
      const info = parseInfoLine(line);
      if (info) {
        yield { type: "info", data: info };
        continue;
      }

      // Parse bestmove
      const bm = parseBestMove(line);
      if (bm) {
        yield { type: "bestmove", data: bm };
        break;
      }
    }

    yield { type: "done", data: {} };
  } catch (err) {
    yield {
      type: "error",
      data: { message: err instanceof Error ? err.message : String(err) },
    };
  } finally {
    // Clean up
    try {
      send("quit");
    } catch {
      // ignore
    }
    setTimeout(() => {
      try {
        proc.kill("SIGKILL");
      } catch {
        // ignore
      }
    }, 2000);
  }
}
