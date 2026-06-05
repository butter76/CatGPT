/**
 * CatGPT Engine Runner — spawns the catgpt_search binary (LKS-backed)
 * and parses its output.
 *
 * Unlike the UCI engines (Stockfish/lc0), catgpt_search uses a simpler
 * protocol:
 *   - JSON lines to stdout (search stats: root_eval, search_update,
 *     search_complete) emitted by cpp/src/lks_search_main.cpp
 *   - Final "bestmove <uci>" line
 *   - Stderr for loading/error messages (ignored by the web UI)
 *
 * Output events are yielded as they arrive for SSE streaming.
 */

import { spawn, type ChildProcess } from "child_process";

// ─── CatGPT binary paths ─────────────────────────────────────────

const CATGPT_BINARY =
  process.env.CATGPT_SEARCH_PATH ||
  `${process.env.HOME}/CatGPT/cpp/build/bin/catgpt_search`;

const CATGPT_ENGINE =
  process.env.CATGPT_ENGINE_PATH ||
  `${process.env.HOME}/CatGPT/S4.network`;

/** Check if catgpt_search binary is available */
export function isCatGPTAvailable(): boolean {
  try {
    const { execSync } = require("child_process");
    execSync(`test -x "${CATGPT_BINARY}"`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

// ─── CatGPT stats types ──────────────────────────────────────────

export interface CatGPTSearchPolicyEntry {
  move: string;
  /** Softmax-normalized exp(log_alloc) over all legal moves. */
  weight: number;
  /** Q from parent-STM perspective [-1, 1]. Only present for expanded children. */
  q?: number;
}

export interface CatGPTSearchStats {
  type: "root_eval" | "search_update" | "search_complete";
  bestMove: string;
  cp: number;
  nodes: number;
  /** Centi-depth (round(LksSearch::max_depth() * 100)). */
  iteration: number;
  policy: CatGPTSearchPolicyEntry[];
  /** Greedy best-child PV walk; absent only when the root has no TT entry yet. */
  pv?: string[];
}

// ─── CatGPT event types ──────────────────────────────────────────

export type CatGPTEvent =
  | { type: "stats"; data: CatGPTSearchStats }
  | { type: "bestmove"; data: { bestMove: string } }
  | { type: "error"; data: { message: string } }
  | { type: "done"; data: Record<string, never> };

// ─── Run CatGPT analysis ─────────────────────────────────────────

export interface CatGPTAnalysisRequest {
  fen: string;
  nodes: number;
  /**
   * Called once the child process is spawned. Lets the caller capture the
   * handle so it can kill the process for external cancellation. Only called
   * on successful spawn.
   */
  onSpawn?: (child: ChildProcess) => void;
}

/**
 * Run CatGPT (LKS) search on a position.
 * Yields events as they arrive for SSE streaming.
 */
export async function* runCatGPTAnalysis(
  request: CatGPTAnalysisRequest
): AsyncGenerator<CatGPTEvent> {
  let proc: ChildProcess;
  try {
    const args = [CATGPT_ENGINE, request.fen, request.nodes.toString()];
    proc = spawn(CATGPT_BINARY, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });
    request.onSpawn?.(proc);
  } catch (err) {
    yield {
      type: "error",
      data: { message: `Failed to spawn catgpt_search: ${err}` },
    };
    return;
  }

  // Line-buffered stdout reading
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
    // TRT loading messages go to stderr — ignore
  });

  proc.on("close", () => {
    done = true;
    if (partial.trim()) pushLine(partial.trim());
    pushLine(""); // sentinel
  });

  proc.on("error", (err) => {
    done = true;
    pushLine(`__error__ ${err.message}`);
    pushLine(""); // sentinel
  });

  try {
    while (!done) {
      const line = await Promise.race([
        nextLine(),
        new Promise<string>((resolve) =>
          setTimeout(() => resolve("__timeout__"), 1800000)
        ),
      ]);

      if (line === "__timeout__" || line === "") break;

      if (line.startsWith("__error__")) {
        yield {
          type: "error",
          data: { message: line.slice(10) },
        };
        break;
      }

      // Parse bestmove line
      if (line.startsWith("bestmove")) {
        const tokens = line.split(/\s+/);
        yield { type: "bestmove", data: { bestMove: tokens[1] } };
        break;
      }

      // Try to parse JSON stats line
      try {
        const parsed = JSON.parse(line) as CatGPTSearchStats;
        if (
          parsed.type === "root_eval" ||
          parsed.type === "search_update" ||
          parsed.type === "search_complete"
        ) {
          yield { type: "stats", data: parsed };
        }
      } catch {
        // Not JSON — skip
      }
    }

    yield { type: "done", data: {} };
  } catch (err) {
    yield {
      type: "error",
      data: { message: err instanceof Error ? err.message : String(err) },
    };
  } finally {
    try {
      proc.kill("SIGKILL");
    } catch {
      // ignore
    }
  }
}
