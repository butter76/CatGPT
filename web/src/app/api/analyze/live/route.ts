import { type NextRequest } from "next/server";
import { runEngineAnalysis } from "@/lib/uci-engine";
import { isValidFEN } from "@/lib/chess-utils";
import type { EngineInfoLine } from "@/lib/types";

/**
 * GET /api/analyze/live?fen=...&engine=stockfish&nodes=500000&positionId=...
 *
 * Server-Sent Events endpoint that streams UCI engine analysis in real time.
 *
 * Events:
 *   - "info"     → EngineInfoLine (one per depth)
 *   - "bestmove" → { bestMove, ponder? }
 *   - "error"    → { message }
 *   - "done"     → {}
 */
export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const fen = searchParams.get("fen");
  const engine = searchParams.get("engine") ?? "stockfish";
  const nodes = parseInt(searchParams.get("nodes") ?? "500000", 10);
  const positionId = searchParams.get("positionId"); // optional — for DB persistence

  // Validate
  if (!fen || !isValidFEN(fen)) {
    return new Response(JSON.stringify({ error: "Invalid or missing FEN" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (!["stockfish", "leela"].includes(engine)) {
    return new Response(JSON.stringify({ error: "engine must be stockfish or leela" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (isNaN(nodes) || nodes < 1 || nodes > 100_000_000) {
    return new Response(JSON.stringify({ error: "nodes must be 1–100000000" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Create SSE stream
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      function send(event: string, data: unknown) {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
        );
      }

      try {
        // Collect all info lines for depth history
        const depthHistory: EngineInfoLine[] = [];
        let lastInfo: EngineInfoLine | null = null;

        for await (const event of runEngineAnalysis({
          engine: engine as "stockfish" | "leela",
          fen,
          nodes,
        })) {
          switch (event.type) {
            case "info": {
              const info = event.data as EngineInfoLine;
              depthHistory.push(info);
              lastInfo = info;
              send("info", event.data);
              break;
            }
            case "bestmove":
              send("bestmove", event.data);
              // If positionId provided, persist the final result with full depth history
              if (positionId && lastInfo) {
                try {
                  const { createEngineAnalysisRecord } = await import("@/db/queries");
                  const bm = event.data as { bestMove: string };
                  await createEngineAnalysisRecord(positionId, {
                    engine: engine as "leela" | "stockfish",
                    bestMove: bm.bestMove,
                    evaluation: lastInfo.score?.value ?? 0,
                    depth: lastInfo.depth ?? 0,
                    nodes: lastInfo.nodes ?? nodes,
                    pv: lastInfo.pv ?? [bm.bestMove],
                    depthHistory,
                  });
                  send("saved", { positionId });
                } catch (err) {
                  send("error", {
                    message: `Analysis complete but failed to save: ${err}`,
                  });
                }
              }
              break;
            case "error":
              send("error", event.data);
              break;
            case "done":
              send("done", {});
              break;
          }
        }
      } catch (err) {
        send("error", {
          message: err instanceof Error ? err.message : String(err),
        });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
