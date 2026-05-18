import { type NextRequest } from "next/server";
import { getBenchmarkRun } from "@/db/queries";
import { longBenchRunner, type RunnerEvent } from "@/lib/longbench-runner";

export const dynamic = "force-dynamic";

/**
 * GET /api/longbench/runs/[id]/stream
 *
 * Subscribe to a run's live events. Unlike the old /execute route, this
 * endpoint does NOT drive the executor — the executor runs in the process
 * singleton [longBenchRunner]. This makes the SSE stream safe to drop and
 * reopen: the run keeps going regardless.
 *
 * Events:
 *   snapshot         → full current state (sent exactly once, at connect).
 *                      Lets a fresh/reconnecting client rehydrate.
 *   position_started → { positionId, index }
 *   stats            → { positionId, nodes, cp, bestMove, correct }
 *   position_done    → { positionId, score, stableNodes, failed, ... }
 *   run_complete     → { aggregateScore }
 *   error            → { message, positionId? }
 *   ping             → {}   heartbeat, every 15s
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const runId = Number(id);
  if (!Number.isFinite(runId)) {
    return new Response(JSON.stringify({ error: "Invalid run id" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // If the run is still pending in the DB but not in memory (e.g. after a
  // server restart), try to (re)start the executor now.
  let snapshot = longBenchRunner.getSnapshot(runId);
  if (!snapshot) {
    const dbRun = await getBenchmarkRun(runId);
    if (!dbRun) {
      return new Response(JSON.stringify({ error: "Run not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (dbRun.status === "pending") {
      snapshot = await longBenchRunner.start(runId);
    }
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      function send(event: string, data: unknown) {
        if (closed) return;
        try {
          controller.enqueue(
            encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
          );
        } catch {
          closed = true;
        }
      }

      // 1. Send the snapshot (or a `finished` sentinel if there's nothing
      //    live in memory — the client will fall back to the DB detail).
      if (snapshot) {
        send("snapshot", snapshot);
      } else {
        send("snapshot", null);
        send("done", {});
        try {
          controller.close();
        } catch {
          // already closed
        }
        return;
      }

      // 2. Subscribe to live events.
      const { unsubscribe } = longBenchRunner.subscribe(
        runId,
        (event: RunnerEvent) => {
          send(event.type, event.data);
          if (event.type === "run_complete" || event.type === "error") {
            // Drain: let any immediate follow-ups flush, then close.
            setTimeout(() => {
              send("done", {});
              try {
                controller.close();
              } catch {
                // already closed
              }
            }, 50);
          }
        }
      );

      // 3. Heartbeat — keeps proxies from closing the idle connection.
      const heartbeat = setInterval(() => {
        send("ping", { t: Date.now() });
      }, 15_000);

      const cleanup = () => {
        closed = true;
        clearInterval(heartbeat);
        unsubscribe();
      };

      request.signal.addEventListener("abort", cleanup);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
