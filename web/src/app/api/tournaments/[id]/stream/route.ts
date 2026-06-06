import { type NextRequest } from "next/server";
import { getTournamentRow } from "@/db/queries";
import { cutechessRunner, type TournamentEvent } from "@/lib/cutechess-runner";

export const dynamic = "force-dynamic";

/**
 * GET /api/tournaments/[id]/stream
 *
 * Subscribe to a tournament's live events. The match runs in the process
 * singleton [cutechessRunner], so this SSE stream is safe to drop and reopen.
 *
 * Events: snapshot, run_started, game_started, move, game_finished, score,
 * run_complete, error, ping, done.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const tournamentId = Number(id);
  if (!Number.isFinite(tournamentId)) {
    return new Response(JSON.stringify({ error: "Invalid tournament id" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  let snapshot = cutechessRunner.getSnapshot(tournamentId);
  if (!snapshot) {
    const row = await getTournamentRow(tournamentId);
    if (!row) {
      return new Response(JSON.stringify({ error: "Tournament not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }
    if (row.status === "pending") {
      snapshot = await cutechessRunner.start(tournamentId);
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

      const { unsubscribe } = cutechessRunner.subscribe(
        tournamentId,
        (event: TournamentEvent) => {
          send(event.type, event.data);
          if (event.type === "run_complete" || event.type === "error") {
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
