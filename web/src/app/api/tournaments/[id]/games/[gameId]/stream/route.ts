import { type NextRequest } from "next/server";
import { cutechessRunner, type TournamentEvent } from "@/lib/cutechess-runner";

export const dynamic = "force-dynamic";

/**
 * GET /api/tournaments/[id]/games/[gameId]/stream
 *
 * Subscribe to live moves for a single in-progress game. Sends a `moves`
 * snapshot of what's been played so far, then streams subsequent `move`
 * events. Closes when the game finishes (or there's no live state).
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string; gameId: string }> }
) {
  const { id, gameId } = await params;
  const tournamentId = Number(id);
  const gid = Number(gameId);
  if (!Number.isFinite(tournamentId) || !Number.isFinite(gid)) {
    return new Response(JSON.stringify({ error: "Invalid id" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const snapshot = cutechessRunner.getSnapshot(tournamentId);
  const liveGame = snapshot?.liveGames.find((g) => g.gameId === gid) ?? null;

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

      // No live state — the game is finished/not running; client uses the DB.
      if (!snapshot || !liveGame) {
        send("moves", null);
        send("done", {});
        try {
          controller.close();
        } catch {
          // already closed
        }
        return;
      }

      send("moves", { gameId: gid, moves: liveGame.moves });

      const { unsubscribe } = cutechessRunner.subscribe(
        tournamentId,
        (event: TournamentEvent) => {
          if (event.type === "move" && event.data.gameId === gid) {
            send("move", event.data.move);
          } else if (
            event.type === "game_finished" &&
            event.data.game.id === gid
          ) {
            setTimeout(() => {
              send("done", {});
              try {
                controller.close();
              } catch {
                // already closed
              }
            }, 50);
          } else if (event.type === "run_complete" || event.type === "error") {
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
