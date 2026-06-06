import { NextResponse } from "next/server";
import { getTournamentGameDetail } from "@/db/queries";

// GET /api/tournaments/[id]/games/[gameId] — full game (moves, pgn, uci logs)
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string; gameId: string }> }
) {
  const { id, gameId } = await params;
  const tournamentId = Number(id);
  const gid = Number(gameId);
  if (!Number.isFinite(tournamentId) || !Number.isFinite(gid)) {
    return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  }
  const detail = await getTournamentGameDetail(tournamentId, gid);
  if (!detail) {
    return NextResponse.json({ error: "Game not found" }, { status: 404 });
  }
  return NextResponse.json(detail);
}
