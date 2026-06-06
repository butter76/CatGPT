import { NextResponse } from "next/server";
import { getTournamentDetail } from "@/db/queries";

// GET /api/tournaments/[id] — tournament + games summary
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const tournamentId = Number(id);
  if (!Number.isFinite(tournamentId)) {
    return NextResponse.json({ error: "Invalid tournament id" }, { status: 400 });
  }
  const detail = await getTournamentDetail(tournamentId);
  if (!detail) {
    return NextResponse.json({ error: "Tournament not found" }, { status: 404 });
  }
  return NextResponse.json(detail);
}
