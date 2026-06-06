import { NextResponse } from "next/server";
import { failTournament, getTournamentRow } from "@/db/queries";
import { cutechessRunner } from "@/lib/cutechess-runner";

// POST /api/tournaments/[id]/abort — cancel a running or pending tournament.
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const tournamentId = Number(id);
  if (!Number.isFinite(tournamentId)) {
    return NextResponse.json({ error: "Invalid tournament id" }, { status: 400 });
  }

  const aborted = cutechessRunner.abort(tournamentId);
  if (!aborted) {
    const row = await getTournamentRow(tournamentId);
    if (!row) {
      return NextResponse.json({ error: "Tournament not found" }, { status: 404 });
    }
    if (row.status === "pending" || row.status === "running") {
      await failTournament(tournamentId, "Aborted by user");
    }
  }
  return NextResponse.json({ ok: true });
}
