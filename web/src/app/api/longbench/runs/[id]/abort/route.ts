import { NextResponse } from "next/server";
import { failBenchmarkRun, getBenchmarkRun } from "@/db/queries";
import { longBenchRunner } from "@/lib/longbench-runner";

// POST /api/longbench/runs/[id]/abort — cancel a running or pending run.
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const runId = Number(id);
  if (!Number.isFinite(runId)) {
    return NextResponse.json({ error: "Invalid run id" }, { status: 400 });
  }

  const aborted = longBenchRunner.abort(runId);
  if (!aborted) {
    // The runner has no in-memory state for this run; mark the DB row failed
    // so it doesn't stay stuck in `pending`/`running`.
    const dbRun = await getBenchmarkRun(runId);
    if (!dbRun) {
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }
    if (dbRun.status === "pending" || dbRun.status === "running") {
      await failBenchmarkRun(runId, "Aborted by user");
    }
  }
  return NextResponse.json({ ok: true });
}
