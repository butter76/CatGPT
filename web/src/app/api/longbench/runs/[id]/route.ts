import { NextResponse } from "next/server";
import { getBenchmarkRun } from "@/db/queries";

// GET /api/longbench/runs/[id] — full run detail
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const runId = Number(id);
    if (!Number.isFinite(runId)) {
      return NextResponse.json({ error: "Invalid run id" }, { status: 400 });
    }
    const run = await getBenchmarkRun(runId);
    if (!run) {
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }
    return NextResponse.json(run);
  } catch (error) {
    console.error("Failed to fetch benchmark run:", error);
    return NextResponse.json(
      { error: "Failed to fetch benchmark run" },
      { status: 500 }
    );
  }
}
