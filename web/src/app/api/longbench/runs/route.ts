import { NextRequest, NextResponse } from "next/server";
import { createBenchmarkRun, listBenchmarkRuns } from "@/db/queries";
import { longBenchRunner } from "@/lib/longbench-runner";

const DEFAULT_MAX_NODES = 1_000_000;

function defaultEngineLabel(): string {
  return (
    process.env.CATGPT_ENGINE_PATH ||
    (process.env.HOME ? `${process.env.HOME}/CatGPT/main.trt` : "main.trt")
  );
}

// GET /api/longbench/runs — list all benchmark runs
export async function GET() {
  try {
    const runs = await listBenchmarkRuns();
    return NextResponse.json(runs);
  } catch (error) {
    console.error("Failed to list benchmark runs:", error);
    return NextResponse.json(
      { error: "Failed to list benchmark runs" },
      { status: 500 }
    );
  }
}

// POST /api/longbench/runs — create a new pending run
export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));

    const engine: string =
      typeof body.engine === "string" && body.engine.trim()
        ? body.engine.trim()
        : defaultEngineLabel();

    const maxNodesRaw = body.maxNodes;
    const maxNodes =
      typeof maxNodesRaw === "number" && Number.isFinite(maxNodesRaw) && maxNodesRaw >= 1
        ? Math.floor(maxNodesRaw)
        : DEFAULT_MAX_NODES;

    const run = await createBenchmarkRun(engine, maxNodes);
    // Kick off the executor immediately so the run is independent of the
    // creating HTTP request. The client will connect via /stream afterward.
    void longBenchRunner.start(run.id);
    return NextResponse.json(run, { status: 201 });
  } catch (error) {
    console.error("Failed to create benchmark run:", error);
    return NextResponse.json(
      { error: "Failed to create benchmark run" },
      { status: 500 }
    );
  }
}
