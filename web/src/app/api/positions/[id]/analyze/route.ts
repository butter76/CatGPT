import { NextResponse } from "next/server";
import { getPositionById, createNetworkAnalysis } from "@/db/queries";
import type { NetworkAnalysis } from "@/lib/types";

// POST /api/positions/[id]/analyze — store a network analysis result
export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const position = await getPositionById(id);
    if (!position) {
      return NextResponse.json({ error: "Position not found" }, { status: 404 });
    }

    const body = await request.json();

    // Validate
    if (body.policy == null || body.wdl == null || body.bestQ == null) {
      return NextResponse.json(
        { error: "Missing required fields: policy, wdl, bestQ" },
        { status: 400 }
      );
    }

    const analysis = await createNetworkAnalysis(id, {
      policy: body.policy,
      wdl: body.wdl,
      bestQ: body.bestQ,
      nodes: body.nodes ?? 1,
    });

    return NextResponse.json(analysis, { status: 201 });
  } catch (error) {
    console.error("Failed to store analysis:", error);
    return NextResponse.json(
      { error: "Failed to store analysis" },
      { status: 500 }
    );
  }
}
