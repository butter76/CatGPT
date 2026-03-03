import { NextRequest, NextResponse } from "next/server";
import { getPositionById, setMoveAnnotations } from "@/db/queries";

// PUT /api/positions/[id]/annotations — replace all move annotations
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const position = await getPositionById(id);
    if (!position) {
      return NextResponse.json({ error: "Position not found" }, { status: 404 });
    }

    const body = await request.json();
    const annotations: { move: string; annotation: "correct" | "blunder" }[] =
      body.annotations ?? [];

    // Validate
    for (const a of annotations) {
      if (!a.move || !["correct", "blunder"].includes(a.annotation)) {
        return NextResponse.json(
          { error: "Each annotation must have a move and annotation (correct|blunder)" },
          { status: 400 }
        );
      }
    }

    await setMoveAnnotations(id, annotations);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to update annotations:", error);
    return NextResponse.json(
      { error: "Failed to update annotations" },
      { status: 500 }
    );
  }
}
