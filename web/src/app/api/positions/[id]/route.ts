import { NextRequest, NextResponse } from "next/server";
import { getPositionById, deletePosition, updatePositionMeta } from "@/db/queries";

// GET /api/positions/[id] — get a single position
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const position = await getPositionById(id);
    if (!position) {
      return NextResponse.json({ error: "Position not found" }, { status: 404 });
    }
    return NextResponse.json(position);
  } catch (error) {
    console.error("Failed to fetch position:", error);
    return NextResponse.json(
      { error: "Failed to fetch position" },
      { status: 500 }
    );
  }
}

// PATCH /api/positions/[id] — update name/description
export async function PATCH(
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
    const updates: {
      name?: string;
      description?: string | null;
      blunderTag?: "catgpt" | "stockfish" | "leela" | null;
      type?: "SHARP" | "FORTRESS";
      expectedOutcome?: "win" | "loss" | "draw" | "unknown" | null;
    } = {};

    if (typeof body.name === "string" && body.name.trim()) {
      updates.name = body.name.trim();
    }
    if (body.description !== undefined) {
      updates.description = body.description?.trim() || null;
    }
    if (body.blunderTag !== undefined) {
      const validTags = ["catgpt", "stockfish", "leela", null];
      if (validTags.includes(body.blunderTag)) {
        updates.blunderTag = body.blunderTag;
      }
    }
    if (body.type !== undefined) {
      const validTypes = ["SHARP", "FORTRESS"];
      if (validTypes.includes(body.type)) {
        updates.type = body.type;
      }
    }
    if (body.expectedOutcome !== undefined) {
      const validOutcomes = ["win", "loss", "draw", "unknown", null];
      if (validOutcomes.includes(body.expectedOutcome)) {
        updates.expectedOutcome = body.expectedOutcome;
      }
    }

    if (Object.keys(updates).length === 0) {
      return NextResponse.json({ error: "No valid fields to update" }, { status: 400 });
    }

    await updatePositionMeta(id, updates);
    const updated = await getPositionById(id);
    return NextResponse.json(updated);
  } catch (error) {
    console.error("Failed to update position:", error);
    return NextResponse.json(
      { error: "Failed to update position" },
      { status: 500 }
    );
  }
}

// DELETE /api/positions/[id] — delete a position
export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    await deletePosition(id);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete position:", error);
    return NextResponse.json(
      { error: "Failed to delete position" },
      { status: 500 }
    );
  }
}
