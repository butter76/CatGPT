import { NextResponse } from "next/server";
import { getPositionById, deletePosition } from "@/db/queries";

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
