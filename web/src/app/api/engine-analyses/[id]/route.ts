import { NextResponse } from "next/server";
import { deleteEngineAnalysis } from "@/db/queries";

// DELETE /api/engine-analyses/[id]
export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    await deleteEngineAnalysis(parseInt(id, 10));
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Failed to delete engine analysis:", error);
    return NextResponse.json(
      { error: "Failed to delete engine analysis" },
      { status: 500 }
    );
  }
}
