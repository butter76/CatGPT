import { NextRequest, NextResponse } from "next/server";
import { getAllPositions, createPosition } from "@/db/queries";

// GET /api/positions — list all positions
export async function GET() {
  try {
    const positions = await getAllPositions();
    return NextResponse.json(positions);
  } catch (error) {
    console.error("Failed to fetch positions:", error);
    return NextResponse.json(
      { error: "Failed to fetch positions" },
      { status: 500 }
    );
  }
}

// POST /api/positions — create a new position
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.name || !body.fen || !body.type) {
      return NextResponse.json(
        { error: "Missing required fields: name, fen, type" },
        { status: 400 }
      );
    }

    if (!["SHARP", "FORTRESS"].includes(body.type)) {
      return NextResponse.json(
        { error: "type must be SHARP or FORTRESS" },
        { status: 400 }
      );
    }

    const position = await createPosition({
      name: body.name,
      description: body.description,
      type: body.type,
      fen: body.fen,
      expectedOutcome: body.expectedOutcome,
      moveAnnotations: body.moveAnnotations,
    });

    return NextResponse.json(position, { status: 201 });
  } catch (error) {
    console.error("Failed to create position:", error);
    return NextResponse.json(
      { error: "Failed to create position" },
      { status: 500 }
    );
  }
}
