import { NextResponse } from "next/server";
import { getAvailableEngines } from "@/lib/uci-engine";

// GET /api/analyze/engines — list available engines on this system
export async function GET() {
  const engines = getAvailableEngines();
  return NextResponse.json({ engines });
}
