import { create } from "zustand";
import type { Position, PositionType, Outcome, NotationFormat, BlunderTag } from "./types";

interface PositionStore {
  // Settings (client-only)
  notationFormat: NotationFormat;
  setNotationFormat: (format: NotationFormat) => void;
}

/** Lightweight store — only for client-side settings (notation format, etc.). */
export const usePositionStore = create<PositionStore>((set) => ({
  notationFormat: "algebraic",
  setNotationFormat: (format) => set({ notationFormat: format }),
}));

// ─── API client functions ─────────────────────────────────────────

export async function fetchPositions(): Promise<Position[]> {
  const res = await fetch("/api/positions");
  if (!res.ok) throw new Error("Failed to fetch positions");
  return res.json();
}

export async function fetchPosition(id: string): Promise<Position | null> {
  const res = await fetch(`/api/positions/${id}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error("Failed to fetch position");
  return res.json();
}

export async function createPosition(
  data: Pick<Position, "name" | "description" | "type" | "fen" | "expectedOutcome" | "moveAnnotations">
): Promise<Position> {
  const res = await fetch("/api/positions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to create position");
  return res.json();
}

export async function deletePositionAPI(id: string): Promise<void> {
  const res = await fetch(`/api/positions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete position");
}

export async function updatePositionMetaAPI(
  id: string,
  updates: {
    name?: string;
    description?: string | null;
    blunderTag?: BlunderTag | null;
    type?: PositionType;
    expectedOutcome?: Outcome | null;
    longBench?: boolean;
  }
): Promise<Position> {
  const res = await fetch(`/api/positions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!res.ok) throw new Error("Failed to update position");
  return res.json();
}

export async function updateAnnotationsAPI(
  positionId: string,
  annotations: { move: string; annotation: "correct" | "blunder" }[]
): Promise<void> {
  const res = await fetch(`/api/positions/${positionId}/annotations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ annotations }),
  });
  if (!res.ok) throw new Error("Failed to update annotations");
}

export async function deleteEngineAnalysisAPI(id: number): Promise<void> {
  const res = await fetch(`/api/engine-analyses/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete engine analysis");
}

export async function requestAnalysis(
  positionId: string,
  data: { policy: { move: string; probability: number }[]; wdl: { win: number; draw: number; loss: number }; bestQ: number; nodes: number }
) {
  const res = await fetch(`/api/positions/${positionId}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to store analysis");
  return res.json();
}
