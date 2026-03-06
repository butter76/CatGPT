import { eq, desc, inArray } from "drizzle-orm";
import { db } from "./index";
import {
  positions,
  moveAnnotations,
  networkAnalyses,
  policyEntries,
  engineAnalyses,
} from "./schema";
import type { Position, SharpMoveAnnotation, NetworkAnalysis, PolicyEntry, EngineAnalysis, EngineInfoLine, CatGPTSearchStats } from "@/lib/types";

// ─── Helpers to assemble full Position objects ────────────────────

function generateId(): string {
  return `pos-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ─── Read ─────────────────────────────────────────────────────────

export async function getAllPositions(): Promise<Position[]> {
  const rows = await db.select().from(positions).orderBy(desc(positions.createdAt));

  // Batch-fetch all related data
  const posIds = rows.map((r) => r.id);
  if (posIds.length === 0) return [];

  const [annRows, naRows, eaRows] = await Promise.all([
    db.select().from(moveAnnotations).where(inArray(moveAnnotations.positionId, posIds)),
    db
      .select()
      .from(networkAnalyses)
      .where(inArray(networkAnalyses.positionId, posIds))
      .orderBy(desc(networkAnalyses.createdAt)),
    db
      .select()
      .from(engineAnalyses)
      .where(inArray(engineAnalyses.positionId, posIds))
      .orderBy(desc(engineAnalyses.createdAt)),
  ]);

  // Fetch policy entries for all network analyses
  const naIds = naRows.map((r) => r.id);
  const peRows =
    naIds.length > 0
      ? await db
          .select()
          .from(policyEntries)
          .where(inArray(policyEntries.analysisId, naIds))
      : [];

  // Group by position
  const annByPos = groupBy(annRows, (r) => r.positionId);
  const naByPos = groupBy(naRows, (r) => r.positionId);
  const eaByPos = groupBy(eaRows, (r) => r.positionId);
  const peByAnalysis = groupBy(peRows, (r) => r.analysisId.toString());

  return rows.map((row) => assemblePosition(row, annByPos, naByPos, eaByPos, peByAnalysis));
}

export async function getPositionById(id: string): Promise<Position | null> {
  const [row] = await db.select().from(positions).where(eq(positions.id, id));
  if (!row) return null;

  const [annRows, naRows, eaRows] = await Promise.all([
    db.select().from(moveAnnotations).where(eq(moveAnnotations.positionId, id)),
    db
      .select()
      .from(networkAnalyses)
      .where(eq(networkAnalyses.positionId, id))
      .orderBy(desc(networkAnalyses.createdAt)),
    db
      .select()
      .from(engineAnalyses)
      .where(eq(engineAnalyses.positionId, id))
      .orderBy(desc(engineAnalyses.createdAt)),
  ]);

  const naIds = naRows.map((r) => r.id);
  const peRows =
    naIds.length > 0
      ? await db
          .select()
          .from(policyEntries)
          .where(inArray(policyEntries.analysisId, naIds))
      : [];

  const annByPos = { [id]: annRows };
  const naByPos = { [id]: naRows };
  const eaByPos = { [id]: eaRows };
  const peByAnalysis = groupBy(peRows, (r) => r.analysisId.toString());

  return assemblePosition(row, annByPos, naByPos, eaByPos, peByAnalysis);
}

// ─── Write ────────────────────────────────────────────────────────

export async function createPosition(
  data: Pick<Position, "name" | "description" | "type" | "fen" | "expectedOutcome" | "moveAnnotations">
): Promise<Position> {
  const id = generateId();
  const now = new Date();

  await db.insert(positions).values({
    id,
    name: data.name,
    description: data.description ?? null,
    type: data.type,
    fen: data.fen,
    expectedOutcome: data.expectedOutcome ?? null,
    createdAt: now,
    updatedAt: now,
  });

  // Insert move annotations if SHARP
  if (data.type === "SHARP" && data.moveAnnotations && data.moveAnnotations.length > 0) {
    await db.insert(moveAnnotations).values(
      data.moveAnnotations.map((a) => ({
        positionId: id,
        move: a.move,
        annotation: a.annotation,
      }))
    );
  }

  return (await getPositionById(id))!;
}

export async function setMoveAnnotations(
  positionId: string,
  annotations: { move: string; annotation: "correct" | "blunder" }[]
): Promise<void> {
  // Delete existing annotations
  await db.delete(moveAnnotations).where(eq(moveAnnotations.positionId, positionId));

  // Insert new ones
  if (annotations.length > 0) {
    await db.insert(moveAnnotations).values(
      annotations.map((a) => ({
        positionId,
        move: a.move,
        annotation: a.annotation,
      }))
    );
  }

  // Update the position's updatedAt
  await db
    .update(positions)
    .set({ updatedAt: new Date() })
    .where(eq(positions.id, positionId));
}

export async function updatePositionMeta(
  id: string,
  updates: { name?: string; description?: string | null; blunderTag?: "catgpt" | "stockfish" | "leela" | null }
): Promise<void> {
  await db
    .update(positions)
    .set({ ...updates, updatedAt: new Date() })
    .where(eq(positions.id, id));
}

export async function deletePosition(id: string): Promise<boolean> {
  const result = await db.delete(positions).where(eq(positions.id, id));
  // Cascading deletes handle related rows
  return true;
}

// ─── Analysis ─────────────────────────────────────────────────────

export async function createNetworkAnalysis(
  positionId: string,
  data: Omit<NetworkAnalysis, "timestamp">
): Promise<NetworkAnalysis> {
  const [na] = await db
    .insert(networkAnalyses)
    .values({
      positionId,
      bestQ: data.bestQ,
      wdlWin: data.wdl.win,
      wdlDraw: data.wdl.draw,
      wdlLoss: data.wdl.loss,
      nodes: data.nodes,
    })
    .returning();

  if (data.policy.length > 0) {
    await db.insert(policyEntries).values(
      data.policy.map((p) => ({
        analysisId: na.id,
        move: p.move,
        probability: p.probability,
      }))
    );
  }

  // Update the position's updatedAt
  await db
    .update(positions)
    .set({ updatedAt: new Date() })
    .where(eq(positions.id, positionId));

  return {
    policy: data.policy,
    wdl: data.wdl,
    bestQ: data.bestQ,
    nodes: data.nodes,
    timestamp: na.createdAt.toISOString(),
  };
}

// ─── Engine Analysis ──────────────────────────────────────────────

export async function createEngineAnalysisRecord(
  positionId: string,
  data: {
    engine: "leela" | "stockfish" | "catgpt";
    bestMove: string;
    evaluation: number;
    depth: number;
    nodes: number;
    pv: string[];
    depthHistory?: EngineInfoLine[];
    catgptHistory?: CatGPTSearchStats[];
  }
): Promise<void> {
  // For CatGPT, store catgptHistory in the depthHistory jsonb column
  const historyPayload = data.engine === "catgpt"
    ? (data.catgptHistory ?? [])
    : (data.depthHistory ?? []);

  await db.insert(engineAnalyses).values({
    positionId,
    engine: data.engine,
    bestMove: data.bestMove,
    evaluation: data.evaluation,
    depth: data.depth,
    nodes: data.nodes,
    pv: data.pv,
    depthHistory: historyPayload,
  });

  // Update the position's updatedAt
  await db
    .update(positions)
    .set({ updatedAt: new Date() })
    .where(eq(positions.id, positionId));
}

export async function deleteEngineAnalysis(id: number): Promise<void> {
  await db.delete(engineAnalyses).where(eq(engineAnalyses.id, id));
}

// ─── Assembly helpers ─────────────────────────────────────────────

function assemblePosition(
  row: typeof positions.$inferSelect,
  annByPos: Record<string, (typeof moveAnnotations.$inferSelect)[]>,
  naByPos: Record<string, (typeof networkAnalyses.$inferSelect)[]>,
  eaByPos: Record<string, (typeof engineAnalyses.$inferSelect)[]>,
  peByAnalysis: Record<string, (typeof policyEntries.$inferSelect)[]>
): Position {
  const anns = annByPos[row.id] ?? [];
  const nas = naByPos[row.id] ?? [];
  const eas = eaByPos[row.id] ?? [];

  // Take the most recent network analysis
  const latestNA = nas[0];
  let networkAnalysis: NetworkAnalysis | undefined;
  if (latestNA) {
    const policies = peByAnalysis[latestNA.id.toString()] ?? [];
    networkAnalysis = {
      policy: policies
        .map((p): PolicyEntry => ({ move: p.move, probability: p.probability }))
        .sort((a, b) => b.probability - a.probability),
      wdl: { win: latestNA.wdlWin, draw: latestNA.wdlDraw, loss: latestNA.wdlLoss },
      bestQ: latestNA.bestQ,
      nodes: latestNA.nodes,
      timestamp: latestNA.createdAt.toISOString(),
    };
  }

  const engineAnalysesData: EngineAnalysis[] = eas.map((ea) => {
    const base = {
      id: ea.id,
      engine: ea.engine,
      bestMove: ea.bestMove,
      evaluation: ea.evaluation,
      depth: ea.depth,
      nodes: ea.nodes,
      pv: ea.pv ?? [],
      timestamp: ea.createdAt.toISOString(),
    };
    if (ea.engine === "catgpt") {
      return {
        ...base,
        catgptHistory: (ea.depthHistory as unknown as CatGPTSearchStats[]) ?? [],
      };
    }
    return {
      ...base,
      depthHistory: (ea.depthHistory as EngineInfoLine[]) ?? [],
    };
  });

  const moveAnnotationsData: SharpMoveAnnotation[] = anns.map((a) => ({
    move: a.move,
    annotation: a.annotation,
  }));

  return {
    id: row.id,
    name: row.name,
    description: row.description ?? undefined,
    type: row.type,
    fen: row.fen,
    expectedOutcome: row.expectedOutcome ?? undefined,
    blunderTag: row.blunderTag ?? undefined,
    moveAnnotations: moveAnnotationsData.length > 0 ? moveAnnotationsData : undefined,
    networkAnalysis,
    engineAnalyses: engineAnalysesData.length > 0 ? engineAnalysesData : undefined,
    createdAt: row.createdAt.toISOString(),
    updatedAt: row.updatedAt.toISOString(),
  };
}

function groupBy<T>(arr: T[], keyFn: (item: T) => string): Record<string, T[]> {
  const map: Record<string, T[]> = {};
  for (const item of arr) {
    const key = keyFn(item);
    (map[key] ??= []).push(item);
  }
  return map;
}
