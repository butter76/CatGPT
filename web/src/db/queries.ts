import { eq, desc, inArray, and } from "drizzle-orm";
import { db } from "./index";
import {
  positions,
  moveAnnotations,
  networkAnalyses,
  policyEntries,
  engineAnalyses,
  benchmarkRuns,
  benchmarkPositionResults,
} from "./schema";
import type {
  Position,
  SharpMoveAnnotation,
  NetworkAnalysis,
  PolicyEntry,
  EngineAnalysis,
  EngineInfoLine,
  CatGPTSearchStats,
  BenchmarkRun,
  BenchmarkPositionResult,
  BenchmarkRunDetail,
  BenchmarkRunStatus,
  BenchmarkStatsSample,
} from "@/lib/types";

// ─── Helpers to assemble full Position objects ────────────────────

function generateId(): string {
  return `pos-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ─── Read ─────────────────────────────────────────────────────────

/**
 * Lightweight list query for the /positions page.
 * Skips engine analyses and policy entries (the heaviest payloads) since the
 * list view only needs basic metadata, move-annotation counts, and a boolean
 * flag for whether a network analysis exists.
 */
export async function getAllPositionsSummary(): Promise<Position[]> {
  const rows = await db.select().from(positions).orderBy(desc(positions.createdAt));

  const posIds = rows.map((r) => r.id);
  if (posIds.length === 0) return [];

  const [annRows, naRows] = await Promise.all([
    db.select().from(moveAnnotations).where(inArray(moveAnnotations.positionId, posIds)),
    db
      .select({
        id: networkAnalyses.id,
        positionId: networkAnalyses.positionId,
        bestQ: networkAnalyses.bestQ,
        wdlWin: networkAnalyses.wdlWin,
        wdlDraw: networkAnalyses.wdlDraw,
        wdlLoss: networkAnalyses.wdlLoss,
        nodes: networkAnalyses.nodes,
        createdAt: networkAnalyses.createdAt,
      })
      .from(networkAnalyses)
      .where(inArray(networkAnalyses.positionId, posIds))
      .orderBy(desc(networkAnalyses.createdAt)),
  ]);

  const annByPos = groupBy(annRows, (r) => r.positionId);
  const naByPos = groupBy(naRows, (r) => r.positionId);

  return rows.map((row) => {
    const anns = annByPos[row.id] ?? [];
    const nas = naByPos[row.id] ?? [];
    const latestNA = nas[0];

    return {
      id: row.id,
      name: row.name,
      description: row.description ?? undefined,
      type: row.type,
      fen: row.fen,
      expectedOutcome: row.expectedOutcome ?? undefined,
      blunderTag: row.blunderTag ?? undefined,
      longBench: row.longBench,
      moveAnnotations: anns.length > 0
        ? anns.map((a) => ({ move: a.move, annotation: a.annotation }))
        : undefined,
      networkAnalysis: latestNA
        ? {
            policy: [],
            wdl: { win: latestNA.wdlWin, draw: latestNA.wdlDraw, loss: latestNA.wdlLoss },
            bestQ: latestNA.bestQ,
            nodes: latestNA.nodes,
            timestamp: latestNA.createdAt.toISOString(),
          }
        : undefined,
      createdAt: row.createdAt.toISOString(),
      updatedAt: row.updatedAt.toISOString(),
    };
  });
}

/** Full query with all related data — used by getPositionById and createPosition. */
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
  updates: {
    name?: string;
    description?: string | null;
    blunderTag?: "catgpt" | "stockfish" | "leela" | null;
    type?: "SHARP" | "FORTRESS";
    expectedOutcome?: "win" | "loss" | "draw" | "unknown" | null;
    longBench?: boolean;
  }
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
    engine: "leela" | "stockfish" | "catgpt" | "catgpt_mcts";
    bestMove: string;
    evaluation: number;
    depth: number;
    nodes: number;
    pv: string[];
    depthHistory?: EngineInfoLine[];
    catgptHistory?: CatGPTSearchStats[];
  }
): Promise<void> {
  const isCatGPT = data.engine === "catgpt" || data.engine === "catgpt_mcts";
  const historyPayload = isCatGPT
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
    if (ea.engine === "catgpt" || ea.engine === "catgpt_mcts") {
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
    longBench: row.longBench,
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

// ─── LongBench ────────────────────────────────────────────────────

/** Return all positions flagged as LongBench, fully assembled. */
export async function getLongBenchPositions(): Promise<Position[]> {
  const rows = await db
    .select()
    .from(positions)
    .where(eq(positions.longBench, true))
    .orderBy(desc(positions.createdAt));

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

  const naIds = naRows.map((r) => r.id);
  const peRows =
    naIds.length > 0
      ? await db
          .select()
          .from(policyEntries)
          .where(inArray(policyEntries.analysisId, naIds))
      : [];

  const annByPos = groupBy(annRows, (r) => r.positionId);
  const naByPos = groupBy(naRows, (r) => r.positionId);
  const eaByPos = groupBy(eaRows, (r) => r.positionId);
  const peByAnalysis = groupBy(peRows, (r) => r.analysisId.toString());

  return rows.map((row) => assemblePosition(row, annByPos, naByPos, eaByPos, peByAnalysis));
}

export async function setLongBench(positionId: string, value: boolean): Promise<void> {
  await db
    .update(positions)
    .set({ longBench: value, updatedAt: new Date() })
    .where(eq(positions.id, positionId));
}

// ─── Benchmark runs ───────────────────────────────────────────────

function toBenchmarkRun(row: typeof benchmarkRuns.$inferSelect): BenchmarkRun {
  return {
    id: row.id,
    engine: row.engine,
    maxNodes: row.maxNodes,
    status: row.status,
    aggregateScore: row.aggregateScore,
    positionCount: row.positionCount,
    errorMessage: row.errorMessage,
    createdAt: row.createdAt.toISOString(),
    startedAt: row.startedAt?.toISOString() ?? null,
    finishedAt: row.finishedAt?.toISOString() ?? null,
  };
}

function toBenchmarkPositionResult(
  row: typeof benchmarkPositionResults.$inferSelect
): BenchmarkPositionResult {
  return {
    id: row.id,
    runId: row.runId,
    positionId: row.positionId,
    score: row.score,
    stableNodes: row.stableNodes,
    failed: row.failed,
    finalCp: row.finalCp,
    finalBestMove: row.finalBestMove,
    totalNodes: row.totalNodes,
    statsHistory: (row.statsHistory as BenchmarkStatsSample[]) ?? [],
    createdAt: row.createdAt.toISOString(),
  };
}

export async function createBenchmarkRun(
  engine: string,
  maxNodes: number
): Promise<BenchmarkRun> {
  const [row] = await db
    .insert(benchmarkRuns)
    .values({ engine, maxNodes, status: "pending" })
    .returning();
  return toBenchmarkRun(row);
}

export async function markBenchmarkRunRunning(
  runId: number,
  positionCount: number
): Promise<void> {
  await db
    .update(benchmarkRuns)
    .set({ status: "running", positionCount, startedAt: new Date() })
    .where(eq(benchmarkRuns.id, runId));
}

export async function completeBenchmarkRun(
  runId: number,
  aggregateScore: number
): Promise<void> {
  await db
    .update(benchmarkRuns)
    .set({
      status: "completed",
      aggregateScore,
      finishedAt: new Date(),
    })
    .where(eq(benchmarkRuns.id, runId));
}

export async function failBenchmarkRun(
  runId: number,
  errorMessage: string
): Promise<void> {
  await db
    .update(benchmarkRuns)
    .set({ status: "failed", errorMessage, finishedAt: new Date() })
    .where(eq(benchmarkRuns.id, runId));
}

export async function upsertBenchmarkPositionResult(
  runId: number,
  positionId: string,
  result: {
    score: number | null;
    stableNodes: number | null;
    failed: boolean;
    finalCp: number | null;
    finalBestMove: string | null;
    totalNodes: number | null;
    statsHistory: BenchmarkStatsSample[];
  }
): Promise<void> {
  await db
    .insert(benchmarkPositionResults)
    .values({
      runId,
      positionId,
      score: result.score,
      stableNodes: result.stableNodes,
      failed: result.failed,
      finalCp: result.finalCp,
      finalBestMove: result.finalBestMove,
      totalNodes: result.totalNodes,
      statsHistory: result.statsHistory,
    })
    .onConflictDoUpdate({
      target: [benchmarkPositionResults.runId, benchmarkPositionResults.positionId],
      set: {
        score: result.score,
        stableNodes: result.stableNodes,
        failed: result.failed,
        finalCp: result.finalCp,
        finalBestMove: result.finalBestMove,
        totalNodes: result.totalNodes,
        statsHistory: result.statsHistory,
      },
    });
}

export async function listBenchmarkRuns(): Promise<BenchmarkRun[]> {
  const rows = await db
    .select()
    .from(benchmarkRuns)
    .orderBy(desc(benchmarkRuns.createdAt));
  return rows.map(toBenchmarkRun);
}

export async function getBenchmarkRun(runId: number): Promise<BenchmarkRunDetail | null> {
  const [runRow] = await db
    .select()
    .from(benchmarkRuns)
    .where(eq(benchmarkRuns.id, runId));
  if (!runRow) return null;

  const resultRows = await db
    .select()
    .from(benchmarkPositionResults)
    .where(eq(benchmarkPositionResults.runId, runId));

  const posIds = resultRows.map((r) => r.positionId);
  const posRows =
    posIds.length > 0
      ? await db.select().from(positions).where(inArray(positions.id, posIds))
      : [];
  const annRows =
    posIds.length > 0
      ? await db
          .select()
          .from(moveAnnotations)
          .where(inArray(moveAnnotations.positionId, posIds))
      : [];

  const posById = new Map(posRows.map((p) => [p.id, p] as const));
  const annByPos = groupBy(annRows, (r) => r.positionId);

  const results = resultRows.map((row) => {
    const p = posById.get(row.positionId);
    const anns = annByPos[row.positionId] ?? [];
    const moveAnns: SharpMoveAnnotation[] = anns.map((a) => ({
      move: a.move,
      annotation: a.annotation,
    }));
    return {
      ...toBenchmarkPositionResult(row),
      position: {
        id: row.positionId,
        name: p?.name ?? "(deleted)",
        type: (p?.type ?? "FORTRESS") as Position["type"],
        fen: p?.fen ?? "",
        expectedOutcome: p?.expectedOutcome ?? undefined,
        moveAnnotations: moveAnns.length > 0 ? moveAnns : undefined,
      },
    };
  });

  return {
    ...toBenchmarkRun(runRow),
    results,
  };
}

/** Sweep runs that were left in `running` state (e.g. server restart). */
export async function markStaleRunsFailed(): Promise<void> {
  await db
    .update(benchmarkRuns)
    .set({
      status: "failed",
      errorMessage: "Interrupted (server restart)",
      finishedAt: new Date(),
    })
    .where(and(eq(benchmarkRuns.status, "running")));
}
