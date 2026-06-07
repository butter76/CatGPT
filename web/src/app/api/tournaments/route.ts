import { NextRequest, NextResponse } from "next/server";
import { createTournament, listTournaments } from "@/db/queries";
import { cutechessRunner } from "@/lib/cutechess-runner";
import {
  SYZYGY_HOME,
  defaultCatgptConfig,
  defaultStockfishConfig,
  getTournamentEnvAvailability,
} from "@/lib/cutechess";
import type { EngineConfig } from "@/lib/types";

function sanitizeConfig(
  raw: unknown,
  fallback: EngineConfig
): EngineConfig {
  if (!raw || typeof raw !== "object") return fallback;
  const r = raw as Record<string, unknown>;
  const name =
    typeof r.name === "string" && r.name.trim() ? r.name.trim() : fallback.name;
  const command =
    typeof r.command === "string" && r.command.trim()
      ? r.command.trim()
      : fallback.command;
  const options = Array.isArray(r.options)
    ? r.options
        .filter(
          (o): o is { name: string; value: string } =>
            !!o &&
            typeof o === "object" &&
            typeof (o as { name?: unknown }).name === "string" &&
            typeof (o as { value?: unknown }).value === "string"
        )
        .map((o) => ({ name: o.name, value: String(o.value) }))
    : fallback.options;
  const initStrings = Array.isArray(r.initStrings)
    ? r.initStrings.filter((s): s is string => typeof s === "string")
    : fallback.initStrings;
  const timeControl =
    typeof r.timeControl === "string" && r.timeControl.trim()
      ? r.timeControl.trim()
      : fallback.timeControl;
  return { name, command, options, initStrings, timeControl };
}

function posInt(v: unknown, fallback: number, min: number): number {
  return typeof v === "number" && Number.isFinite(v) && v >= min
    ? Math.floor(v)
    : fallback;
}

// GET /api/tournaments — list all tournaments
export async function GET() {
  try {
    const rows = await listTournaments();
    return NextResponse.json(rows);
  } catch (error) {
    console.error("Failed to list tournaments:", error);
    return NextResponse.json(
      { error: "Failed to list tournaments" },
      { status: 500 }
    );
  }
}

// POST /api/tournaments — create + start a tournament
export async function POST(request: NextRequest) {
  try {
    const env = getTournamentEnvAvailability();
    if (!env.cutechess) {
      return NextResponse.json(
        {
          error:
            "cutechess-cli not found. Build it with scripts/build-cutechess.sh and set CUTECHESS_CLI_PATH.",
        },
        { status: 400 }
      );
    }

    const body = await request.json().catch(() => ({}));

    const whiteConfig = sanitizeConfig(body.whiteConfig, defaultCatgptConfig());
    let blackConfig = sanitizeConfig(body.blackConfig, defaultStockfishConfig());

    // cutechess requires distinct engine names (used for `-engine conf=`).
    if (blackConfig.name === whiteConfig.name) {
      blackConfig = { ...blackConfig, name: `${blackConfig.name} (2)` };
    }

    const timeControl =
      typeof body.timeControl === "string" && body.timeControl.trim()
        ? body.timeControl.trim()
        : "900+5";
    const totalGames = posInt(body.totalGames, 2, 1);
    const concurrency = posInt(body.concurrency, 1, 1);
    const openingBook =
      typeof body.openingBook === "string" && body.openingBook.trim()
        ? body.openingBook.trim()
        : null;
    const drawMoveNumber = posInt(body.drawMoveNumber, 1, 1);
    const drawMoveCount = posInt(body.drawMoveCount, 7, 1);
    const drawScoreCp = posInt(body.drawScoreCp, 25, 0);
    const tbPath =
      typeof body.tbPath === "string"
        ? body.tbPath.trim() || null
        : SYZYGY_HOME || null;

    const name =
      typeof body.name === "string" && body.name.trim()
        ? body.name.trim()
        : `${whiteConfig.name} vs ${blackConfig.name}`;

    const tournament = await createTournament({
      name,
      whiteLabel: whiteConfig.name,
      blackLabel: blackConfig.name,
      whiteConfig,
      blackConfig,
      timeControl,
      totalGames,
      concurrency,
      openingBook,
      drawMoveNumber,
      drawMoveCount,
      drawScoreCp,
      tbPath,
    });

    // Kick off the runner immediately so the match is independent of this
    // HTTP request. The client connects via /stream afterward.
    void cutechessRunner.start(tournament.id);

    return NextResponse.json(tournament, { status: 201 });
  } catch (error) {
    console.error("Failed to create tournament:", error);
    return NextResponse.json(
      { error: "Failed to create tournament" },
      { status: 500 }
    );
  }
}
