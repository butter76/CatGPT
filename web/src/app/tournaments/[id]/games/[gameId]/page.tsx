"use client";

import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AnalysisBoard } from "@/components/chess/analysis-board";
import {
  ArrowLeft,
  ChevronFirst,
  ChevronLast,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  Loader2,
  Terminal,
  Wifi,
} from "lucide-react";
import type {
  GameMove,
  GameResult,
  TournamentGameDetail,
} from "@/lib/types";
import { fetchTournamentGame } from "@/lib/store";

const STANDARD_START =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

function resultScore(result: GameResult | null): string {
  if (result === "white_win") return "1–0";
  if (result === "black_win") return "0–1";
  if (result === "draw") return "½–½";
  return "*";
}

function formatEval(cp: number | null): string {
  if (cp == null) return "";
  if (Math.abs(cp) >= 100000) return cp > 0 ? "#" : "-#";
  const pawns = cp / 100;
  return (pawns >= 0 ? "+" : "") + pawns.toFixed(2);
}

interface TimeControl {
  baseMs: number;
  incMs: number;
}

/** Parse a cutechess tc string like "900+5", "40/900+5", or "60". */
function parseTimeControl(tc: string | null | undefined): TimeControl | null {
  if (!tc) return null;
  const slash = tc.lastIndexOf("/");
  const tail = slash >= 0 ? tc.slice(slash + 1) : tc;
  const [baseStr, incStr] = tail.split("+");
  const base = Number(baseStr);
  if (!Number.isFinite(base) || base <= 0) return null;
  const inc = incStr != null ? Number(incStr) : 0;
  return { baseMs: base * 1000, incMs: Number.isFinite(inc) ? inc * 1000 : 0 };
}

function formatClock(ms: number | null): string {
  if (ms == null) return "—";
  const clamped = Math.max(0, ms);
  const totalSec = clamped / 1000;
  if (totalSec >= 3600) {
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = Math.floor(totalSec % 60);
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  if (clamped < 20000) {
    // Show tenths under 20s for the urgency cue.
    return (clamped / 1000).toFixed(1);
  }
  const m = Math.floor(totalSec / 60);
  const s = Math.floor(totalSec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

/** Compact "time used" for a single move, e.g. "0.4s", "12s", "1m03". */
function formatThinkTime(ms: number | null): string {
  if (ms == null) return "";
  const sec = ms / 1000;
  if (sec < 10) return `${sec.toFixed(1)}s`;
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}m${String(s).padStart(2, "0")}`;
}

/**
 * Remaining clock for each side at the position reached after `currentPly`
 * plies. Stored clocks are the snapshot *before* each ply, so the state after
 * ply k equals the next ply's pre-move snapshot. The final position has no
 * subsequent snapshot, so it's derived from the last move's think time.
 */
function clocksAtPly(
  moves: GameMove[],
  currentPly: number,
  tc: TimeControl | null
): { white: number | null; black: number | null } {
  const baseMs = tc?.baseMs ?? null;
  if (moves.length === 0) return { white: baseMs, black: baseMs };

  if (currentPly < moves.length) {
    const next = moves[currentPly];
    return {
      white: next.whiteClockMs ?? baseMs,
      black: next.blackClockMs ?? baseMs,
    };
  }

  // Final position: adjust the last move's pre-move snapshot by its think time.
  const last = moves[moves.length - 1];
  const white = last.whiteClockMs ?? baseMs;
  const black = last.blackClockMs ?? baseMs;
  const inc = tc?.incMs ?? 0;
  const spent = last.timeMs ?? 0;
  if (last.mover === "white" && white != null) {
    return { white: white - spent + inc, black };
  }
  if (last.mover === "black" && black != null) {
    return { white, black: black - spent + inc };
  }
  return { white, black };
}

export default function GameReplayPage({
  params,
}: {
  params: Promise<{ id: string; gameId: string }>;
}) {
  const { id, gameId } = use(params);
  const tournamentId = Number(id);
  const gid = Number(gameId);

  const [detail, setDetail] = useState<TournamentGameDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveMoves, setLiveMoves] = useState<GameMove[] | null>(null);
  const [currentPly, setCurrentPly] = useState(0);
  const [showLogs, setShowLogs] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const followRef = useRef(true);

  const load = useCallback(() => {
    setLoading(true);
    fetchTournamentGame(tournamentId, gid)
      .then((d) => setDetail(d))
      .finally(() => setLoading(false));
  }, [tournamentId, gid]);

  useEffect(() => {
    load();
  }, [load]);

  // Live stream for in-progress games.
  useEffect(() => {
    if (!detail || detail.status !== "in_progress") return;
    if (esRef.current) return;

    const es = new EventSource(
      `/api/tournaments/${tournamentId}/games/${gid}/stream`
    );
    esRef.current = es;
    setStreaming(true);

    es.addEventListener("moves", (ev) => {
      const raw = (ev as MessageEvent).data;
      if (raw === "null") {
        es.close();
        esRef.current = null;
        setStreaming(false);
        load();
        return;
      }
      const data = JSON.parse(raw) as { gameId: number; moves: GameMove[] };
      setLiveMoves(data.moves);
      if (followRef.current) setCurrentPly(data.moves.length);
    });

    es.addEventListener("move", (ev) => {
      const move = JSON.parse((ev as MessageEvent).data) as GameMove;
      setLiveMoves((prev) => {
        const next = [...(prev ?? []), move];
        if (followRef.current) setCurrentPly(next.length);
        return next;
      });
    });

    es.addEventListener("done", () => {
      es.close();
      esRef.current = null;
      setStreaming(false);
      load();
    });

    return () => {
      es.close();
      esRef.current = null;
      setStreaming(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tournamentId, gid, detail?.status]);

  const moves: GameMove[] = useMemo(() => {
    if (liveMoves && liveMoves.length > 0) return liveMoves;
    return detail?.moves ?? [];
  }, [liveMoves, detail]);

  const startFen = detail?.openingFen || STANDARD_START;
  const currentFen =
    currentPly <= 0 || moves.length === 0
      ? startFen
      : moves[Math.min(currentPly, moves.length) - 1].fenAfter;

  const timeControl = useMemo(
    () => parseTimeControl(detail?.timeControl),
    [detail?.timeControl]
  );
  const hasClockData = useMemo(
    () => moves.some((m) => m.whiteClockMs != null || m.blackClockMs != null),
    [moves]
  );
  const clocks = useMemo(
    () => clocksAtPly(moves, currentPly, timeControl),
    [moves, currentPly, timeControl]
  );
  const sideToMove: "white" | "black" =
    currentFen.split(" ")[1] === "b" ? "black" : "white";

  const goTo = useCallback(
    (ply: number) => {
      const clamped = Math.max(0, Math.min(ply, moves.length));
      followRef.current = clamped >= moves.length;
      setCurrentPly(clamped);
    },
    [moves.length]
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)
        return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goTo(currentPly - 1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goTo(currentPly + 1);
      } else if (e.key === "Home") {
        e.preventDefault();
        goTo(0);
      } else if (e.key === "End") {
        e.preventDefault();
        goTo(moves.length);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [currentPly, goTo, moves.length]);

  if (loading && !detail) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="text-center py-16">
        <p className="text-lg text-muted-foreground">Game not found</p>
        <Button variant="ghost" asChild className="mt-4">
          <Link href={`/tournaments/${tournamentId}`}>
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to tournament
          </Link>
        </Button>
      </div>
    );
  }

  // Group plies into rows of (white, black) for the move list.
  const rows: { moveNo: number; white?: GameMove; black?: GameMove }[] = [];
  for (const m of moves) {
    if (m.mover === "white") {
      rows.push({ moveNo: rows.length + 1, white: m });
    } else {
      const last = rows[rows.length - 1];
      if (last && !last.black) last.black = m;
      else rows.push({ moveNo: rows.length + 1, black: m });
    }
  }

  const uciLog = detail.uciLogs.map((l) => l.content).join("\n\n");

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <Button variant="ghost" size="sm" asChild>
            <Link href={`/tournaments/${tournamentId}`}>
              <ArrowLeft className="w-4 h-4 mr-1" /> Back to tournament
            </Link>
          </Button>
          <h1 className="text-2xl font-bold">
            Game {detail.gameNumber}: {detail.whiteEngine}
            <span className="text-muted-foreground"> vs </span>
            {detail.blackEngine}
          </h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="font-mono text-foreground">
              {resultScore(detail.result)}
            </span>
            {detail.termination && <span>· {detail.termination}</span>}
            {streaming && (
              <Badge variant="outline" className="border-blue-500 text-blue-600 gap-1">
                <Wifi className="w-3 h-3" /> live
              </Badge>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-6">
        {/* Board + controls */}
        <div className="space-y-3">
          {hasClockData && (
            <ClockBar
              name={detail.blackEngine}
              ms={clocks.black}
              active={sideToMove === "black"}
            />
          )}
          <AnalysisBoard fen={currentFen} width={420} />
          {hasClockData && (
            <ClockBar
              name={detail.whiteEngine}
              ms={clocks.white}
              active={sideToMove === "white"}
            />
          )}
          <div className="flex items-center justify-between gap-2">
            <div className="flex gap-1">
              <Button variant="outline" size="sm" onClick={() => goTo(0)} title="Start (Home)">
                <ChevronFirst className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={() => goTo(currentPly - 1)} title="Previous (←)">
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={() => goTo(currentPly + 1)} title="Next (→)">
                <ChevronRight className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm" onClick={() => goTo(moves.length)} title="End (End)">
                <ChevronLast className="w-4 h-4" />
              </Button>
            </div>
            <span className="text-xs text-muted-foreground font-mono">
              ply {currentPly}/{moves.length}
            </span>
          </div>
          <Button variant="secondary" size="sm" asChild className="w-full">
            <Link href={`/analyze?fen=${encodeURIComponent(currentFen)}`}>
              <FlaskConical className="w-4 h-4 mr-1.5" />
              Open position in Quick Analysis
            </Link>
          </Button>
        </div>

        {/* Move list */}
        <Card className="min-w-0">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Moves</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="max-h-[480px] overflow-y-auto">
              {rows.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted-foreground">
                  {detail.status === "in_progress"
                    ? "Waiting for moves…"
                    : "No moves recorded."}
                </p>
              ) : (
                <table className="w-full text-sm">
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.moveNo} className="border-b last:border-b-0">
                        <td className="py-1 px-3 text-xs text-muted-foreground font-mono w-10">
                          {row.moveNo}.
                        </td>
                        <MoveCell move={row.white} currentPly={currentPly} onClick={goTo} />
                        <MoveCell move={row.black} currentPly={currentPly} onClick={goTo} />
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* UCI logs */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-muted-foreground" />
              UCI debug logs
            </span>
            <Button variant="ghost" size="sm" onClick={() => setShowLogs((s) => !s)}>
              {showLogs ? "Hide" : "Show"}
            </Button>
          </CardTitle>
        </CardHeader>
        {showLogs && (
          <CardContent>
            {uciLog ? (
              <pre className="text-[11px] leading-relaxed font-mono bg-muted/40 rounded p-3 max-h-[480px] overflow-auto whitespace-pre-wrap break-all">
                {uciLog}
              </pre>
            ) : (
              <p className="text-sm text-muted-foreground">
                {detail.status === "in_progress"
                  ? "Logs are stored when the game finishes."
                  : "No UCI logs stored for this game."}
              </p>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  );
}

function ClockBar({
  name,
  ms,
  active,
}: {
  name: string;
  ms: number | null;
  active: boolean;
}) {
  const low = ms != null && ms < 30000;
  return (
    <div
      className={`flex items-center justify-between rounded-md border px-3 py-1.5 ${
        active
          ? "border-rose-500/60 bg-rose-500/10"
          : "border-border bg-muted/30"
      }`}
    >
      <span className="text-sm font-medium truncate mr-3">{name}</span>
      <span
        className={`font-mono tabular-nums text-lg ${
          active ? "text-foreground" : "text-muted-foreground"
        } ${low ? "text-red-600" : ""}`}
      >
        {formatClock(ms)}
      </span>
    </div>
  );
}

function MoveCell({
  move,
  currentPly,
  onClick,
}: {
  move: GameMove | undefined;
  currentPly: number;
  onClick: (ply: number) => void;
}) {
  if (!move) return <td className="py-1 px-3" />;
  const isCurrent = currentPly === move.ply;
  const evalStr = formatEval(move.evalCp);
  const timeStr = formatThinkTime(move.timeMs);
  return (
    <td className="py-1 px-3">
      <button
        onClick={() => onClick(move.ply)}
        className={`inline-flex items-baseline gap-1.5 rounded px-1.5 py-0.5 transition-colors ${
          isCurrent ? "bg-rose-500/20 text-foreground" : "hover:bg-muted"
        }`}
      >
        <span className="font-medium">{move.san}</span>
        {evalStr && (
          <span
            className={`text-[10px] font-mono ${
              (move.evalCp ?? 0) >= 0 ? "text-green-600" : "text-red-600"
            }`}
          >
            {evalStr}
          </span>
        )}
        {timeStr && (
          <span className="text-[10px] font-mono text-muted-foreground">
            {timeStr}
          </span>
        )}
      </button>
    </td>
  );
}
