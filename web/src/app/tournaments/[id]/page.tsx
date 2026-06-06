"use client";

import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CompactBoard } from "@/components/chess/analysis-board";
import {
  ArrowLeft,
  Swords,
  Loader2,
  RefreshCw,
  Hourglass,
  CheckCircle2,
  XCircle,
  Square,
  Wifi,
  WifiOff,
} from "lucide-react";
import type {
  GameMove,
  GameResult,
  TournamentDetail,
  TournamentGameSummary,
  TournamentStatus,
} from "@/lib/types";
import { abortTournamentAPI, fetchTournament } from "@/lib/store";

const STANDARD_START =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

interface LiveGame {
  gameId: number;
  gameNumber: number;
  white: string;
  black: string;
  openingFen: string;
  moves: GameMove[];
}

type ConnectionState = "connecting" | "open" | "closed";

interface LiveState {
  status: TournamentStatus;
  error: string | null;
  scoreFirst: number;
  scoreSecond: number;
  scoreDraws: number;
  finishedGames: TournamentGameSummary[];
  liveGames: LiveGame[];
  connection: ConnectionState;
  streamDone: boolean;
}

function statusBadge(status: TournamentStatus) {
  switch (status) {
    case "pending":
      return (
        <Badge variant="outline" className="border-yellow-500 text-yellow-600">
          <Hourglass className="w-3 h-3 mr-1" /> pending
        </Badge>
      );
    case "running":
      return (
        <Badge variant="outline" className="border-blue-500 text-blue-600">
          <Loader2 className="w-3 h-3 mr-1 animate-spin" /> running
        </Badge>
      );
    case "completed":
      return (
        <Badge variant="outline" className="border-green-500 text-green-600">
          <CheckCircle2 className="w-3 h-3 mr-1" /> completed
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="outline" className="border-red-500 text-red-600">
          <XCircle className="w-3 h-3 mr-1" /> failed
        </Badge>
      );
    case "cancelled":
      return (
        <Badge variant="outline" className="border-gray-400 text-gray-500">
          cancelled
        </Badge>
      );
  }
}

function resultLabel(
  game: { result: GameResult | null; whiteEngine: string; blackEngine: string }
): string {
  if (game.result === "white_win") return `${game.whiteEngine} won (W)`;
  if (game.result === "black_win") return `${game.blackEngine} won (B)`;
  if (game.result === "draw") return "Draw";
  return "—";
}

function resultScore(result: GameResult | null): string {
  if (result === "white_win") return "1–0";
  if (result === "black_win") return "0–1";
  if (result === "draw") return "½–½";
  return "*";
}

export default function TournamentDashboardPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const tournamentId = Number(id);

  const [detail, setDetail] = useState<TournamentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [aborting, setAborting] = useState(false);
  const [live, setLive] = useState<LiveState | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    fetchTournament(tournamentId)
      .then((d) => setDetail(d))
      .finally(() => setLoading(false));
  }, [tournamentId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!detail) return;
    if (
      detail.status === "completed" ||
      detail.status === "failed" ||
      detail.status === "cancelled"
    ) {
      return;
    }
    if (esRef.current) return;

    const es = new EventSource(`/api/tournaments/${tournamentId}/stream`);
    esRef.current = es;

    const setConn = (connection: ConnectionState) =>
      setLive((prev) => (prev ? { ...prev, connection } : prev));

    es.addEventListener("open", () => setConn("open"));

    es.addEventListener("snapshot", (ev) => {
      const raw = (ev as MessageEvent).data;
      if (raw === "null") {
        es.close();
        esRef.current = null;
        load();
        return;
      }
      const snap = JSON.parse(raw) as {
        status: TournamentStatus;
        error: string | null;
        scoreFirst: number;
        scoreSecond: number;
        scoreDraws: number;
        games: TournamentGameSummary[];
        liveGames: LiveGame[];
      };
      setLive({
        status: snap.status,
        error: snap.error,
        scoreFirst: snap.scoreFirst,
        scoreSecond: snap.scoreSecond,
        scoreDraws: snap.scoreDraws,
        finishedGames: snap.games,
        liveGames: snap.liveGames,
        connection: "open",
        streamDone: false,
      });
    });

    es.addEventListener("game_started", (ev) => {
      const g = JSON.parse((ev as MessageEvent).data) as LiveGame;
      setLive((prev) =>
        prev
          ? {
              ...prev,
              liveGames: [
                ...prev.liveGames.filter((x) => x.gameId !== g.gameId),
                g,
              ],
            }
          : prev
      );
    });

    es.addEventListener("move", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        gameId: number;
        gameNumber: number;
        move: GameMove;
      };
      setLive((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          liveGames: prev.liveGames.map((g) =>
            g.gameId === data.gameId
              ? { ...g, moves: [...g.moves, data.move] }
              : g
          ),
        };
      });
    });

    es.addEventListener("game_finished", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        game: TournamentGameSummary;
      };
      setLive((prev) =>
        prev
          ? {
              ...prev,
              liveGames: prev.liveGames.filter(
                (g) => g.gameId !== data.game.id
              ),
              finishedGames: [
                ...prev.finishedGames.filter((x) => x.id !== data.game.id),
                data.game,
              ],
            }
          : prev
      );
    });

    es.addEventListener("score", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data) as {
        first: number;
        second: number;
        draws: number;
      };
      setLive((prev) =>
        prev
          ? {
              ...prev,
              scoreFirst: data.first,
              scoreSecond: data.second,
              scoreDraws: data.draws,
            }
          : prev
      );
    });

    es.addEventListener("run_complete", () => {
      setLive((prev) => (prev ? { ...prev, status: "completed" } : prev));
    });

    es.addEventListener("error", (ev) => {
      const raw = (ev as MessageEvent).data;
      if (typeof raw === "string" && raw.length > 0) {
        try {
          const parsed = JSON.parse(raw) as { message?: string };
          if (parsed?.message) {
            setLive((prev) =>
              prev ? { ...prev, error: parsed.message ?? null, status: "failed" } : prev
            );
            return;
          }
        } catch {
          // native connection error
        }
      }
      setConn("connecting");
    });

    es.addEventListener("ping", () => setConn("open"));

    es.addEventListener("done", () => {
      setLive((prev) =>
        prev ? { ...prev, streamDone: true, connection: "closed" } : prev
      );
      es.close();
      esRef.current = null;
      load();
    });

    return () => {
      es.close();
      esRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tournamentId, detail?.status]);

  const handleAbort = useCallback(async () => {
    if (!confirm("Abort this tournament? The running cutechess process will be killed.")) return;
    setAborting(true);
    try {
      await abortTournamentAPI(tournamentId);
    } finally {
      setAborting(false);
    }
  }, [tournamentId]);

  // Merge DB games with live overlay.
  const mergedGames = useMemo(() => {
    const map = new Map<number, TournamentGameSummary>();
    if (detail) for (const g of detail.games) map.set(g.id, g);
    if (live) for (const g of live.finishedGames) map.set(g.id, g);
    return [...map.values()].sort((a, b) => a.gameNumber - b.gameNumber);
  }, [detail, live]);

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
        <p className="text-lg text-muted-foreground">Tournament not found</p>
        <Button variant="ghost" asChild className="mt-4">
          <Link href="/tournaments">
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to Tournaments
          </Link>
        </Button>
      </div>
    );
  }

  const status = live?.status ?? detail.status;
  const isLive = !!live && !live.streamDone;
  const canAbort = status === "pending" || status === "running";

  const scoreFirst = live?.scoreFirst ?? detail.scoreWhite;
  const scoreSecond = live?.scoreSecond ?? detail.scoreBlack;
  const scoreDraws = live?.scoreDraws ?? detail.scoreDraw;
  const played = scoreFirst + scoreSecond + scoreDraws;
  const points = scoreFirst + scoreDraws / 2;
  const liveGames = live?.liveGames ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/tournaments">
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Link>
          </Button>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Swords className="w-6 h-6 text-rose-500" />
            {detail.name}
          </h1>
          <p className="text-sm text-muted-foreground font-mono">
            {detail.whiteLabel} vs {detail.blackLabel} · tc={detail.timeControl}
            {detail.tbPath ? " · syzygy adjudication" : ""}
          </p>
        </div>
        <div className="flex gap-2">
          {isLive && <ConnectionIndicator state={live!.connection} />}
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-1.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          {canAbort && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleAbort}
              disabled={aborting}
              className="border-red-500/50 text-red-500 hover:bg-red-500/10 hover:text-red-600"
            >
              {aborting ? (
                <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
              ) : (
                <Square className="w-4 h-4 mr-1.5" />
              )}
              Abort
            </Button>
          )}
        </div>
      </div>

      {/* Standings */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center justify-between">
            <span>Standings</span>
            {statusBadge(status)}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-8">
            <div>
              <div className="text-xs text-muted-foreground">Score (W–D–L)</div>
              <div className="text-3xl font-bold font-mono">
                {scoreFirst}<span className="text-muted-foreground text-xl">–</span>
                {scoreDraws}<span className="text-muted-foreground text-xl">–</span>
                {scoreSecond}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {detail.whiteLabel} perspective
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Points</div>
              <div className="text-2xl font-mono">
                {points.toFixed(1)}
                <span className="text-muted-foreground text-base"> / {played}</span>
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Progress</div>
              <div className="text-2xl font-mono">
                {played}
                <span className="text-muted-foreground text-base"> / {detail.totalGames}</span>
              </div>
            </div>
          </div>
          <div className="h-2 w-full rounded bg-muted overflow-hidden mt-4">
            <div
              className="h-full bg-rose-500 transition-all"
              style={{
                width: `${detail.totalGames > 0 ? (played / detail.totalGames) * 100 : 0}%`,
              }}
            />
          </div>
          {(live?.error || detail.errorMessage) && (
            <p className="mt-3 text-sm text-red-500 font-mono break-all">
              {live?.error || detail.errorMessage}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Live games */}
      {liveGames.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              In progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {liveGames
                .slice()
                .sort((a, b) => a.gameNumber - b.gameNumber)
                .map((g) => {
                  const fen =
                    g.moves.length > 0
                      ? g.moves[g.moves.length - 1].fenAfter
                      : g.openingFen || STANDARD_START;
                  const last = g.moves[g.moves.length - 1];
                  return (
                    <Link
                      key={g.gameId}
                      href={`/tournaments/${tournamentId}/games/${g.gameId}`}
                      className="block rounded-md border p-3 hover:bg-muted/30 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold">Game {g.gameNumber}</span>
                        <Badge variant="outline" className="border-blue-500 text-blue-600 text-[10px]">
                          live
                        </Badge>
                      </div>
                      <CompactBoard fen={fen} width={180} />
                      <div className="mt-2 text-xs">
                        <div className="truncate">
                          <span className="text-muted-foreground">W:</span> {g.white}
                        </div>
                        <div className="truncate">
                          <span className="text-muted-foreground">B:</span> {g.black}
                        </div>
                        <div className="font-mono text-muted-foreground mt-1">
                          {g.moves.length} plies
                          {last ? ` · ${last.san}` : ""}
                          {last?.evalCp != null ? ` (${(last.evalCp / 100).toFixed(2)})` : ""}
                        </div>
                      </div>
                    </Link>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Games table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Games</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {mergedGames.length === 0 ? (
            <p className="py-10 text-center text-sm text-muted-foreground">
              No completed games yet.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                  <tr>
                    <th className="text-left py-2 px-4">#</th>
                    <th className="text-left py-2 px-4">White</th>
                    <th className="text-left py-2 px-4">Black</th>
                    <th className="text-center py-2 px-4">Result</th>
                    <th className="text-left py-2 px-4">Outcome</th>
                    <th className="text-left py-2 px-4">Termination</th>
                    <th className="text-right py-2 px-4">Plies</th>
                    <th className="py-2 px-4" />
                  </tr>
                </thead>
                <tbody>
                  {mergedGames.map((g) => (
                    <tr
                      key={g.id}
                      className="border-b last:border-b-0 hover:bg-muted/30 transition-colors"
                    >
                      <td className="py-2 px-4 font-mono text-xs">{g.gameNumber}</td>
                      <td className="py-2 px-4 text-xs">{g.whiteEngine}</td>
                      <td className="py-2 px-4 text-xs">{g.blackEngine}</td>
                      <td className="py-2 px-4 text-center font-mono">
                        {resultScore(g.result)}
                      </td>
                      <td className="py-2 px-4 text-xs text-muted-foreground">
                        {resultLabel(g)}
                      </td>
                      <td className="py-2 px-4 text-xs text-muted-foreground max-w-[200px] truncate">
                        {g.termination ?? (g.status === "in_progress" ? "playing…" : "—")}
                      </td>
                      <td className="py-2 px-4 text-right font-mono">{g.plyCount}</td>
                      <td className="py-2 px-4 text-right">
                        <Button asChild variant="ghost" size="sm" className="h-7 text-xs">
                          <Link href={`/tournaments/${tournamentId}/games/${g.id}`}>
                            View
                          </Link>
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ConnectionIndicator({ state }: { state: ConnectionState }) {
  if (state === "open") {
    return (
      <Badge variant="outline" className="border-green-500 text-green-600 h-8 gap-1">
        <Wifi className="w-3.5 h-3.5" /> live
      </Badge>
    );
  }
  if (state === "connecting") {
    return (
      <Badge variant="outline" className="border-yellow-500 text-yellow-600 h-8 gap-1">
        <Loader2 className="w-3.5 h-3.5 animate-spin" /> reconnecting
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="border-gray-400 text-gray-500 h-8 gap-1">
      <WifiOff className="w-3.5 h-3.5" /> closed
    </Badge>
  );
}
