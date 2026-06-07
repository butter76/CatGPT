"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Swords,
  Loader2,
  PlayCircle,
  CheckCircle2,
  XCircle,
  Hourglass,
  RefreshCw,
  Play,
  AlertTriangle,
} from "lucide-react";
import type { Tournament, TournamentStatus, EngineConfig } from "@/lib/types";
import {
  createTournamentAPI,
  fetchTournamentEngines,
  fetchTournaments,
  type TournamentEnginesInfo,
} from "@/lib/store";

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

export default function TournamentsPage() {
  const router = useRouter();
  const [tournaments, setTournaments] = useState<Tournament[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    fetchTournaments()
      .then(setTournaments)
      .catch(() => setTournaments([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const hasActive = tournaments.some(
      (t) => t.status === "running" || t.status === "pending"
    );
    if (!hasActive) return;
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [tournaments, load]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Swords className="w-6 h-6 text-rose-500" />
            Tournaments
          </h1>
          <p className="text-sm text-muted-foreground">
            Run engine-vs-engine matches with cutechess. Games stream live,
            store full UCI logs, and are replayable move-by-move.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-1.5 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <NewTournamentDialog
            open={dialogOpen}
            onOpenChange={setDialogOpen}
            onCreated={(id) => {
              setDialogOpen(false);
              router.push(`/tournaments/${id}`);
            }}
          />
        </div>
      </div>

      {loading && tournaments.length === 0 ? (
        <div className="flex justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      ) : tournaments.length === 0 ? (
        <Card>
          <CardContent className="py-16 text-center text-muted-foreground space-y-2">
            <Swords className="w-10 h-10 mx-auto text-rose-500/60" />
            <p className="text-lg font-medium">No tournaments yet</p>
            <p className="text-sm">
              Start a new match above (e.g. CatGPT vs Stockfish at 15m+5s).
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b bg-muted/30">
                  <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="py-2 px-4">#</th>
                    <th className="py-2 px-4">Name</th>
                    <th className="py-2 px-4">Status</th>
                    <th className="py-2 px-4">Match</th>
                    <th className="py-2 px-4 text-center">Score (W–D–L)</th>
                    <th className="py-2 px-4 text-right">Games</th>
                    <th className="py-2 px-4">TC</th>
                    <th className="py-2 px-4">Started</th>
                    <th className="py-2 px-4" />
                  </tr>
                </thead>
                <tbody>
                  {tournaments.map((t) => {
                    const played = t.scoreWhite + t.scoreBlack + t.scoreDraw;
                    return (
                      <tr
                        key={t.id}
                        className="border-b last:border-b-0 hover:bg-muted/30 transition-colors"
                      >
                        <td className="py-2 px-4 font-mono text-xs">
                          <Link
                            href={`/tournaments/${t.id}`}
                            className="hover:underline text-rose-500"
                          >
                            #{t.id}
                          </Link>
                        </td>
                        <td className="py-2 px-4 max-w-[200px] truncate">{t.name}</td>
                        <td className="py-2 px-4">{statusBadge(t.status)}</td>
                        <td className="py-2 px-4 text-xs">
                          <span className="font-medium">{t.whiteLabel}</span>
                          <span className="text-muted-foreground"> vs </span>
                          <span className="font-medium">{t.blackLabel}</span>
                        </td>
                        <td className="py-2 px-4 text-center font-mono">
                          {t.scoreWhite}–{t.scoreDraw}–{t.scoreBlack}
                        </td>
                        <td className="py-2 px-4 text-right font-mono">
                          {played}/{t.totalGames}
                        </td>
                        <td className="py-2 px-4 font-mono text-xs">{t.timeControl}</td>
                        <td className="py-2 px-4 text-xs text-muted-foreground">
                          {t.startedAt
                            ? new Date(t.startedAt).toLocaleString()
                            : new Date(t.createdAt).toLocaleString()}
                        </td>
                        <td className="py-2 px-4 text-right">
                          <Button asChild variant="ghost" size="sm" className="h-7 text-xs">
                            <Link href={`/tournaments/${t.id}`}>Open</Link>
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ─── New tournament dialog ────────────────────────────────────────

function parseOptions(raw: string): { name: string; value: string }[] {
  return raw
    .split(/[,\n]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((pair) => {
      const eq = pair.indexOf("=");
      if (eq === -1) return null;
      return { name: pair.slice(0, eq).trim(), value: pair.slice(eq + 1).trim() };
    })
    .filter((o): o is { name: string; value: string } => !!o && !!o.name);
}

function optionsToString(opts?: { name: string; value: string }[]): string {
  if (!opts || opts.length === 0) return "";
  return opts.map((o) => `${o.name}=${o.value}`).join(", ");
}

type PresetKey = "catgpt" | "stockfish" | "lc0" | "custom";

interface EnginePreset {
  key: PresetKey;
  label: string;
  available?: boolean;
  config?: EngineConfig;
}

function buildPresets(info: TournamentEnginesInfo | null): EnginePreset[] {
  return [
    {
      key: "catgpt",
      label: "CatGPT",
      available: info?.catgpt,
      config: info?.defaultConfigs.catgpt,
    },
    {
      key: "stockfish",
      label: "Stockfish",
      available: info?.stockfish,
      config: info?.defaultConfigs.stockfish,
    },
    {
      key: "lc0",
      label: "Lc0 (onnx-trt)",
      available: info?.lc0,
      config: info?.defaultConfigs.lc0,
    },
    { key: "custom", label: "Custom" },
  ];
}

function NewTournamentDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onCreated: (id: number) => void;
}) {
  const [info, setInfo] = useState<TournamentEnginesInfo | null>(null);
  const [name, setName] = useState("");
  const [whitePreset, setWhitePreset] = useState<PresetKey>("catgpt");
  const [whiteName, setWhiteName] = useState("CatGPT");
  const [whiteCommand, setWhiteCommand] = useState("");
  const [whiteOptions, setWhiteOptions] = useState("");
  const [whiteTc, setWhiteTc] = useState("");
  const [blackPreset, setBlackPreset] = useState<PresetKey>("stockfish");
  const [blackName, setBlackName] = useState("Stockfish");
  const [blackCommand, setBlackCommand] = useState("");
  const [blackOptions, setBlackOptions] = useState("Threads=8, Hash=8192");
  const [blackTc, setBlackTc] = useState("");
  const [timeControl, setTimeControl] = useState("900+5");
  const [totalGames, setTotalGames] = useState("2");
  const [concurrency, setConcurrency] = useState("1");
  const [openingBook, setOpeningBook] = useState("");
  const [drawMoveNumber, setDrawMoveNumber] = useState("1");
  const [drawMoveCount, setDrawMoveCount] = useState("7");
  const [drawScoreCp, setDrawScoreCp] = useState("25");
  const [tbPath, setTbPath] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    fetchTournamentEngines()
      .then((i) => {
        setInfo(i);
        setWhiteName(i.defaultConfigs.catgpt.name);
        setWhiteCommand(i.defaultConfigs.catgpt.command);
        setWhiteOptions(optionsToString(i.defaultConfigs.catgpt.options));
        setBlackName(i.defaultConfigs.stockfish.name);
        setBlackCommand(i.defaultConfigs.stockfish.command);
        setBlackOptions(optionsToString(i.defaultConfigs.stockfish.options));
        setTbPath(i.defaults.syzygyPath || "");
      })
      .catch(() => setInfo(null));
  }, [open]);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const whiteConfig: EngineConfig = {
        name: whiteName.trim() || "Engine A",
        command: whiteCommand.trim(),
        options: parseOptions(whiteOptions),
      };
      if (whiteTc.trim()) whiteConfig.timeControl = whiteTc.trim();
      const blackConfig: EngineConfig = {
        name: blackName.trim() || "Engine B",
        command: blackCommand.trim(),
        options: parseOptions(blackOptions),
      };
      if (blackTc.trim()) blackConfig.timeControl = blackTc.trim();
      const tournament = await createTournamentAPI({
        name: name.trim() || undefined,
        whiteConfig,
        blackConfig,
        timeControl: timeControl.trim() || "900+5",
        totalGames: Math.max(1, Number(totalGames) || 1),
        concurrency: Math.max(1, Number(concurrency) || 1),
        openingBook: openingBook.trim() || null,
        drawMoveNumber: Math.max(1, Number(drawMoveNumber) || 1),
        drawMoveCount: Math.max(1, Number(drawMoveCount) || 1),
        drawScoreCp: Math.max(0, Number(drawScoreCp) || 0),
        tbPath: tbPath.trim() || null,
      });
      onCreated(tournament.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const presets = buildPresets(info);

  const applyPreset = (
    key: PresetKey,
    setPreset: (k: PresetKey) => void,
    setters: {
      setName: (v: string) => void;
      setCommand: (v: string) => void;
      setOptions: (v: string) => void;
    }
  ) => {
    setPreset(key);
    if (key === "custom") return;
    const cfg = presets.find((p) => p.key === key)?.config;
    if (!cfg) return;
    setters.setName(cfg.name);
    setters.setCommand(cfg.command);
    setters.setOptions(optionsToString(cfg.options));
  };

  const cutechessMissing = info && !info.cutechess;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button size="sm">
          <Play className="w-4 h-4 mr-1.5" />
          New tournament
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>New tournament</DialogTitle>
          <DialogDescription>
            Configure an engine-vs-engine match. Adjudication uses Syzygy
            tablebases plus a draw rule (both evals within the threshold for N
            consecutive moves).
          </DialogDescription>
        </DialogHeader>

        {cutechessMissing && (
          <div className="flex items-start gap-2 rounded-md border border-yellow-500/50 bg-yellow-500/10 p-3 text-xs text-yellow-700 dark:text-yellow-400">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <span>
              cutechess-cli was not found. Build it with{" "}
              <code>scripts/build-cutechess.sh</code> and set{" "}
              <code>CUTECHESS_CLI_PATH</code>.
            </span>
          </div>
        )}

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="t-name">Name</Label>
            <Input
              id="t-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="(defaults to 'A vs B')"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <EngineFields
              title="Engine A"
              presets={presets}
              preset={whitePreset}
              onPreset={(key) =>
                applyPreset(key, setWhitePreset, {
                  setName: setWhiteName,
                  setCommand: setWhiteCommand,
                  setOptions: setWhiteOptions,
                })
              }
              name={whiteName}
              setName={(v) => {
                setWhiteName(v);
                setWhitePreset("custom");
              }}
              command={whiteCommand}
              setCommand={(v) => {
                setWhiteCommand(v);
                setWhitePreset("custom");
              }}
              options={whiteOptions}
              setOptions={(v) => {
                setWhiteOptions(v);
                setWhitePreset("custom");
              }}
              timeControl={whiteTc}
              setTimeControl={setWhiteTc}
              tcPlaceholder={timeControl}
            />
            <EngineFields
              title="Engine B"
              presets={presets}
              preset={blackPreset}
              onPreset={(key) =>
                applyPreset(key, setBlackPreset, {
                  setName: setBlackName,
                  setCommand: setBlackCommand,
                  setOptions: setBlackOptions,
                })
              }
              name={blackName}
              setName={(v) => {
                setBlackName(v);
                setBlackPreset("custom");
              }}
              command={blackCommand}
              setCommand={(v) => {
                setBlackCommand(v);
                setBlackPreset("custom");
              }}
              options={blackOptions}
              setOptions={(v) => {
                setBlackOptions(v);
                setBlackPreset("custom");
              }}
              timeControl={blackTc}
              setTimeControl={setBlackTc}
              tcPlaceholder={timeControl}
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="t-tc">Time control (default)</Label>
              <Input id="t-tc" value={timeControl} onChange={(e) => setTimeControl(e.target.value)} />
              <p className="text-[11px] text-muted-foreground">900+5 = 15m+5s. Per-engine override below.</p>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="t-games">Games</Label>
              <Input id="t-games" value={totalGames} inputMode="numeric" onChange={(e) => setTotalGames(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="t-conc">Concurrency</Label>
              <Input id="t-conc" value={concurrency} inputMode="numeric" onChange={(e) => setConcurrency(e.target.value)} />
            </div>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="t-book">Openings book (optional path)</Label>
            <Input
              id="t-book"
              value={openingBook}
              onChange={(e) => setOpeningBook(e.target.value)}
              placeholder="/path/to/openings.pgn or .epd"
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="t-tb">Syzygy path (-tb)</Label>
              <Input
                id="t-tb"
                value={tbPath}
                onChange={(e) => setTbPath(e.target.value)}
                placeholder="$SYZYGY_HOME"
                className="text-xs"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="t-dn">Draw from move</Label>
              <Input id="t-dn" value={drawMoveNumber} inputMode="numeric" onChange={(e) => setDrawMoveNumber(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="t-dc">Draw moves</Label>
              <Input id="t-dc" value={drawMoveCount} inputMode="numeric" onChange={(e) => setDrawMoveCount(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="t-ds">Draw score (cp)</Label>
              <Input id="t-ds" value={drawScoreCp} inputMode="numeric" onChange={(e) => setDrawScoreCp(e.target.value)} />
            </div>
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={submitting}>
            {submitting ? (
              <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
            ) : (
              <PlayCircle className="w-4 h-4 mr-1.5" />
            )}
            Start
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function EngineFields({
  title,
  presets,
  preset,
  onPreset,
  name,
  setName,
  command,
  setCommand,
  options,
  setOptions,
  timeControl,
  setTimeControl,
  tcPlaceholder,
}: {
  title: string;
  presets: EnginePreset[];
  preset: PresetKey;
  onPreset: (key: PresetKey) => void;
  name: string;
  setName: (v: string) => void;
  command: string;
  setCommand: (v: string) => void;
  options: string;
  setOptions: (v: string) => void;
  timeControl: string;
  setTimeControl: (v: string) => void;
  tcPlaceholder: string;
}) {
  const selected = presets.find((p) => p.key === preset);
  const presetMissing =
    selected && selected.key !== "custom" && selected.available === false;
  return (
    <div className="space-y-2 rounded-md border p-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">{title}</span>
        {presetMissing && (
          <Badge variant="outline" className="border-yellow-500 text-yellow-600 text-[10px]">
            binary not found
          </Badge>
        )}
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs">Engine</Label>
        <Select value={preset} onValueChange={(v) => onPreset(v as PresetKey)}>
          <SelectTrigger size="sm" className="w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {presets.map((p) => (
              <SelectItem key={p.key} value={p.key} className="text-xs">
                {p.label}
                {p.available === false ? " (not found)" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs">Name</Label>
        <Input value={name} onChange={(e) => setName(e.target.value)} className="h-8 text-xs" />
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs">Command</Label>
        <Input
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          className="h-8 text-xs font-mono"
          placeholder="/path/to/engine [args]"
        />
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs">UCI options</Label>
        <Input
          value={options}
          onChange={(e) => setOptions(e.target.value)}
          className="h-8 text-xs font-mono"
          placeholder="Threads=8, Hash=8192"
        />
      </div>
      <div className="space-y-1.5">
        <Label className="text-xs">Time control (override)</Label>
        <Input
          value={timeControl}
          onChange={(e) => setTimeControl(e.target.value)}
          className="h-8 text-xs font-mono"
          placeholder={`default: ${tcPlaceholder || "900+5"}`}
        />
        <p className="text-[11px] text-muted-foreground">
          Blank inherits the tournament default.
        </p>
      </div>
    </div>
  );
}
