"use client";

import { use, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AnalysisBoard } from "@/components/chess/analysis-board";
import {
  PolicyChart,
  WDLBar,
  QValueDisplay,
} from "@/components/chess/policy-chart";
import {
  MoveAnnotationsList,
  ExpectedOutcomeBadge,
} from "@/components/chess/move-annotations";
import { EngineAnalysisPanel } from "@/components/chess/engine-analysis-panel";
import {
  fetchPosition,
  deletePositionAPI,
  deleteEngineAnalysisAPI,
  updateAnnotationsAPI,
  updatePositionMetaAPI,
} from "@/lib/store";
import { usePositionStore } from "@/lib/store";
import { sideToMove, uciToAlgebraic } from "@/lib/chess-utils";
import type { Position, EngineAnalysis, EngineInfoLine, CatGPTSearchStats, BlunderTag } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ArrowLeft,
  Zap,
  Castle,
  Copy,
  Check,
  RotateCcw,
  Trash2,
  Loader2,
  Pencil,
  Tag,
} from "lucide-react";

export default function PositionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const [position, setPosition] = useState<Position | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [showAnnotations, setShowAnnotations] = useState(true);

  useEffect(() => {
    fetchPosition(id)
      .then(setPosition)
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!position) {
    return (
      <div className="text-center py-16">
        <p className="text-lg text-muted-foreground">Position not found</p>
        <Button
          variant="ghost"
          onClick={() => router.push("/positions")}
          className="mt-4"
        >
          <ArrowLeft className="w-4 h-4 mr-1" /> Back to database
        </Button>
      </div>
    );
  }

  const side = sideToMove(position.fen);
  const bestMove = position.networkAnalysis?.policy[0]?.move;

  const copyFen = () => {
    navigator.clipboard.writeText(position.fen);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDelete = async () => {
    await deletePositionAPI(position.id);
    router.push("/positions");
  };

  return (
    <div className="space-y-6">
      {/* Back + Title */}
      <PositionHeader
        position={position}
        onUpdate={async (updates) => {
          const updated = await updatePositionMetaAPI(position.id, updates);
          setPosition(updated);
        }}
        onDelete={handleDelete}
        onBack={() => router.push("/positions")}
      />

      {/* Main content: Board + Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-6">
        {/* Board Column */}
        <div className="space-y-3">
          <div className="pl-6">
            <AnalysisBoard
              fen={position.fen}
              policy={
                position.networkAnalysis && showAnnotations
                  ? position.networkAnalysis.policy
                  : undefined
              }
              moveAnnotations={
                position.type === "SHARP" && showAnnotations
                  ? position.moveAnnotations
                  : undefined
              }
              bestMove={showAnnotations ? bestMove : undefined}
              width={480}
              orientation={orientation}
            />
          </div>

          {/* Board controls */}
          <div className="flex items-center gap-2 pl-6">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setOrientation((o) =>
                      o === "white" ? "black" : "white"
                    )
                  }
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Flip board</TooltipContent>
            </Tooltip>

            <Button
              variant={showAnnotations ? "secondary" : "outline"}
              size="sm"
              onClick={() => setShowAnnotations(!showAnnotations)}
            >
              {showAnnotations ? "Hide" : "Show"} Annotations
            </Button>

            <div className="flex-1" />

            <span className="text-xs text-muted-foreground">
              {side === "w" ? "White" : "Black"} to move
            </span>
          </div>

          {/* FEN */}
          <div className="pl-6">
            <Card>
              <CardContent className="p-3 flex items-center gap-2">
                <code className="flex-1 text-xs font-mono text-muted-foreground break-all">
                  {position.fen}
                </code>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="sm" onClick={copyFen}>
                      {copied ? (
                        <Check className="w-4 h-4 text-green-500" />
                      ) : (
                        <Copy className="w-4 h-4" />
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Copy FEN</TooltipContent>
                </Tooltip>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Analysis Column */}
        <div className="space-y-4">
          {/* Type-specific section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">
                {position.type === "SHARP"
                  ? "⚡ Sharp Position Analysis"
                  : "🏰 Fortress Evaluation"}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {position.type === "SHARP" && (
                <MoveAnnotationsList
                  annotations={position.moveAnnotations ?? []}
                  fen={position.fen}
                  onSave={async (annotations) => {
                    await updateAnnotationsAPI(position.id, annotations);
                    const refreshed = await fetchPosition(id);
                    if (refreshed) setPosition(refreshed);
                  }}
                />
              )}
              {position.type === "FORTRESS" && position.expectedOutcome && (
                <ExpectedOutcomeBadge outcome={position.expectedOutcome} />
              )}
            </CardContent>
          </Card>

          {/* Network Analysis */}
          {position.networkAnalysis && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">
                  🧠 CatGPT Network Analysis
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  {position.networkAnalysis.nodes} node
                  {position.networkAnalysis.nodes !== 1 && "s"} •{" "}
                  {new Date(
                    position.networkAnalysis.timestamp
                  ).toLocaleDateString()}
                </p>
              </CardHeader>
              <CardContent className="space-y-5">
                <QValueDisplay
                  q={position.networkAnalysis.bestQ}
                  nodes={position.networkAnalysis.nodes}
                />
                <Separator />
                <WDLBar wdl={position.networkAnalysis.wdl} />
                <Separator />
                <PolicyChart
                  policy={position.networkAnalysis.policy}
                  fen={position.fen}
                />
              </CardContent>
            </Card>
          )}

          {/* Live Engine Analysis */}
          <EngineAnalysisPanel
            fen={position.fen}
            positionId={position.id}
            onSaved={() => {
              // Refresh position data after save
              fetchPosition(id).then((p) => p && setPosition(p));
            }}
          />

          {/* Stored Engine Analyses */}
          {position.engineAnalyses && position.engineAnalyses.length > 0 &&
            position.engineAnalyses.map((ea, i) => (
              <StoredEngineResultCard
                key={ea.id ?? i}
                ea={ea}
                fen={position.fen}
                onDelete={async () => {
                  if (ea.id != null) {
                    await deleteEngineAnalysisAPI(ea.id);
                    const refreshed = await fetchPosition(id);
                    if (refreshed) setPosition(refreshed);
                  }
                }}
              />
            ))
          }

          {!position.networkAnalysis && (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground">
                <p>No analysis available yet.</p>
                <p className="text-sm mt-1">
                  Analysis will be available when the backend is connected.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Editable Position Header ─────────────────────────────────────

function PositionHeader({
  position,
  onUpdate,
  onDelete,
  onBack,
}: {
  position: Position;
  onUpdate: (updates: { name?: string; description?: string | null; blunderTag?: BlunderTag | null }) => Promise<void>;
  onDelete: () => void;
  onBack: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(position.name);
  const [description, setDescription] = useState(position.description ?? "");
  const [saving, setSaving] = useState(false);

  const startEditing = () => {
    setName(position.name);
    setDescription(position.description ?? "");
    setEditing(true);
  };

  const cancel = () => {
    setName(position.name);
    setDescription(position.description ?? "");
    setEditing(false);
  };

  const save = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await onUpdate({
        name: name.trim(),
        description: description.trim() || null,
      });
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex items-start justify-between gap-4">
      <div className="space-y-1 flex-1">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ArrowLeft className="w-4 h-4 mr-1" /> Back
        </Button>

        {editing ? (
          <div className="space-y-2 max-w-2xl">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="text-xl font-bold h-10"
              placeholder="Position name"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") save();
                if (e.key === "Escape") cancel();
              }}
            />
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="text-sm resize-none"
              placeholder="Description (optional)"
              rows={2}
              onKeyDown={(e) => {
                if (e.key === "Escape") cancel();
              }}
            />
            <div className="flex gap-2">
              <Button size="sm" onClick={save} disabled={!name.trim() || saving}>
                {saving && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                Save
              </Button>
              <Button size="sm" variant="ghost" onClick={cancel} disabled={saving}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold">{position.name}</h1>
              <Badge
                variant="outline"
                className={`text-xs ${
                  position.type === "SHARP"
                    ? "border-amber-500 text-amber-600"
                    : "border-blue-500 text-blue-600"
                }`}
              >
                {position.type === "SHARP" ? (
                  <Zap className="w-3 h-3 mr-0.5" />
                ) : (
                  <Castle className="w-3 h-3 mr-0.5" />
                )}
                {position.type}
              </Badge>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 text-muted-foreground"
                onClick={startEditing}
              >
                <Pencil className="w-3.5 h-3.5" />
              </Button>
            </div>
            {position.description && (
              <p className="text-sm text-muted-foreground max-w-2xl">
                {position.description}
              </p>
            )}
            {!position.description && (
              <button
                className="text-xs text-muted-foreground hover:text-foreground italic cursor-pointer"
                onClick={startEditing}
              >
                + Add description
              </button>
            )}
            {/* Blunder Tag Selector */}
            <BlunderTagSelector
              value={position.blunderTag}
              onChange={(tag) => onUpdate({ blunderTag: tag })}
            />
          </>
        )}
      </div>

      {!editing && (
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:text-destructive"
          onClick={onDelete}
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      )}
    </div>
  );
}

// ─── Blunder Tag Selector ─────────────────────────────────────────

const BLUNDER_TAG_OPTIONS: { value: BlunderTag; label: string; color: string }[] = [
  { value: "catgpt", label: "🐱 CatGPT Blunder", color: "border-orange-500 text-orange-600 bg-orange-50" },
  { value: "stockfish", label: "🐟 Stockfish Blunder", color: "border-green-500 text-green-600 bg-green-50" },
  { value: "leela", label: "♟️ Leela Blunder", color: "border-purple-500 text-purple-600 bg-purple-50" },
];

function BlunderTagSelector({
  value,
  onChange,
}: {
  value?: BlunderTag;
  onChange: (tag: BlunderTag | null) => void;
}) {
  return (
    <div className="flex items-center gap-2 mt-2">
      <Tag className="w-3.5 h-3.5 text-muted-foreground" />
      <Select
        value={value ?? "none"}
        onValueChange={(v) => onChange(v === "none" ? null : (v as BlunderTag))}
      >
        <SelectTrigger className="h-7 w-[180px] text-xs">
          <SelectValue placeholder="Tag as blunder..." />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="none" className="text-xs text-muted-foreground">
            No tag
          </SelectItem>
          {BLUNDER_TAG_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value} className="text-xs">
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {value && (
        <Badge
          variant="outline"
          className={`text-xs ${
            BLUNDER_TAG_OPTIONS.find((o) => o.value === value)?.color ?? ""
          }`}
        >
          {BLUNDER_TAG_OPTIONS.find((o) => o.value === value)?.label}
        </Badge>
      )}
    </div>
  );
}

// ─── Stored Engine Result with Depth History ──────────────────────

function StoredEngineResultCard({
  ea,
  fen,
  onDelete,
}: {
  ea: EngineAnalysis;
  fen: string;
  onDelete: () => Promise<void>;
}) {
  const [deleting, setDeleting] = useState(false);
  const { notationFormat } = usePositionStore();

  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  const fmtEval = (scoreType: string, value: number) => {
    if (scoreType === "mate") return `M${value}`;
    const cp = value / 100;
    return cp >= 0 ? `+${cp.toFixed(2)}` : cp.toFixed(2);
  };

  const engineLabel =
    ea.engine === "catgpt"
      ? "🐱 CatGPT (Fractional)"
      : ea.engine === "catgpt_mcts"
      ? "🐱 CatGPT (MCTS)"
      : ea.engine === "stockfish"
      ? "Stockfish"
      : "Leela Chess Zero";

  const hasUCIHistory = ea.depthHistory && ea.depthHistory.length > 0;
  const hasCatGPTHistory = ea.catgptHistory && ea.catgptHistory.length > 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center justify-between">
          <span>📊 {engineLabel}</span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground font-normal">
              {new Date(ea.timestamp).toLocaleDateString()}
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
              disabled={deleting}
              onClick={async () => {
                setDeleting(true);
                await onDelete();
                setDeleting(false);
              }}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          {ea.engine === "catgpt" || ea.engine === "catgpt_mcts"
            ? `${ea.nodes.toLocaleString()} evals • iter ${ea.depth}`
            : `depth ${ea.depth} • ${ea.nodes.toLocaleString()} nodes`}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Summary */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`text-2xl font-bold font-mono ${
                ea.evaluation >= 0 ? "text-green-600" : "text-red-600"
              }`}
            >
              {fmtEval("cp", ea.evaluation)}
            </span>
            <Badge variant="outline" className="font-mono">
              {fmtMove(ea.bestMove)}
            </Badge>
          </div>
        </div>

        {/* PV (UCI engines only) */}
        {ea.engine !== "catgpt" && ea.engine !== "catgpt_mcts" && ea.pv.length > 0 && (
          <div className="text-xs font-mono text-muted-foreground break-all">
            <span className="text-foreground font-medium">{fmtMove(ea.pv[0])}</span>
            {ea.pv.slice(1, 10).map((m, i) => (
              <span key={i}>{" "}{m}</span>
            ))}
            {ea.pv.length > 10 && " ..."}
          </div>
        )}

        {/* UCI Depth History (interactive) */}
        {hasUCIHistory && (
          <StoredUCIHistoryViewer
            history={ea.depthHistory!}
            fen={fen}
          />
        )}

        {/* CatGPT: Interactive search history + details viewer */}
        {hasCatGPTHistory && (
          <StoredCatGPTHistoryViewer
            history={ea.catgptHistory!}
            fen={fen}
          />
        )}
      </CardContent>
    </Card>
  );
}

// ─── Stored UCI history viewer (selectable snapshots) ─────────────

function StoredUCIHistoryViewer({
  history,
  fen,
}: {
  history: EngineInfoLine[];
  fen: string;
}) {
  // Default to the last entry (deepest search)
  const [selectedIdx, setSelectedIdx] = useState(history.length - 1);
  const { notationFormat } = usePositionStore();
  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  const fmtEval = (scoreType: string, value: number) => {
    if (scoreType === "mate") return `M${value}`;
    const cp = value / 100;
    return cp >= 0 ? `+${cp.toFixed(2)}` : cp.toFixed(2);
  };

  const displayInfo = history[selectedIdx];
  if (!displayInfo) return null;

  return (
    <div className="space-y-3">
      <Separator />

      {/* Viewing indicator */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Viewing:</span>
        <Badge variant="outline" className="text-xs font-mono">
          depth {displayInfo.depth} ({selectedIdx + 1}/{history.length})
        </Badge>
        <span className="text-xs text-muted-foreground">
          {displayInfo.nodes >= 1000
            ? `${(displayInfo.nodes / 1000).toFixed(0)}k`
            : displayInfo.nodes}{" "}
          nodes
          {displayInfo.nps
            ? ` • ${(displayInfo.nps / 1000).toFixed(0)}k nps`
            : ""}
          {displayInfo.time != null
            ? ` • ${(displayInfo.time / 1000).toFixed(1)}s`
            : ""}
        </span>
      </div>

      {/* PV for selected depth */}
      {displayInfo.pv.length > 0 && (
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
            Principal Variation
          </span>
          <div className="text-xs font-mono text-muted-foreground break-all">
            {displayInfo.pv.slice(0, 16).map((m, i) => (
              <span key={i}>
                {i > 0 && " "}
                <span
                  className={i === 0 ? "text-foreground font-medium" : ""}
                >
                  {m}
                </span>
              </span>
            ))}
            {displayInfo.pv.length > 16 && " …"}
          </div>
        </div>
      )}

      {/* WDL for selected depth (Leela) */}
      {displayInfo.wdl && (
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
            WDL
          </span>
          <div className="flex h-5 rounded overflow-hidden text-[10px] font-medium">
            <div
              className="bg-green-500 flex items-center justify-center text-white"
              style={{ width: `${displayInfo.wdl.win / 10}%` }}
            >
              {displayInfo.wdl.win > 80 &&
                `${(displayInfo.wdl.win / 10).toFixed(0)}%`}
            </div>
            <div
              className="bg-gray-400 flex items-center justify-center text-white"
              style={{ width: `${displayInfo.wdl.draw / 10}%` }}
            >
              {displayInfo.wdl.draw > 80 &&
                `${(displayInfo.wdl.draw / 10).toFixed(0)}%`}
            </div>
            <div
              className="bg-red-500 flex items-center justify-center text-white"
              style={{ width: `${displayInfo.wdl.loss / 10}%` }}
            >
              {displayInfo.wdl.loss > 80 &&
                `${(displayInfo.wdl.loss / 10).toFixed(0)}%`}
            </div>
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span className="text-green-600">
              W {(displayInfo.wdl.win / 10).toFixed(1)}%
            </span>
            <span>D {(displayInfo.wdl.draw / 10).toFixed(1)}%</span>
            <span className="text-red-600">
              L {(displayInfo.wdl.loss / 10).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Clickable depth history table */}
      {history.length > 1 && (
        <>
          <Separator />
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
              Depth History
              <span className="ml-1 font-normal normal-case">(click to inspect)</span>
            </span>
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full text-xs font-mono">
                <thead className="text-muted-foreground sticky top-0 bg-card">
                  <tr>
                    <th className="text-left py-1 pr-2">Depth</th>
                    <th className="text-right py-1 pr-2">Eval</th>
                    <th className="text-left py-1 pr-2">Best</th>
                    <th className="text-left py-1 pr-2">PV</th>
                    <th className="text-right py-1">Nodes</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((info, i) => {
                    const prevBest =
                      i > 0 ? history[i - 1].pv[0] : null;
                    const bestChanged =
                      prevBest !== null && info.pv[0] !== prevBest;
                    const isSelected = selectedIdx === i;
                    return (
                      <tr
                        key={i}
                        className={`cursor-pointer hover:bg-muted/50 ${
                          isSelected
                            ? "bg-muted ring-1 ring-blue-500/40 text-foreground font-medium"
                            : bestChanged
                            ? "text-amber-500 font-medium"
                            : i === history.length - 1
                            ? "text-foreground font-medium"
                            : "text-muted-foreground"
                        }`}
                        onClick={() => setSelectedIdx(i)}
                      >
                        <td className="py-0.5 pr-2">{info.depth}</td>
                        <td
                          className={`text-right py-0.5 pr-2 ${
                            info.score.value >= 0
                              ? "text-green-600"
                              : "text-red-600"
                          }`}
                        >
                          {fmtEval(info.score.type, info.score.value)}
                        </td>
                        <td className="py-0.5 pr-2">
                          <span className={bestChanged ? "underline" : ""}>
                            {info.pv[0] ? fmtMove(info.pv[0]) : "-"}
                          </span>
                        </td>
                        <td className="py-0.5 pr-2 text-muted-foreground truncate max-w-[160px]">
                          {info.pv.slice(1, 5).join(" ")}
                          {info.pv.length > 5 && " …"}
                        </td>
                        <td className="text-right py-0.5">
                          {info.nodes >= 1000
                            ? `${(info.nodes / 1000).toFixed(0)}k`
                            : info.nodes}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── Stored CatGPT history viewer (selectable snapshots) ──────────

function StoredCatGPTHistoryViewer({
  history,
  fen,
}: {
  history: CatGPTSearchStats[];
  fen: string;
}) {
  // Default to the last entry (final search result)
  const [selectedIdx, setSelectedIdx] = useState(history.length - 1);
  const { notationFormat } = usePositionStore();
  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  const displayStats = history[selectedIdx];
  if (!displayStats) return null;

  const typeLabel = (t: string) =>
    t === "root_eval" ? "root" : t === "search_update" ? "update" : "done";

  const topPolicy = [...displayStats.policy]
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 8);
  const maxWeight = topPolicy.length > 0 ? topPolicy[0].weight : 1;

  return (
    <div className="space-y-3">
      <Separator />

      {/* Viewing indicator */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Viewing:</span>
        <Badge variant="outline" className="text-xs font-mono">
          {typeLabel(displayStats.type)} #{selectedIdx + 1}/{history.length}
        </Badge>
        <span className="text-xs text-muted-foreground">
          {displayStats.nodes} evals • iter {displayStats.iteration}
        </span>
      </div>

      {/* Modified Policy with Q values */}
      <div className="space-y-1.5">
        <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
          Modified Policy
        </span>
        {topPolicy.map((entry) => {
          const label = fmtMove(entry.move);
          const pct = (entry.weight * 100).toFixed(1);
          const barWidth = (entry.weight / maxWeight) * 100;
          const isBest = entry.move === displayStats.bestMove;
          return (
            <div key={entry.move} className="space-y-0">
              <div className="flex items-center gap-2">
                <span
                  className={`w-12 text-right font-mono text-xs ${
                    isBest ? "font-bold text-foreground" : ""
                  }`}
                >
                  {label}
                </span>
                <div className="flex-1 h-4 bg-muted rounded overflow-hidden">
                  <div
                    className={`h-full rounded transition-all ${
                      isBest ? "bg-amber-500" : "bg-blue-500"
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
                <span className="w-12 text-right text-[10px] text-muted-foreground font-mono">
                  {pct}%
                </span>
              </div>
              {entry.q != null && (
                <div className="flex items-center gap-2 ml-14">
                  <span
                    className={`text-[10px] font-mono ${
                      entry.q >= 0 ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    Q {entry.q >= 0 ? "+" : ""}{entry.q.toFixed(3)} ({(() => {
                      const cp = 100.7066 * Math.tan(entry.q * 1.5637541897);
                      return cp >= 0 ? `+${(cp/100).toFixed(2)}` : (cp/100).toFixed(2);
                    })()})
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* DistQ mini histogram */}
      {displayStats.distQ && displayStats.distQ.length > 0 && (
        <div className="space-y-1">
          <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
            Value Distribution
          </span>
          <div className="flex items-end gap-px h-12 bg-muted/30 rounded p-1">
            {(() => {
              const maxProb = Math.max(...displayStats.distQ, 0.001);
              return displayStats.distQ.map((prob, i) => {
                const height = (prob / maxProb) * 100;
                const t = i / Math.max(displayStats.distQ.length - 1, 1);
                const r = Math.round(239 * (1 - t) + 34 * t);
                const g = Math.round(68 * (1 - t) + 197 * t);
                const b = Math.round(68 * (1 - t) + 94 * t);
                return (
                  <div
                    key={i}
                    className="flex-1 rounded-t"
                    style={{
                      height: `${Math.max(height, 1)}%`,
                      backgroundColor: `rgb(${r},${g},${b})`,
                      opacity: prob > 0.001 ? 1 : 0.2,
                    }}
                    title={`Bin ${i}: ${(prob * 100).toFixed(1)}%`}
                  />
                );
              });
            })()}
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
            <span>Loss</span>
            <span>Draw</span>
            <span>Win</span>
          </div>
        </div>
      )}

      {/* Search history table (clickable) */}
      {history.length > 1 && (
        <>
          <Separator />
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wide">
              Search History
              <span className="ml-1 font-normal normal-case">(click to inspect)</span>
            </span>
            <div className="max-h-48 overflow-y-auto">
              <table className="w-full text-xs font-mono">
                <thead className="text-muted-foreground sticky top-0 bg-card">
                  <tr>
                    <th className="text-left py-1 pr-2">Type</th>
                    <th className="text-right py-1 pr-2">Eval</th>
                    <th className="text-left py-1 pr-2">Best</th>
                    <th className="text-right py-1">Evals</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((s, i) => {
                    const prevBest =
                      i > 0 ? history[i - 1].bestMove : null;
                    const changed =
                      prevBest !== null && s.bestMove !== prevBest;
                    const isSelected = selectedIdx === i;
                    return (
                      <tr
                        key={i}
                        className={`cursor-pointer hover:bg-muted/50 ${
                          isSelected
                            ? "bg-muted ring-1 ring-blue-500/40 text-foreground font-medium"
                            : changed
                            ? "text-amber-500 font-medium"
                            : i === history.length - 1
                            ? "text-foreground font-medium"
                            : "text-muted-foreground"
                        }`}
                        onClick={() => setSelectedIdx(i)}
                      >
                        <td className="py-0.5 pr-2 text-xs">
                          {typeLabel(s.type)}
                        </td>
                        <td
                          className={`text-right py-0.5 pr-2 ${
                            s.cp >= 0 ? "text-green-600" : "text-red-600"
                          }`}
                        >
                          {(s.cp / 100).toFixed(2)}
                        </td>
                        <td className="py-0.5 pr-2">
                          <span className={changed ? "underline" : ""}>
                            {fmtMove(s.bestMove)}
                          </span>
                        </td>
                        <td className="text-right py-0.5">{s.nodes}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
