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
import type { Position, EngineAnalysis } from "@/lib/types";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
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
  onUpdate: (updates: { name?: string; description?: string | null }) => Promise<void>;
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
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const { notationFormat } = usePositionStore();

  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  const fmtEval = (scoreType: string, value: number) => {
    if (scoreType === "mate") return `M${value}`;
    const cp = value / 100;
    return cp >= 0 ? `+${cp.toFixed(2)}` : cp.toFixed(2);
  };

  const hasHistory = ea.depthHistory && ea.depthHistory.length > 0;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center justify-between">
          <span>
            📊 {ea.engine === "stockfish" ? "Stockfish" : "Leela Chess Zero"}
          </span>
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
          depth {ea.depth} • {ea.nodes.toLocaleString()} nodes
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

        {/* PV */}
        {ea.pv.length > 0 && (
          <div className="text-xs font-mono text-muted-foreground break-all">
            <span className="text-foreground font-medium">{fmtMove(ea.pv[0])}</span>
            {ea.pv.slice(1, 10).map((m, i) => (
              <span key={i}>{" "}{m}</span>
            ))}
            {ea.pv.length > 10 && " ..."}
          </div>
        )}

        {/* Depth History */}
        {hasHistory && (
          <>
            <Separator />
            <Button
              variant="ghost"
              size="sm"
              className="w-full text-xs"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "Hide" : "Show"} depth history ({ea.depthHistory!.length} entries)
            </Button>
            {expanded && (
              <div className="max-h-64 overflow-y-auto">
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
                    {ea.depthHistory!.map((info, i) => {
                      const prevBest =
                        i > 0 ? ea.depthHistory![i - 1].pv[0] : null;
                      const bestChanged =
                        prevBest !== null && info.pv[0] !== prevBest;
                      return (
                        <tr
                          key={i}
                          className={
                            bestChanged
                              ? "text-amber-500 font-medium"
                              : i === ea.depthHistory!.length - 1
                              ? "text-foreground font-medium"
                              : "text-muted-foreground"
                          }
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
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
