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
import { fetchPosition, deletePositionAPI } from "@/lib/store";
import { sideToMove } from "@/lib/chess-utils";
import type { Position } from "@/lib/types";
import {
  ArrowLeft,
  Zap,
  Castle,
  Copy,
  Check,
  RotateCcw,
  Trash2,
  Loader2,
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
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push("/positions")}
          >
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            {position.name}
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
          </h1>
          {position.description && (
            <p className="text-sm text-muted-foreground max-w-2xl">
              {position.description}
            </p>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:text-destructive"
          onClick={handleDelete}
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>

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
              {position.type === "SHARP" && position.moveAnnotations && (
                <MoveAnnotationsList
                  annotations={position.moveAnnotations}
                  fen={position.fen}
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

          {/* Engine Analysis (future) */}
          {position.engineAnalyses && position.engineAnalyses.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">🔧 Engine Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                {position.engineAnalyses.map((ea, i) => (
                  <div key={i} className="text-sm">
                    <span className="font-medium capitalize">{ea.engine}</span>:{" "}
                    depth {ea.depth}, eval {ea.evaluation}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

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
