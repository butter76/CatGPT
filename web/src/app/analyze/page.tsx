"use client";

import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Chessboard } from "react-chessboard";
import { AddPositionDialog } from "@/components/chess/add-position-dialog";
import {
  PolicyChart,
  WDLBar,
  QValueDisplay,
} from "@/components/chess/policy-chart";
import { EngineAnalysisPanel } from "@/components/chess/engine-analysis-panel";
import { isValidFEN, sideToMove } from "@/lib/chess-utils";
import type { NetworkAnalysis } from "@/lib/types";
import {
  FlaskConical,
  RotateCcw,
  Save,
  Loader2,
  AlertCircle,
} from "lucide-react";

const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Mock analysis function (will be replaced by backend call)
function mockAnalyze(): NetworkAnalysis {
  const moves = ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "g2g3"];
  const probs = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05];

  return {
    policy: moves.map((move, i) => ({ move, probability: probs[i] })),
    wdl: {
      win: 0.35 + Math.random() * 0.2,
      draw: 0.3 + Math.random() * 0.1,
      loss: 0.1 + Math.random() * 0.15,
    },
    bestQ: (Math.random() - 0.3) * 0.6,
    nodes: 1,
    timestamp: new Date().toISOString(),
  };
}

export default function AnalyzePage() {
  const [fen, setFen] = useState(STARTING_FEN);
  const [fenInput, setFenInput] = useState(STARTING_FEN);
  const [fenValid, setFenValid] = useState(true);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [analysis, setAnalysis] = useState<NetworkAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const handleFenChange = useCallback((val: string) => {
    setFenInput(val);
    const valid = isValidFEN(val);
    setFenValid(valid);
    if (valid) {
      setFen(val);
      setAnalysis(null);
    }
  }, []);

  const handleAnalyze = async () => {
    if (!fenValid) return;
    setAnalyzing(true);
    // Simulate network latency
    await new Promise((r) => setTimeout(r, 800));
    setAnalysis(mockAnalyze());
    setAnalyzing(false);
  };

  const side = sideToMove(fen);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <FlaskConical className="w-6 h-6" />
          Quick Analysis
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Paste a FEN to view the position and request analysis. Optionally save
          it to the database.
        </p>
      </div>

      {/* FEN Input */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="grid gap-2">
            <Label htmlFor="fen-input">FEN String</Label>
            <div className="flex gap-2">
              <Input
                id="fen-input"
                value={fenInput}
                onChange={(e) => handleFenChange(e.target.value)}
                placeholder="Paste FEN here..."
                className={`font-mono text-sm flex-1 ${
                  fenInput && !fenValid ? "border-red-500" : ""
                }`}
              />
              <Button
                onClick={handleAnalyze}
                disabled={!fenValid || analyzing}
              >
                {analyzing ? (
                  <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />
                ) : (
                  <FlaskConical className="w-4 h-4 mr-1.5" />
                )}
                Analyze
              </Button>
            </div>
            {fenInput && !fenValid && (
              <p className="text-xs text-red-500 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> Invalid FEN string
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleFenChange(STARTING_FEN)}
            >
              Starting Position
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                navigator.clipboard.readText().then((text) => {
                  handleFenChange(text.trim());
                });
              }}
            >
              Paste from Clipboard
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Board + Analysis */}
      {fenValid && (
        <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-6">
          {/* Board */}
          <div className="space-y-3">
            <div className="flex justify-center lg:justify-start">
              <Chessboard
                options={{
                  id: "analyze-board",
                  position: fen,
                  boardOrientation: orientation,
                  allowDragging: false,
                  boardStyle: {
                    borderRadius: "4px",
                    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
                    width: "480px",
                    height: "480px",
                  },
                  darkSquareStyle: { backgroundColor: "#779952" },
                  lightSquareStyle: { backgroundColor: "#edeed1" },
                }}
              />
            </div>

            {/* Board controls */}
            <div className="flex items-center gap-2">
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

              <div className="flex-1" />

              <span className="text-xs text-muted-foreground">
                {side === "w" ? "White" : "Black"} to move
              </span>
            </div>

            {/* Save to DB */}
            <AddPositionDialog initialFen={fen}>
              <Button variant="outline" className="w-full">
                <Save className="w-4 h-4 mr-1.5" />
                Save to Database
              </Button>
            </AddPositionDialog>
          </div>

          {/* Analysis Panel */}
          <div className="space-y-4">
            {/* Engine Analysis (live) */}
            <EngineAnalysisPanel fen={fen} />

            {/* CatGPT Network Analysis (mock for now) */}
            {analysis ? (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">
                    🧠 CatGPT Network Analysis
                  </CardTitle>
                  <p className="text-xs text-muted-foreground">
                    {analysis.nodes} node{analysis.nodes !== 1 && "s"} •{" "}
                    {new Date(analysis.timestamp).toLocaleTimeString()}
                  </p>
                </CardHeader>
                <CardContent className="space-y-5">
                  <QValueDisplay q={analysis.bestQ} nodes={analysis.nodes} />
                  <Separator />
                  <WDLBar wdl={analysis.wdl} />
                  <Separator />
                  <PolicyChart policy={analysis.policy} fen={fen} />
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="p-12 text-center text-muted-foreground">
                  <FlaskConical className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p className="text-lg font-medium">No CatGPT analysis yet</p>
                  <p className="text-sm mt-1">
                    Click &quot;Analyze&quot; above to request CatGPT network evaluation.
                  </p>
                  <p className="text-xs mt-3 opacity-60">
                    Currently using mock data. CatGPT backend integration coming soon.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
