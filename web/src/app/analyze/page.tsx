"use client";

import { useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Chessboard } from "react-chessboard";
import { AddPositionDialog } from "@/components/chess/add-position-dialog";
import { EngineAnalysisPanel } from "@/components/chess/engine-analysis-panel";
import { isValidFEN, sideToMove } from "@/lib/chess-utils";
import {
  FlaskConical,
  RotateCcw,
  Save,
  AlertCircle,
} from "lucide-react";

const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

export default function AnalyzePage() {
  const [fen, setFen] = useState(STARTING_FEN);
  const [fenInput, setFenInput] = useState(STARTING_FEN);
  const [fenValid, setFenValid] = useState(true);
  const [orientation, setOrientation] = useState<"white" | "black">("white");

  const handleFenChange = useCallback((val: string) => {
    setFenInput(val);
    const valid = isValidFEN(val);
    setFenValid(valid);
    if (valid) {
      setFen(val);
    }
  }, []);

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
            <Input
              id="fen-input"
              value={fenInput}
              onChange={(e) => handleFenChange(e.target.value)}
              placeholder="Paste FEN here..."
              className={`font-mono text-sm ${
                fenInput && !fenValid ? "border-red-500" : ""
              }`}
            />
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
          <div>
            <EngineAnalysisPanel fen={fen} />
          </div>
        </div>
      )}
    </div>
  );
}
