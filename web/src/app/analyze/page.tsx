"use client";

import {
  Suspense,
  useState,
  useCallback,
  useEffect,
  useRef,
  useMemo,
} from "react";
import { useSearchParams } from "next/navigation";
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
import {
  isValidFEN,
  sideToMove,
  lichessAnalysisUrl,
  tryMove,
  isPromotion,
  parseUCIMove,
} from "@/lib/chess-utils";
import {
  FlaskConical,
  RotateCcw,
  Save,
  AlertCircle,
  ExternalLink,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react";

const STARTING_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

interface MoveEntry {
  san: string;
  uci: string;
  fenAfter: string;
}

export default function AnalyzePage() {
  return (
    <Suspense>
      <AnalyzePageContent />
    </Suspense>
  );
}

function AnalyzePageContent() {
  const searchParams = useSearchParams();

  const [startingFen, setStartingFen] = useState(() => {
    const fenParam = searchParams.get("fen");
    return fenParam && isValidFEN(fenParam) ? fenParam : STARTING_FEN;
  });
  const [fenInput, setFenInput] = useState(startingFen);
  const [fenValid, setFenValid] = useState(true);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [moves, setMoves] = useState<MoveEntry[]>([]);
  const [currentPly, setCurrentPly] = useState(0);

  // Promotion state
  const [pendingPromotion, setPendingPromotion] = useState<{
    from: string;
    to: string;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  const boardContainerRef = useRef<HTMLDivElement>(null);
  const moveListRef = useRef<HTMLDivElement>(null);

  const currentFen =
    currentPly === 0 ? startingFen : moves[currentPly - 1].fenAfter;
  const side = sideToMove(currentFen);

  const handleFenChange = useCallback((val: string) => {
    setFenInput(val);
    const valid = isValidFEN(val);
    setFenValid(valid);
    if (valid) {
      setStartingFen(val);
      setMoves([]);
      setCurrentPly(0);
    }
  }, []);

  const executeMove = useCallback(
    (from: string, to: string, promotion?: string) => {
      const result = tryMove(currentFen, from, to, promotion);
      if (!result) return false;

      setMoves((prev) => {
        const truncated = prev.slice(0, currentPly);
        return [
          ...truncated,
          { san: result.san, uci: result.uci, fenAfter: result.newFen },
        ];
      });
      setCurrentPly((p) => p + 1);
      return true;
    },
    [currentFen, currentPly]
  );

  const handlePieceDrop = useCallback(
    ({
      sourceSquare,
      targetSquare,
    }: {
      piece: { isSparePiece: boolean; position: string; pieceType: string };
      sourceSquare: string;
      targetSquare: string | null;
    }) => {
      if (!targetSquare) return false;

      if (isPromotion(currentFen, sourceSquare, targetSquare)) {
        setPendingPromotion({ from: sourceSquare, to: targetSquare });
        return false;
      }

      return executeMove(sourceSquare, targetSquare);
    },
    [currentFen, executeMove]
  );

  const handlePromotionChoice = useCallback(
    (piece: "q" | "r" | "b" | "n") => {
      if (!pendingPromotion) return;
      executeMove(pendingPromotion.from, pendingPromotion.to, piece);
      setPendingPromotion(null);
    },
    [pendingPromotion, executeMove]
  );

  // Navigation
  const goFirst = useCallback(() => setCurrentPly(0), []);
  const goPrev = useCallback(
    () => setCurrentPly((p) => Math.max(0, p - 1)),
    []
  );
  const goNext = useCallback(
    () => setCurrentPly((p) => Math.min(moves.length, p + 1)),
    [moves.length]
  );
  const goLast = useCallback(
    () => setCurrentPly(moves.length),
    [moves.length]
  );

  // Keyboard navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goNext();
      } else if (e.key === "Home") {
        e.preventDefault();
        goFirst();
      } else if (e.key === "End") {
        e.preventDefault();
        goLast();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [goFirst, goPrev, goNext, goLast]);

  // Auto-scroll move list to current ply
  useEffect(() => {
    const el = moveListRef.current?.querySelector(
      `[data-ply="${currentPly}"]`
    ) as HTMLElement | null;
    if (el) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [currentPly]);

  // Highlight last-move squares
  const lastMoveStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};
    if (currentPly > 0) {
      const { from, to } = parseUCIMove(moves[currentPly - 1].uci);
      const highlight = { backgroundColor: "rgba(255, 255, 0, 0.4)" };
      styles[from] = highlight;
      styles[to] = highlight;
    }
    return styles;
  }, [currentPly, moves]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <FlaskConical className="w-6 h-6" />
          Quick Analysis
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Drag and drop pieces to explore lines, or paste a FEN to set the
          starting position.
        </p>
      </div>

      {/* FEN Input */}
      <Card>
        <CardContent className="p-4 space-y-3">
          <div className="grid gap-2">
            <Label htmlFor="fen-input">Starting Position (FEN)</Label>
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

      {/* Board + Move List + Analysis */}
      {fenValid && (
        <div className="grid grid-cols-1 lg:grid-cols-[auto_1fr] gap-6">
          {/* Board Column */}
          <div className="space-y-3">
            <div
              className="flex justify-center lg:justify-start relative"
              ref={boardContainerRef}
            >
              <Chessboard
                options={{
                  id: "analyze-board",
                  position: currentFen,
                  boardOrientation: orientation,
                  allowDragging: !pendingPromotion,
                  onPieceDrop: handlePieceDrop,
                  squareStyles: lastMoveStyles,
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

              {/* Promotion picker overlay */}
              {pendingPromotion && (
                <PromotionPicker
                  color={side}
                  onSelect={handlePromotionChoice}
                  onCancel={() => setPendingPromotion(null)}
                />
              )}
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

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      navigator.clipboard.writeText(currentFen);
                      setCopied(true);
                      setTimeout(() => setCopied(false), 2000);
                    }}
                  >
                    {copied ? (
                      <Check className="w-4 h-4 text-green-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Copy FEN</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" asChild>
                    <a
                      href={lichessAnalysisUrl(currentFen)}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Open in Lichess</TooltipContent>
              </Tooltip>

              <div className="flex-1" />

              <span className="text-xs text-muted-foreground">
                {side === "w" ? "White" : "Black"} to move
              </span>
            </div>

            {/* Navigation controls */}
            <div className="flex items-center justify-center gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={goFirst}
                disabled={currentPly === 0}
                className="px-2"
              >
                <ChevronsLeft className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={goPrev}
                disabled={currentPly === 0}
                className="px-2"
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={goNext}
                disabled={currentPly === moves.length}
                className="px-2"
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={goLast}
                disabled={currentPly === moves.length}
                className="px-2"
              >
                <ChevronsRight className="w-4 h-4" />
              </Button>
              <span className="text-xs text-muted-foreground ml-2">
                {currentPly > 0
                  ? `Ply ${currentPly} / ${moves.length}`
                  : "Starting position"}
              </span>
            </div>

            {/* Save to DB */}
            <AddPositionDialog initialFen={currentFen}>
              <Button variant="outline" className="w-full">
                <Save className="w-4 h-4 mr-1.5" />
                Save to Database
              </Button>
            </AddPositionDialog>
          </div>

          {/* Right Column: Move List + Analysis */}
          <div className="space-y-4">
            {/* Move List */}
            <MoveList
              ref={moveListRef}
              moves={moves}
              currentPly={currentPly}
              startingFen={startingFen}
              onClickPly={setCurrentPly}
            />

            {/* Analysis Panel */}
            <EngineAnalysisPanel fen={currentFen} />
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Move List Component ──────────────────────────────────────────

import { forwardRef } from "react";

const MoveList = forwardRef<
  HTMLDivElement,
  {
    moves: MoveEntry[];
    currentPly: number;
    startingFen: string;
    onClickPly: (ply: number) => void;
  }
>(function MoveList({ moves, currentPly, startingFen, onClickPly }, ref) {
  if (moves.length === 0) {
    return (
      <Card>
        <CardContent className="p-4 text-center text-sm text-muted-foreground">
          Make a move on the board to start exploring.
          <br />
          <span className="text-xs">
            Use arrow keys or the navigation buttons to browse moves.
          </span>
        </CardContent>
      </Card>
    );
  }

  // Figure out the starting move number from the FEN
  const fullMoveNumber = parseInt(startingFen.split(" ")[5] || "1", 10);
  const startingSide = sideToMove(startingFen);

  // Group moves into pairs for display
  const rows: {
    moveNum: number;
    white?: { san: string; ply: number };
    black?: { san: string; ply: number };
  }[] = [];

  let plyIdx = 0;
  let moveNum = fullMoveNumber;

  // If starting side is black, first move is black's
  if (startingSide === "b" && moves.length > 0) {
    rows.push({
      moveNum,
      white: undefined,
      black: { san: moves[0].san, ply: 1 },
    });
    plyIdx = 1;
    moveNum++;
  }

  while (plyIdx < moves.length) {
    const white =
      plyIdx < moves.length
        ? { san: moves[plyIdx].san, ply: plyIdx + 1 }
        : undefined;
    plyIdx++;
    const black =
      plyIdx < moves.length
        ? { san: moves[plyIdx].san, ply: plyIdx + 1 }
        : undefined;
    if (black !== undefined) plyIdx++;
    rows.push({ moveNum, white, black });
    moveNum++;
  }

  return (
    <Card>
      <CardContent className="p-3">
        <div
          ref={ref}
          className="max-h-64 overflow-y-auto font-mono text-sm leading-relaxed"
        >
          <div className="flex flex-wrap gap-x-1 gap-y-0.5">
            {rows.map((row) => (
              <span key={row.moveNum} className="inline-flex items-center">
                <span className="text-muted-foreground w-8 text-right mr-1 tabular-nums select-none">
                  {row.moveNum}.
                </span>
                {row.white ? (
                  <button
                    data-ply={row.white.ply}
                    className={`px-1.5 py-0.5 rounded cursor-pointer hover:bg-muted transition-colors ${
                      currentPly === row.white.ply
                        ? "bg-blue-500/20 text-blue-400 font-semibold"
                        : ""
                    }`}
                    onClick={() => onClickPly(row.white!.ply)}
                  >
                    {row.white.san}
                  </button>
                ) : (
                  <span className="px-1.5 py-0.5 text-muted-foreground">
                    …
                  </span>
                )}
                {row.black && (
                  <button
                    data-ply={row.black.ply}
                    className={`px-1.5 py-0.5 rounded cursor-pointer hover:bg-muted transition-colors ${
                      currentPly === row.black.ply
                        ? "bg-blue-500/20 text-blue-400 font-semibold"
                        : ""
                    }`}
                    onClick={() => onClickPly(row.black!.ply)}
                  >
                    {row.black.san}
                  </button>
                )}
              </span>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

// ─── Promotion Picker ──────────────────────────────────────────────

function PromotionPicker({
  color,
  onSelect,
  onCancel,
}: {
  color: "w" | "b";
  onSelect: (piece: "q" | "r" | "b" | "n") => void;
  onCancel: () => void;
}) {
  const pieces: { key: "q" | "r" | "b" | "n"; label: string; symbol: string }[] =
    color === "w"
      ? [
          { key: "q", label: "Queen", symbol: "♕" },
          { key: "r", label: "Rook", symbol: "♖" },
          { key: "b", label: "Bishop", symbol: "♗" },
          { key: "n", label: "Knight", symbol: "♘" },
        ]
      : [
          { key: "q", label: "Queen", symbol: "♛" },
          { key: "r", label: "Rook", symbol: "♜" },
          { key: "b", label: "Bishop", symbol: "♝" },
          { key: "n", label: "Knight", symbol: "♞" },
        ];

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30 rounded z-20"
        onClick={onCancel}
      />
      {/* Picker */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 bg-card border border-border rounded-lg shadow-2xl p-2 flex gap-1">
        {pieces.map((p) => (
          <button
            key={p.key}
            className="w-16 h-16 flex items-center justify-center text-4xl hover:bg-muted rounded-lg transition-colors cursor-pointer"
            title={p.label}
            onClick={() => onSelect(p.key)}
          >
            {p.symbol}
          </button>
        ))}
      </div>
    </>
  );
}
