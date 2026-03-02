"use client";

import { useMemo } from "react";
import { Chessboard } from "react-chessboard";
import type { SharpMoveAnnotation, PolicyEntry } from "@/lib/types";
import { parseUCIMove, sideToMove } from "@/lib/chess-utils";

interface AnalysisBoardProps {
  fen: string;
  /** If provided, show policy-based arrows for the best moves */
  policy?: PolicyEntry[];
  /** If provided, highlight correct/blunder squares for SHARP positions */
  moveAnnotations?: SharpMoveAnnotation[];
  /** Which move to show as the main arrow (best move) */
  bestMove?: string;
  /** Board width in pixels */
  width?: number;
  /** Board orientation */
  orientation?: "white" | "black";
  /** Whether to allow interaction */
  interactive?: boolean;
}

// Color palette for arrows (from strongest to weakest)
const ARROW_COLORS = [
  "rgba(0, 150, 255, 0.8)", // bright blue - top move
  "rgba(0, 150, 255, 0.5)", // medium blue
  "rgba(0, 150, 255, 0.3)", // light blue
  "rgba(0, 150, 255, 0.15)", // very light blue
];

export function AnalysisBoard({
  fen,
  policy,
  moveAnnotations,
  bestMove,
  width = 480,
  orientation = "white",
  interactive = false,
}: AnalysisBoardProps) {
  // Build custom arrows from policy
  const customArrows = useMemo(() => {
    const arrows: Array<{ startSquare: string; endSquare: string; color: string }> = [];

    // Show best move as a prominent arrow
    if (bestMove) {
      const { from, to } = parseUCIMove(bestMove);
      arrows.push({ startSquare: from, endSquare: to, color: "rgba(255, 170, 0, 0.9)" });
    }

    // Show policy arrows (top N moves)
    if (policy && policy.length > 0) {
      const sorted = [...policy].sort((a, b) => b.probability - a.probability);
      const topMoves = sorted.slice(0, 4);
      topMoves.forEach((entry, i) => {
        if (bestMove && entry.move === bestMove) return;
        const { from, to } = parseUCIMove(entry.move);
        arrows.push({
          startSquare: from,
          endSquare: to,
          color: ARROW_COLORS[Math.min(i, ARROW_COLORS.length - 1)],
        });
      });
    }

    return arrows;
  }, [policy, bestMove]);

  // Build square styles for SHARP position annotations
  const customSquareStyles = useMemo(() => {
    const styles: Record<string, React.CSSProperties> = {};

    if (moveAnnotations) {
      for (const ann of moveAnnotations) {
        const { from, to } = parseUCIMove(ann.move);
        if (ann.annotation === "blunder") {
          styles[to] = {
            background:
              "radial-gradient(circle, rgba(255, 50, 50, 0.5) 60%, transparent 60%)",
          };
          styles[from] = {
            background:
              "radial-gradient(circle, rgba(255, 50, 50, 0.25) 60%, transparent 60%)",
          };
        } else if (ann.annotation === "correct") {
          styles[to] = {
            background:
              "radial-gradient(circle, rgba(50, 200, 50, 0.5) 60%, transparent 60%)",
          };
          styles[from] = {
            background:
              "radial-gradient(circle, rgba(50, 200, 50, 0.25) 60%, transparent 60%)",
          };
        }
      }
    }

    return styles;
  }, [moveAnnotations]);

  const side = sideToMove(fen);

  return (
    <div className="relative">
      {/* Side-to-move indicator */}
      <div className="absolute -left-6 top-1/2 -translate-y-1/2 flex flex-col items-center gap-1 z-10">
        <div
          className={`w-3 h-3 rounded-full border border-gray-400 ${
            side === "w" ? "bg-white shadow-md" : "bg-gray-800"
          }`}
          title={side === "w" ? "White to move" : "Black to move"}
        />
      </div>
      <Chessboard
        options={{
          id: "analysis-board",
          position: fen,
          boardOrientation: orientation,
          allowDragging: interactive,
          arrows: customArrows,
          squareStyles: customSquareStyles,
          boardStyle: {
            borderRadius: "4px",
            boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
            width: `${width}px`,
            height: `${width}px`,
          },
          darkSquareStyle: { backgroundColor: "#779952" },
          lightSquareStyle: { backgroundColor: "#edeed1" },
        }}
      />
    </div>
  );
}

// ─── Compact Board (for lists/cards) ──────────────────────────────

interface CompactBoardProps {
  fen: string;
  width?: number;
}

export function CompactBoard({ fen, width = 200 }: CompactBoardProps) {
  return (
    <Chessboard
      options={{
        id: `compact-${fen.slice(0, 20).replace(/[^a-zA-Z0-9]/g, "")}`,
        position: fen,
        allowDragging: false,
        boardStyle: {
          borderRadius: "4px",
          boxShadow: "0 2px 8px rgba(0, 0, 0, 0.15)",
          width: `${width}px`,
          height: `${width}px`,
        },
        darkSquareStyle: { backgroundColor: "#779952" },
        lightSquareStyle: { backgroundColor: "#edeed1" },
      }}
    />
  );
}

// ─── Interactive Board (for analysis page) ────────────────────────

interface InteractiveBoardProps {
  fen: string;
  width?: number;
  orientation?: "white" | "black";
  onPieceDrop?: (args: {
    piece: { isSparePiece: boolean; position: string; pieceType: string };
    sourceSquare: string;
    targetSquare: string | null;
  }) => boolean;
}

export function InteractiveBoard({
  fen,
  width = 480,
  orientation = "white",
  onPieceDrop,
}: InteractiveBoardProps) {
  return (
    <Chessboard
      options={{
        id: "interactive-board",
        position: fen,
        boardOrientation: orientation,
        allowDragging: true,
        onPieceDrop,
        boardStyle: {
          borderRadius: "4px",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
          width: `${width}px`,
          height: `${width}px`,
        },
        darkSquareStyle: { backgroundColor: "#779952" },
        lightSquareStyle: { backgroundColor: "#edeed1" },
      }}
    />
  );
}
