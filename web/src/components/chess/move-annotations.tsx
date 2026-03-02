"use client";

import type { SharpMoveAnnotation, Outcome } from "@/lib/types";
import { uciToAlgebraic } from "@/lib/chess-utils";
import { usePositionStore } from "@/lib/store";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, XCircle, Trophy, Minus, Skull } from "lucide-react";

// ─── Move Annotations List (SHARP) ───────────────────────────────

interface MoveAnnotationsProps {
  annotations: SharpMoveAnnotation[];
  fen: string;
}

export function MoveAnnotationsList({ annotations, fen }: MoveAnnotationsProps) {
  const { notationFormat } = usePositionStore();
  const correct = annotations.filter((a) => a.annotation === "correct");
  const blunders = annotations.filter((a) => a.annotation === "blunder");

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Key Moves
      </h4>

      {correct.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-green-600">
            <CheckCircle2 className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase">Correct Moves</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {correct.map((a) => (
              <Badge
                key={a.move}
                variant="outline"
                className="border-green-500 text-green-700 bg-green-50 dark:bg-green-950 dark:text-green-400 font-mono"
              >
                {notationFormat === "algebraic"
                  ? uciToAlgebraic(fen, a.move)
                  : a.move}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {blunders.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-red-600">
            <XCircle className="w-4 h-4" />
            <span className="text-xs font-semibold uppercase">Blunders</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {blunders.map((a) => (
              <Badge
                key={a.move}
                variant="outline"
                className="border-red-500 text-red-700 bg-red-50 dark:bg-red-950 dark:text-red-400 font-mono"
              >
                {notationFormat === "algebraic"
                  ? uciToAlgebraic(fen, a.move)
                  : a.move}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Expected Outcome Badge (FORTRESS) ───────────────────────────

interface ExpectedOutcomeBadgeProps {
  outcome: Outcome;
}

export function ExpectedOutcomeBadge({ outcome }: ExpectedOutcomeBadgeProps) {
  const config = {
    win: {
      icon: Trophy,
      label: "Decisive — Win",
      className: "border-green-500 text-green-700 bg-green-50 dark:bg-green-950 dark:text-green-400",
    },
    loss: {
      icon: Skull,
      label: "Decisive — Loss",
      className: "border-red-500 text-red-700 bg-red-50 dark:bg-red-950 dark:text-red-400",
    },
    draw: {
      icon: Minus,
      label: "Drawn",
      className: "border-gray-500 text-gray-700 bg-gray-50 dark:bg-gray-800 dark:text-gray-300",
    },
  }[outcome];

  const Icon = config.icon;

  return (
    <div className="space-y-1.5">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Expected Outcome
      </h4>
      <Badge variant="outline" className={`${config.className} text-sm px-3 py-1`}>
        <Icon className="w-4 h-4 mr-1.5" />
        {config.label}
      </Badge>
    </div>
  );
}
