"use client";

import { useState } from "react";
import type { SharpMoveAnnotation, Outcome } from "@/lib/types";
import { uciToAlgebraic, algebraicToUCI, getLegalMoves } from "@/lib/chess-utils";
import { usePositionStore } from "@/lib/store";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  CheckCircle2,
  XCircle,
  Trophy,
  Minus,
  Skull,
  HelpCircle,
  Pencil,
  X,
  Plus,
  Loader2,
} from "lucide-react";

// ─── Move Annotations List (SHARP) ───────────────────────────────

interface MoveAnnotationsProps {
  annotations: SharpMoveAnnotation[];
  fen: string;
  /** If provided, enables editing */
  onSave?: (annotations: SharpMoveAnnotation[]) => Promise<void>;
}

export function MoveAnnotationsList({ annotations, fen, onSave }: MoveAnnotationsProps) {
  const { notationFormat } = usePositionStore();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<SharpMoveAnnotation[]>(annotations);
  const [moveInput, setMoveInput] = useState("");
  const [saving, setSaving] = useState(false);

  const correct = (editing ? draft : annotations).filter((a) => a.annotation === "correct");
  const blunders = (editing ? draft : annotations).filter((a) => a.annotation === "blunder");

  const startEditing = () => {
    setDraft([...annotations]);
    setEditing(true);
  };

  const cancelEditing = () => {
    setDraft([...annotations]);
    setMoveInput("");
    setEditing(false);
  };

  const handleSave = async () => {
    if (!onSave) return;
    setSaving(true);
    try {
      await onSave(draft);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  const addMove = (type: "correct" | "blunder") => {
    if (!moveInput.trim()) return;

    let uciMove = moveInput.trim();
    if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uciMove)) {
      const parsed = algebraicToUCI(fen, uciMove);
      if (!parsed) return;
      uciMove = parsed;
    }

    const legalMoves = getLegalMoves(fen);
    if (!legalMoves.includes(uciMove)) return;
    if (draft.some((a) => a.move === uciMove)) return;

    setDraft([...draft, { move: uciMove, annotation: type }]);
    setMoveInput("");
  };

  const removeMove = (move: string) => {
    setDraft(draft.filter((a) => a.move !== move));
  };

  const fmtMove = (move: string) =>
    notationFormat === "algebraic" ? uciToAlgebraic(fen, move) : move;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
          Key Moves
        </h4>
        {onSave && !editing && (
          <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={startEditing}>
            <Pencil className="w-3 h-3 mr-1" />
            Edit
          </Button>
        )}
      </div>

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
                className={`border-green-500 text-green-700 bg-green-50 dark:bg-green-950 dark:text-green-400 font-mono ${
                  editing ? "cursor-pointer hover:line-through" : ""
                }`}
                onClick={editing ? () => removeMove(a.move) : undefined}
              >
                {fmtMove(a.move)}
                {editing && <X className="w-3 h-3 ml-1" />}
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
                className={`border-red-500 text-red-700 bg-red-50 dark:bg-red-950 dark:text-red-400 font-mono ${
                  editing ? "cursor-pointer hover:line-through" : ""
                }`}
                onClick={editing ? () => removeMove(a.move) : undefined}
              >
                {fmtMove(a.move)}
                {editing && <X className="w-3 h-3 ml-1" />}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {editing && correct.length === 0 && blunders.length === 0 && (
        <p className="text-xs text-muted-foreground italic">No annotations yet. Add some below.</p>
      )}

      {/* Edit controls */}
      {editing && (
        <div className="space-y-2 p-3 bg-muted rounded-lg">
          <div className="flex gap-2">
            <Input
              placeholder={notationFormat === "algebraic" ? "e.g. Nf3" : "e.g. g1f3"}
              value={moveInput}
              onChange={(e) => setMoveInput(e.target.value)}
              className="font-mono h-8 text-sm"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  addMove("correct");
                }
              }}
            />
            <Button
              size="sm"
              variant="outline"
              className="h-8 border-green-500 text-green-700 hover:bg-green-50 dark:hover:bg-green-950"
              onClick={() => addMove("correct")}
            >
              <Plus className="w-3 h-3 mr-1" />
              Correct
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-8 border-red-500 text-red-700 hover:bg-red-50 dark:hover:bg-red-950"
              onClick={() => addMove("blunder")}
            >
              <Plus className="w-3 h-3 mr-1" />
              Blunder
            </Button>
          </div>
          <div className="flex gap-2 justify-end">
            <Button size="sm" variant="ghost" onClick={cancelEditing} disabled={saving}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleSave} disabled={saving}>
              {saving ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : null}
              Save
            </Button>
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
    unknown: {
      icon: HelpCircle,
      label: "Unknown",
      className: "border-yellow-500 text-yellow-700 bg-yellow-50 dark:bg-yellow-950 dark:text-yellow-400",
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
