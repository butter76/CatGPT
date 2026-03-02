"use client";

import { useState, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Chessboard } from "react-chessboard";
import { isValidFEN, algebraicToUCI, getLegalMoves, uciToAlgebraic } from "@/lib/chess-utils";
import { usePositionStore } from "@/lib/store";
import type { PositionType, Outcome, SharpMoveAnnotation } from "@/lib/types";
import { Plus, X, CheckCircle2, XCircle } from "lucide-react";

interface AddPositionDialogProps {
  /** Pre-fill the FEN (e.g. from the analysis page) */
  initialFen?: string;
  /** Trigger element */
  children: React.ReactNode;
  /** Called after adding */
  onAdded?: () => void;
}

export function AddPositionDialog({
  initialFen,
  children,
  onAdded,
}: AddPositionDialogProps) {
  const { addPosition, notationFormat } = usePositionStore();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [fen, setFen] = useState(initialFen ?? "");
  const [type, setType] = useState<PositionType>("SHARP");
  const [expectedOutcome, setExpectedOutcome] = useState<Outcome>("draw");
  const [annotations, setAnnotations] = useState<SharpMoveAnnotation[]>([]);
  const [moveInput, setMoveInput] = useState("");
  const [fenValid, setFenValid] = useState(!!initialFen && isValidFEN(initialFen));

  const handleFenChange = useCallback((val: string) => {
    setFen(val);
    setFenValid(isValidFEN(val));
    setAnnotations([]);
  }, []);

  const addAnnotation = (annotationType: "correct" | "blunder") => {
    if (!moveInput.trim() || !fenValid) return;

    let uciMove = moveInput.trim();
    // Try to parse as algebraic if it doesn't look like UCI
    if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uciMove)) {
      const parsed = algebraicToUCI(fen, uciMove);
      if (!parsed) return; // invalid move
      uciMove = parsed;
    }

    // Check it's a legal move
    const legalMoves = getLegalMoves(fen);
    if (!legalMoves.includes(uciMove)) return;

    // Don't add duplicates
    if (annotations.some((a) => a.move === uciMove)) return;

    setAnnotations([...annotations, { move: uciMove, annotation: annotationType }]);
    setMoveInput("");
  };

  const removeAnnotation = (move: string) => {
    setAnnotations(annotations.filter((a) => a.move !== move));
  };

  const handleSubmit = () => {
    if (!name.trim() || !fenValid) return;

    const now = new Date().toISOString();
    addPosition({
      id: `pos-${Date.now()}`,
      name: name.trim(),
      description: description.trim() || undefined,
      type,
      fen,
      moveAnnotations: type === "SHARP" ? annotations : undefined,
      expectedOutcome: type === "FORTRESS" ? expectedOutcome : undefined,
      createdAt: now,
      updatedAt: now,
    });

    // Reset
    setName("");
    setDescription("");
    setFen("");
    setType("SHARP");
    setAnnotations([]);
    setMoveInput("");
    setOpen(false);
    onAdded?.();
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add New Position</DialogTitle>
          <DialogDescription>
            Add a position to the database with FEN notation, classification, and optional move annotations.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-2">
          {/* Name */}
          <div className="grid gap-2">
            <Label htmlFor="name">Position Name *</Label>
            <Input
              id="name"
              placeholder="e.g. Fried Liver Trap"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          {/* Description */}
          <div className="grid gap-2">
            <Label htmlFor="desc">Description</Label>
            <Textarea
              id="desc"
              placeholder="Optional description of the position..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
            />
          </div>

          {/* FEN */}
          <div className="grid gap-2">
            <Label htmlFor="fen">FEN *</Label>
            <Input
              id="fen"
              placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
              value={fen}
              onChange={(e) => handleFenChange(e.target.value)}
              className={`font-mono text-sm ${
                fen && !fenValid ? "border-red-500" : ""
              }`}
            />
            {fen && !fenValid && (
              <p className="text-xs text-red-500">Invalid FEN string</p>
            )}
          </div>

          {/* Board preview */}
          {fenValid && (
            <div className="flex justify-center">
              <Chessboard
                options={{
                  id: "preview-board",
                  position: fen,
                  allowDragging: false,
                  darkSquareStyle: { backgroundColor: "#779952" },
                  lightSquareStyle: { backgroundColor: "#edeed1" },
                  boardStyle: {
                    borderRadius: "4px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                    width: "240px",
                    height: "240px",
                  },
                }}
              />
            </div>
          )}

          {/* Type */}
          <div className="grid gap-2">
            <Label>Position Type</Label>
            <Select value={type} onValueChange={(v) => setType(v as PositionType)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="SHARP">⚡ SHARP — Move-critical position</SelectItem>
                <SelectItem value="FORTRESS">🏰 FORTRESS — Outcome evaluation</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* SHARP: move annotations */}
          {type === "SHARP" && fenValid && (
            <div className="grid gap-2 p-3 bg-muted rounded-lg">
              <Label>Move Annotations</Label>
              <p className="text-xs text-muted-foreground">
                Add moves and mark them as correct or blunders. Use{" "}
                {notationFormat === "algebraic" ? "algebraic (e.g. Nf3)" : "UCI (e.g. g1f3)"}{" "}
                notation.
              </p>
              <div className="flex gap-2">
                <Input
                  placeholder={
                    notationFormat === "algebraic" ? "e.g. Nf3" : "e.g. g1f3"
                  }
                  value={moveInput}
                  onChange={(e) => setMoveInput(e.target.value)}
                  className="font-mono"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      addAnnotation("correct");
                    }
                  }}
                />
                <Button
                  size="sm"
                  variant="outline"
                  className="border-green-500 text-green-700 hover:bg-green-50"
                  onClick={() => addAnnotation("correct")}
                >
                  <CheckCircle2 className="w-4 h-4 mr-1" /> Correct
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="border-red-500 text-red-700 hover:bg-red-50"
                  onClick={() => addAnnotation("blunder")}
                >
                  <XCircle className="w-4 h-4 mr-1" /> Blunder
                </Button>
              </div>
              {annotations.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {annotations.map((a) => (
                    <Badge
                      key={a.move}
                      variant="outline"
                      className={`font-mono cursor-pointer ${
                        a.annotation === "correct"
                          ? "border-green-500 text-green-700 bg-green-50"
                          : "border-red-500 text-red-700 bg-red-50"
                      }`}
                      onClick={() => removeAnnotation(a.move)}
                    >
                      {notationFormat === "algebraic"
                        ? uciToAlgebraic(fen, a.move)
                        : a.move}
                      <X className="w-3 h-3 ml-1" />
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* FORTRESS: expected outcome */}
          {type === "FORTRESS" && (
            <div className="grid gap-2">
              <Label>Expected Outcome</Label>
              <Select
                value={expectedOutcome}
                onValueChange={(v) => setExpectedOutcome(v as Outcome)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="win">🏆 Decisive — Win</SelectItem>
                  <SelectItem value="loss">💀 Decisive — Loss</SelectItem>
                  <SelectItem value="draw">➖ Drawn</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!name.trim() || !fenValid}>
            <Plus className="w-4 h-4 mr-1" />
            Add Position
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
