import { Chess, type Square as ChessSquare } from "chess.js";
import type { UCIMove } from "./types";

/**
 * Convert a sequence of UCI moves to algebraic notation, where each move
 * is resolved relative to the position after all preceding moves.
 * Also returns the FEN after each move for linking purposes.
 */
export function uciSequenceToAlgebraic(
  fen: string,
  moves: UCIMove[]
): { san: string; fenAfter: string | null }[] {
  const result: { san: string; fenAfter: string | null }[] = [];
  try {
    const chess = new Chess(fen);
    for (const uci of moves) {
      const { from, to, promotion } = parseUCIMove(uci);
      const move = chess.move({
        from: from as ChessSquare,
        to: to as ChessSquare,
        promotion,
      });
      result.push({
        san: move ? move.san : uci,
        fenAfter: move ? chess.fen() : null,
      });
    }
  } catch {
    for (let i = result.length; i < moves.length; i++) {
      result.push({ san: moves[i], fenAfter: null });
    }
  }
  return result;
}

/**
 * Compute the FEN after applying a sequence of UCI moves.
 * Returns null if any move is illegal.
 */
export function fenAfterMoves(
  fen: string,
  moves: UCIMove[]
): string | null {
  try {
    const chess = new Chess(fen);
    for (const uci of moves) {
      const { from, to, promotion } = parseUCIMove(uci);
      const move = chess.move({
        from: from as ChessSquare,
        to: to as ChessSquare,
        promotion,
      });
      if (!move) return null;
    }
    return chess.fen();
  } catch {
    return null;
  }
}

/**
 * Parse a UCI move string like "e2e4" or "e7e8q" into from/to/promotion.
 */
export function parseUCIMove(uci: UCIMove): {
  from: string;
  to: string;
  promotion?: string;
} {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  };
}

/**
 * Convert a UCI move to algebraic notation given a FEN.
 */
export function uciToAlgebraic(fen: string, uci: UCIMove): string {
  try {
    const chess = new Chess(fen);
    const { from, to, promotion } = parseUCIMove(uci);
    const move = chess.move({ from: from as ChessSquare, to: to as ChessSquare, promotion });
    return move ? move.san : uci;
  } catch {
    return uci;
  }
}

/**
 * Convert an algebraic move to UCI given a FEN.
 */
export function algebraicToUCI(fen: string, san: string): UCIMove | null {
  try {
    const chess = new Chess(fen);
    const move = chess.move(san);
    if (!move) return null;
    return move.from + move.to + (move.promotion ?? "");
  } catch {
    return null;
  }
}

/**
 * Validate a FEN string.
 */
export function isValidFEN(fen: string): boolean {
  try {
    new Chess(fen);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get all legal moves from a position in UCI format.
 */
export function getLegalMoves(fen: string): UCIMove[] {
  try {
    const chess = new Chess(fen);
    return chess.moves({ verbose: true }).map(
      (m) => m.from + m.to + (m.promotion ?? "")
    );
  } catch {
    return [];
  }
}

/**
 * Get the side to move from a FEN.
 */
export function sideToMove(fen: string): "w" | "b" {
  return fen.split(" ")[1] as "w" | "b";
}

/**
 * Convert a UCI move to an arrow for react-chessboard.
 * Returns [from, to] tuple.
 */
export function uciToArrow(uci: UCIMove): [string, string] {
  const { from, to } = parseUCIMove(uci);
  return [from, to];
}

/**
 * Attempt to make a move on the board. Returns the new FEN, SAN, and UCI
 * notation if legal, or null if the move is illegal.
 */
export function tryMove(
  fen: string,
  from: string,
  to: string,
  promotion?: string
): { newFen: string; san: string; uci: UCIMove } | null {
  try {
    const chess = new Chess(fen);
    const move = chess.move({
      from: from as ChessSquare,
      to: to as ChessSquare,
      promotion: promotion as "q" | "r" | "b" | "n" | undefined,
    });
    if (!move) return null;
    return {
      newFen: chess.fen(),
      san: move.san,
      uci: move.from + move.to + (move.promotion ?? ""),
    };
  } catch {
    return null;
  }
}

/**
 * Build a move line from a list of UCI moves, applying them one-by-one
 * starting from the given FEN. Returns the resolved entries (SAN, UCI, and
 * FEN after each move) along with the index of the first illegal move, if any.
 */
export function buildLineFromUCI(
  fen: string,
  ucis: UCIMove[]
): {
  entries: { san: string; uci: UCIMove; fenAfter: string }[];
  invalidAt: number | null;
} {
  const entries: { san: string; uci: UCIMove; fenAfter: string }[] = [];
  try {
    const chess = new Chess(fen);
    for (let i = 0; i < ucis.length; i++) {
      const { from, to, promotion } = parseUCIMove(ucis[i]);
      const move = chess.move({
        from: from as ChessSquare,
        to: to as ChessSquare,
        promotion: promotion as "q" | "r" | "b" | "n" | undefined,
      });
      if (!move) return { entries, invalidAt: i };
      entries.push({
        san: move.san,
        uci: move.from + move.to + (move.promotion ?? ""),
        fenAfter: chess.fen(),
      });
    }
  } catch {
    return { entries, invalidAt: entries.length };
  }
  return { entries, invalidAt: null };
}

/**
 * Check if a move from→to would be a pawn promotion.
 */
export function isPromotion(fen: string, from: string, to: string): boolean {
  try {
    const chess = new Chess(fen);
    const piece = chess.get(from as ChessSquare);
    if (!piece || piece.type !== "p") return false;
    const targetRank = to[1];
    return (piece.color === "w" && targetRank === "8") ||
           (piece.color === "b" && targetRank === "1");
  } catch {
    return false;
  }
}

/**
 * Build a Lichess analysis board URL for the given FEN.
 */
export function lichessAnalysisUrl(fen: string): string {
  return `https://lichess.org/analysis/standard/${encodeURIComponent(fen)}`;
}
