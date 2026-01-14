# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert .bagz game files to .bag training files.

Takes compressed .bagz files (where each record is a game/list of positions)
and converts them to uncompressed .bag files (where each record is a single
position), applying:
1. Position subsampling: select positions at indices i % 3 == cycle_index
2. FEN-based deduplication (ignoring half-move and full-move counters)
3. Meta-feature computation (game result, etc. from whole-game analysis)
4. Field stripping: only keep essential fields for training
"""

from dataclasses import dataclass
from pathlib import Path

import chess
import msgpack

from catgpt.core.data.grain.bagz import BagReader, BagWriter
from catgpt.core.data.grain.coders import decode_game, LeelaPositionData

# Standard starting position (first 4 parts of FEN, ignoring move counters)
STANDARD_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"

# Dangerous invariance_info bits that should fail verification:
#   bit 0 (0x01): flip_transform - vertical flip (ranks 1↔8)
#   bit 1 (0x02): mirror_transform - horizontal flip (files a↔h)
#   bit 2 (0x04): transpose_transform - diagonal flip
#   bit 6 (0x40): marked_for_deletion - flagged by rescorer
#   bit 7 (0x80): side_to_move_black - for canonicalized formats
# Acceptable bits:
#   bit 3 (0x08): best_q_proven - tablebase
#   bit 4 (0x10): max_game_length_exceeded - game cut short
#   bit 5 (0x20): game_adjudicated - game ended by adjudication
DANGEROUS_INVARIANCE_MASK = 0x01 | 0x02 | 0x04 | 0x40 | 0x80  # = 0xC7


@dataclass
class MetaFeatures:
    """Meta-game features computed from analyzing the entire game.

    These features require context beyond a single position, such as
    knowing the game outcome or tracking piece movements across moves.

    Add new fields here as we implement more meta-features.
    """

    game_result: int  # -1=loss, 0=draw, 1=win from side-to-move perspective

    # Piece movement tracking
    # For occupied squares: where will the piece move next?
    # Format: square -> destination (e.g., "e2" -> "e4", or "e7" -> "e8q" for promotion)
    piece_will_move_to: dict[str, str]

    # For any square (empty or occupied): from which square will the next piece come?
    # This is the IMMEDIATE source - the FROM square of the move that occupies this square.
    # Includes both moves to empty squares and captures.
    # Format: square -> source (e.g., "e4" -> "e2", or "e8" -> "e7q" for promotion)
    square_will_be_occupied_from: dict[str, str]

    # For any square: where is the piece CURRENTLY located that will eventually occupy it?
    # This traces through multiple moves. E.g., if pawn goes e2->e4->e5, at position 0:
    #   square_will_be_occupied_from["e5"] = "e4" (immediate source)
    #   square_will_be_occupied_by_piece_on["e5"] = "e2" (current location)
    # If a knight on g1 goes g1->f3->g1, at position 0:
    #   square_will_be_occupied_by_piece_on["g1"] = "g1" (same piece returns)
    # Format: square -> current location (e.g., "e5" -> "e2", or "e8" -> "e7q" for promotion)
    square_will_be_occupied_by_piece_on: dict[str, str]

    # Current location of the piece that will be captured next (None if no future captures)
    # This is where the piece is at the current position, even if it moves before capture.
    # For en passant captures, this correctly tracks the captured pawn's current location.
    next_capture_square: str | None

    # Square of the pawn that will move next (None if no future pawn moves)
    # This is the current location of the pawn before it moves
    next_pawn_move_square: str | None


def _boards_match(board1: chess.Board, board2: chess.Board) -> bool:
    """Check if two boards have the same piece placement."""
    return board1.board_fen() == board2.board_fen()


def find_move_between_positions(
    board_before: chess.Board,
    board_after: chess.Board,
) -> chess.Move:
    """Find the move that transforms board_before into board_after.

    Args:
        board_before: Board state before the move.
        board_after: Board state after the move.

    Returns:
        The chess.Move that was played.

    Raises:
        VerificationError: If no legal move produces the expected position.
    """
    for move in board_before.legal_moves:
        board_before.push(move)
        match = _boards_match(board_before, board_after)
        board_before.pop()
        if match:
            return move

    raise VerificationError(
        f"No legal move found between positions: "
        f"{board_before.fen()} -> {board_after.fen()}"
    )


def _format_move_destination(to_square: int, promotion: int | None) -> str:
    """Format destination square with optional promotion piece.

    Args:
        to_square: The destination square index.
        promotion: The promotion piece type (chess.QUEEN, etc.) or None.

    Returns:
        Square name, optionally with promotion suffix (e.g., "e8q").
    """
    sq_name = chess.square_name(to_square)
    if promotion is not None:
        sq_name += chess.piece_symbol(promotion)
    return sq_name


def compute_meta_features(positions: list[LeelaPositionData]) -> list[MetaFeatures]:
    """Compute meta-game features for each position in a game.

    This function analyzes the entire game to compute features that require
    context beyond a single position. Uses a backward scan for O(n) efficiency.

    Args:
        positions: All positions in a game, in order.

    Returns:
        List of MetaFeatures, one per position, in the same order.
    """
    n = len(positions)
    if n == 0:
        return []

    # -------------------------------------------------------------------------
    # Build boards and determine moves between consecutive positions
    # -------------------------------------------------------------------------
    boards = [chess.Board(pos.fen) for pos in positions]

    # moves[i] = move from position i to i+1, or None for last position
    moves: list[chess.Move | None] = []
    for i in range(n - 1):
        moves.append(find_move_between_positions(boards[i], boards[i + 1]))
    moves.append(None)  # No move after last position

    # -------------------------------------------------------------------------
    # Game result computation
    # -------------------------------------------------------------------------
    # Determine terminal result from last position's evaluation.
    # The original result field is corrupted (always 0), so we infer from Q/D.
    last_pos = positions[-1]
    if last_pos.root_d > 0.5:
        terminal_result = 0  # Draw
    elif last_pos.root_q > 0:
        terminal_result = 1  # Win for side-to-move at terminal position
    else:
        terminal_result = -1  # Loss for side-to-move at terminal position

    # Precompute game results for all positions
    game_results: list[int] = []
    for i in range(n):
        dist_from_end = n - 1 - i
        if dist_from_end % 2 == 0:
            game_results.append(terminal_result)
        else:
            game_results.append(-terminal_result)

    # -------------------------------------------------------------------------
    # Piece movement computation (backward scan)
    # -------------------------------------------------------------------------
    # State: tracking what happens AFTER each position
    # will_be_occupied_from[sq] = source square (with promo suffix) of next piece to land
    # will_move_to[sq] = destination (with promo suffix) of next move from this square
    # piece_at[sq] = current location of piece that will eventually occupy sq (traced through moves)
    will_be_occupied_from: dict[str, str] = {}
    will_move_to: dict[str, str] = {}
    piece_at: dict[str, str] = {}  # Traces back to current location

    # Capture and pawn move tracking
    next_capture_sq: str | None = None  # Square of next captured piece
    next_pawn_sq: str | None = None  # Square of next pawn to move

    # Results collected in reverse order
    results_reversed: list[MetaFeatures] = []

    for i in range(n - 1, -1, -1):
        # Update state with move from position i to i+1 (the "next" move from i's view)
        move = moves[i]
        if move is not None:
            from_sq = chess.square_name(move.from_square)
            to_sq_with_promo = _format_move_destination(move.to_square, move.promotion)
            to_sq_name = chess.square_name(move.to_square)

            # CRITICAL: If this is a capture, the captured piece doesn't move.
            # Clear any will_move_to entry for the captured piece's square.
            if boards[i].is_en_passant(move):
                # En passant: captured pawn is on same file as destination, same rank as source
                captured_sq = chess.square(
                    chess.square_file(move.to_square),
                    chess.square_rank(move.from_square),
                )
                captured_sq_name = chess.square_name(captured_sq)
                if captured_sq_name in will_move_to:
                    del will_move_to[captured_sq_name]
            elif to_sq_name in will_move_to:
                # Normal capture: captured piece is on destination square
                del will_move_to[to_sq_name]

            # Compute source_with_promo for piece tracing (used in multiple places)
            source_with_promo = from_sq
            if move.promotion is not None:
                source_with_promo += chess.piece_symbol(move.promotion)

            # Handle castling: both king and rook move
            if boards[i].is_castling(move):
                # King move is already in from_sq/to_sq
                will_move_to[from_sq] = to_sq_with_promo
                will_be_occupied_from[to_sq_name] = from_sq

                # Update piece_at: trace any square pointing to to_sq back to from_sq
                for sq in list(piece_at.keys()):
                    if piece_at[sq] == to_sq_name:
                        piece_at[sq] = from_sq
                piece_at[to_sq_name] = from_sq

                # Determine rook move based on castling type
                if boards[i].is_kingside_castling(move):
                    # Kingside: rook h1->f1 (white) or h8->f8 (black)
                    if boards[i].turn == chess.WHITE:
                        rook_from, rook_to = "h1", "f1"
                    else:
                        rook_from, rook_to = "h8", "f8"
                else:
                    # Queenside: rook a1->d1 (white) or a8->d8 (black)
                    if boards[i].turn == chess.WHITE:
                        rook_from, rook_to = "a1", "d1"
                    else:
                        rook_from, rook_to = "a8", "d8"

                will_move_to[rook_from] = rook_to
                will_be_occupied_from[rook_to] = rook_from

                # Update piece_at for rook move
                for sq in list(piece_at.keys()):
                    if piece_at[sq] == rook_to:
                        piece_at[sq] = rook_from
                piece_at[rook_to] = rook_from
            else:
                # Normal move (including captures and promotions)
                will_move_to[from_sq] = to_sq_with_promo
                will_be_occupied_from[to_sq_name] = source_with_promo

                # Update piece_at: trace any square pointing to to_sq back to source
                # This chains through multiple moves (e.g., e2->e4->e5 traces e5 back to e2)
                for sq in list(piece_at.keys()):
                    if piece_at[sq] == to_sq_name:
                        piece_at[sq] = source_with_promo
                piece_at[to_sq_name] = source_with_promo

            # Track captures: update next_capture_sq if this move is a capture,
            # or trace back if this move affects the piece that will be captured
            if boards[i].is_capture(move):
                if boards[i].is_en_passant(move):
                    # En passant: captured pawn is on same file as destination, same rank as source
                    captured_sq = chess.square(
                        chess.square_file(move.to_square),
                        chess.square_rank(move.from_square),
                    )
                    next_capture_sq = chess.square_name(captured_sq)
                else:
                    # Normal capture: captured piece is on destination square
                    next_capture_sq = chess.square_name(move.to_square)
            elif next_capture_sq is not None:
                # Trace back: if this move places a piece on next_capture_sq,
                # update to the source (where the piece is at position i)
                if boards[i].is_castling(move):
                    # Check king move
                    if next_capture_sq == to_sq_name:
                        next_capture_sq = from_sq
                    # Check rook move
                    if boards[i].is_kingside_castling(move):
                        rook_from = "h1" if boards[i].turn == chess.WHITE else "h8"
                        rook_to = "f1" if boards[i].turn == chess.WHITE else "f8"
                    else:
                        rook_from = "a1" if boards[i].turn == chess.WHITE else "a8"
                        rook_to = "d1" if boards[i].turn == chess.WHITE else "d8"
                    if next_capture_sq == rook_to:
                        next_capture_sq = rook_from
                else:
                    # Normal move
                    if next_capture_sq == to_sq_name:
                        next_capture_sq = from_sq

            # Track pawn moves: update next_pawn_sq if this move is by a pawn
            piece = boards[i].piece_at(move.from_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                next_pawn_sq = from_sq

        # Record meta-features for position i based on current board state
        piece_dest: dict[str, str] = {}
        square_occup: dict[str, str] = {}
        square_occup_current: dict[str, str] = {}

        for sq in chess.SQUARES:
            sq_name = chess.square_name(sq)

            # For occupied squares: check if the piece will move
            if boards[i].piece_at(sq) is not None:
                if sq_name in will_move_to:
                    piece_dest[sq_name] = will_move_to[sq_name]

            # For any square: check if a piece will move to occupy it
            if sq_name in will_be_occupied_from:
                square_occup[sq_name] = will_be_occupied_from[sq_name]

            # For any square: where is the piece currently that will occupy it?
            if sq_name in piece_at:
                square_occup_current[sq_name] = piece_at[sq_name]

        results_reversed.append(
            MetaFeatures(
                game_result=game_results[i],
                piece_will_move_to=piece_dest,
                square_will_be_occupied_from=square_occup,
                square_will_be_occupied_by_piece_on=square_occup_current,
                next_capture_square=next_capture_sq,
                next_pawn_move_square=next_pawn_sq,
            )
        )

    # Reverse to get results in forward order
    return list(reversed(results_reversed))


@dataclass
class TrainingPositionData:
    """Slimmed-down position data for training.

    Contains essential fields from the original position plus computed
    meta-features from whole-game analysis.
    """

    # Core position data (from LeelaPositionData)
    fen: str
    legal_moves: list[tuple[str, float]]  # (move_uci, probability)
    root_q: float
    root_d: float
    best_move_uci: str | None

    # Meta-game features (from MetaFeatures)
    game_result: int  # -1=loss, 0=draw, 1=win from side-to-move perspective
    piece_will_move_to: dict[str, str]  # occupied_square -> destination
    square_will_be_occupied_from: dict[str, str]  # any_square -> immediate source of move
    square_will_be_occupied_by_piece_on: dict[str, str]  # any_square -> current location of piece
    next_capture_square: str | None  # current square of piece that will be captured next
    next_pawn_move_square: str | None  # square of pawn that will move next


def encode_training_position(position: TrainingPositionData) -> bytes:
    """Encode a training position to bytes."""
    data = {
        "fen": position.fen,
        "legal_moves": position.legal_moves,
        "root_q": position.root_q,
        "root_d": position.root_d,
        "best_move_uci": position.best_move_uci,
        # Meta-features
        "game_result": position.game_result,
        "piece_will_move_to": position.piece_will_move_to,
        "square_will_be_occupied_from": position.square_will_be_occupied_from,
        "square_will_be_occupied_by_piece_on": position.square_will_be_occupied_by_piece_on,
        "next_capture_square": position.next_capture_square,
        "next_pawn_move_square": position.next_pawn_move_square,
    }
    return msgpack.packb(data, use_bin_type=True)


def decode_training_position(encoded: bytes) -> TrainingPositionData:
    """Decode bytes to a training position."""
    data = msgpack.unpackb(encoded, raw=False)
    return TrainingPositionData(
        fen=data["fen"],
        legal_moves=[(m, p) for m, p in data["legal_moves"]],
        root_q=data["root_q"],
        root_d=data["root_d"],
        best_move_uci=data["best_move_uci"],
        # Meta-features
        game_result=data["game_result"],
        piece_will_move_to=data["piece_will_move_to"],
        square_will_be_occupied_from=data["square_will_be_occupied_from"],
        square_will_be_occupied_by_piece_on=data["square_will_be_occupied_by_piece_on"],
        next_capture_square=data["next_capture_square"],
        next_pawn_move_square=data["next_pawn_move_square"],
    )


def fen_dedup_key(fen: str) -> str:
    """Extract the deduplication key from a FEN string.

    Removes the half-move clock and full-move number from the FEN.
    FEN format: <pieces> <side> <castling> <ep> <halfmove> <fullmove>

    Args:
        fen: A FEN string.

    Returns:
        FEN string with halfmove and fullmove counters removed.
    """
    parts = fen.split()
    # Keep only the first 4 parts: pieces, side to move, castling, en passant
    return " ".join(parts[:4])


def is_standard_starting_position(fen: str) -> bool:
    """Check if a FEN represents the standard starting position.

    Ignores half-move clock and full-move number.

    Args:
        fen: A FEN string.

    Returns:
        True if this is the standard starting position.
    """
    return fen_dedup_key(fen) == STANDARD_STARTING_FEN


class VerificationError(Exception):
    """Raised when data verification fails."""

    pass


def verify_legal_moves(pos: LeelaPositionData) -> None:
    """Verify that the legal moves in the position match python-chess.

    Args:
        pos: A position with legal_moves field.

    Raises:
        VerificationError: If legal moves don't match python-chess.
    """
    try:
        board = chess.Board(pos.fen)
    except ValueError as e:
        raise VerificationError(f"Invalid FEN: {e}") from e

    # Get legal moves from python-chess
    chess_legal_moves = {move.uci() for move in board.legal_moves}

    # Get legal moves from the data
    data_legal_moves = {move_uci for move_uci, _ in pos.legal_moves}

    if chess_legal_moves != data_legal_moves:
        missing = chess_legal_moves - data_legal_moves
        extra = data_legal_moves - chess_legal_moves
        error_parts = []
        if missing:
            error_parts.append(f"missing: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra: {sorted(extra)}")
        raise VerificationError(f"Legal moves mismatch - {', '.join(error_parts)}")


def verify_invariance_info(pos: LeelaPositionData) -> None:
    """Verify that the invariance_info doesn't have dangerous bits set.

    Dangerous bits indicate transformed positions or positions marked for deletion,
    which should not be used for training.

    Args:
        pos: A position with invariance_info field.

    Raises:
        VerificationError: If dangerous invariance bits are set.
    """
    inv = pos.invariance_info
    dangerous_bits = inv & DANGEROUS_INVARIANCE_MASK

    if dangerous_bits:
        flags = []
        if inv & 0x01:
            flags.append("flip_transform")
        if inv & 0x02:
            flags.append("mirror_transform")
        if inv & 0x04:
            flags.append("transpose_transform")
        if inv & 0x40:
            flags.append("marked_for_deletion")
        if inv & 0x80:
            flags.append("side_to_move_black")
        raise VerificationError(f"Dangerous invariance flags: {', '.join(flags)} (byte={inv})")


def convert_bagz_to_bag(
    bagz_path: str | Path,
    output_path: str | Path | None = None,
    *,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """Convert a .bagz file to .bag format for training.

    Each game in the input is processed to:
    1. Verify data integrity (Chess960 check, legal moves, invariance flags)
    2. Sub-select positions where index % 3 == cycle_index (cycling 0,1,2)
    3. Deduplicate by FEN (ignoring half-move and full-move counters)
    4. Write individual positions with only training-essential fields

    Args:
        bagz_path: Path to input .bagz file containing games.
        output_path: Path for output .bag file. If None, replaces .bagz with .bag.
        verbose: If True, print progress information.

    Returns:
        Tuple of (games_processed, positions_before_dedup, positions_written).

    Raises:
        VerificationError: If any data integrity check fails.
    """
    bagz_path = Path(bagz_path)

    if output_path is None:
        # Replace .bagz with .bag
        if bagz_path.suffix == ".bagz":
            output_path = bagz_path.with_suffix(".bag")
        else:
            output_path = bagz_path.with_name(bagz_path.name + ".bag")
    else:
        output_path = Path(output_path)

    reader = BagReader(str(bagz_path))
    num_games = len(reader)

    if verbose:
        print(f"Processing {num_games} games from {bagz_path}")

    # Track seen FENs for deduplication (within this file)
    seen_fens: set[str] = set()

    games_processed = 0
    positions_before_dedup = 0
    positions_written = 0
    cycle_index = 0  # Cycles through 0, 1, 2 for each game

    with BagWriter(str(output_path), compress=False) as writer:
        for game_idx in range(num_games):
            encoded_game = reader[game_idx]
            game_positions = decode_game(encoded_game)

            if not game_positions:
                raise VerificationError(f"Game {game_idx}: Empty game")

            # Verify first position is standard starting position (not Chess960)
            first_pos = game_positions[0]
            if not is_standard_starting_position(first_pos.fen):
                raise VerificationError(
                    f"Game {game_idx}: Chess960 detected - first FEN: {first_pos.fen}"
                )

            # Compute meta-features for all positions in the game
            meta_features = compute_meta_features(game_positions)

            # Sub-select positions: keep only those where pos_idx % 3 == cycle_index
            selected_positions = [
                (pos_idx, pos, meta_features[pos_idx])
                for pos_idx, pos in enumerate(game_positions)
                if pos_idx % 3 == cycle_index
            ]

            # Advance cycle for next game
            cycle_index = (cycle_index + 1) % 3

            positions_before_dedup += len(selected_positions)

            # Write each selected position, deduplicating by FEN
            for pos_idx, pos, meta in selected_positions:
                # Verify legal moves and invariance info
                try:
                    verify_legal_moves(pos)
                except VerificationError as e:
                    raise VerificationError(
                        f"Game {game_idx}, pos {pos_idx}: {e} (FEN: {pos.fen})"
                    ) from e

                try:
                    verify_invariance_info(pos)
                except VerificationError as e:
                    raise VerificationError(
                        f"Game {game_idx}, pos {pos_idx}: {e} (FEN: {pos.fen})"
                    ) from e

                dedup_key = fen_dedup_key(pos.fen)

                if dedup_key in seen_fens:
                    continue

                seen_fens.add(dedup_key)

                # Create training position with core fields + meta-features
                training_pos = TrainingPositionData(
                    fen=pos.fen,
                    legal_moves=pos.legal_moves,
                    root_q=pos.root_q,
                    root_d=pos.root_d,
                    best_move_uci=pos.best_move_uci,
                    game_result=meta.game_result,
                    piece_will_move_to=meta.piece_will_move_to,
                    square_will_be_occupied_from=meta.square_will_be_occupied_from,
                    square_will_be_occupied_by_piece_on=meta.square_will_be_occupied_by_piece_on,
                    next_capture_square=meta.next_capture_square,
                    next_pawn_move_square=meta.next_pawn_move_square,
                )

                encoded = encode_training_position(training_pos)
                writer.write(encoded)
                positions_written += 1

            games_processed += 1

            if verbose and games_processed % 1000 == 0:
                print(
                    f"Processed {games_processed}/{num_games} games, "
                    f"{positions_written} unique positions written..."
                )

    if verbose:
        dedup_removed = positions_before_dedup - positions_written
        print(f"✓ Processed {games_processed} games")
        print(f"  Positions selected (after mod 3): {positions_before_dedup}")
        print(f"  Duplicates removed: {dedup_removed}")
        print(f"  Unique positions written: {positions_written}")
        print(f"  Output: {output_path}")

    return games_processed, positions_before_dedup, positions_written


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python -m catgpt.core.data.grain.bagz_to_bag <input.bagz> [options]")
        print()
        print("Convert .bagz game files to .bag training files.")
        print()
        print("This script:")
        print("  1. Verifies data integrity (fails fast on any error)")
        print("  2. Computes meta-features from whole-game analysis:")
        print("     - game_result: win/draw/loss from side-to-move perspective")
        print("     - piece_will_move_to: where each piece will move next")
        print("     - square_will_be_occupied_from: immediate source of next move to each square")
        print("     - square_will_be_occupied_by_piece_on: current location of piece that will")
        print("       eventually occupy each square (traces through multiple moves)")
        print("     - next_capture_square: current location of the piece that will be captured next")
        print("     - next_pawn_move_square: square of the pawn that will move next")
        print("  3. Sub-selects positions where index % 3 == cycle (cycling 0,1,2 per game)")
        print("  4. Deduplicates by FEN (ignoring half-move and full-move counters)")
        print()
        print("Arguments:")
        print("  input.bagz         Path to input .bagz file containing games")
        print()
        print("Options:")
        print("  --output PATH      Output .bag file path (default: input.bag)")
        print("  --verbose, -v      Print progress information")
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)

    input_bagz = sys.argv[1]

    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    games, before, after = convert_bagz_to_bag(
        input_bagz,
        output_path=output_path,
        verbose=verbose,
    )

    if not verbose:
        print(f"Converted {games} games -> {after} unique positions from {input_bagz}")
