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
1. Game verification (standard start, move connectivity, legal move matching)
2. Half-move clock filter (excludes positions with clock > 90)
3. FEN-based deduplication (ignoring half-move and full-move counters)
4. Field stripping: only keep essential fields for training
"""

from dataclasses import dataclass
from pathlib import Path

import chess
import msgpack
import numpy as np

from catgpt.core.data.grain.bagz import BagReader, BagWriter
from catgpt.core.data.grain.coders import LeelaPositionData, decode_game

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


class VerificationError(Exception):
    """Raised when data verification fails."""

    pass


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


def compute_st_q(
    best_qs: list[float],
    alpha: float = 1 - 1 / 6,
) -> list[float]:
    """Compute short-term Q values from bestQ values across a game.

    Uses a backward exponential moving average with sign alternation
    (since each position's Q is from the side-to-move perspective, which
    alternates each ply).

    The EMA is computed backward from the end of the game:
        st_q[-1] = best_q[-1]
        st_q[i]  = alpha * st_q[i+1] + (1 - alpha) * best_q[i]
    (with sign alternation applied before and after to handle perspective changes)

    With alpha = 1 - 1/6 ≈ 0.833, this captures roughly the next ~6 plies,
    providing a smooth "what actually happened short-term" signal that's more
    stable than single-ply root_q but less noisy than the game result.

    Args:
        best_qs: List of bestQ values for each position in game order.
            Each value is in [-1, 1] from the side-to-move perspective.
        alpha: EMA decay factor. Higher = longer memory. Default 1-1/6.

    Returns:
        List of short-term Q values, same length as input.
    """
    qs = np.array(best_qs, dtype=np.float64)
    n = len(qs)
    if n == 0:
        return []

    # Alternate signs: positions at even indices keep sign, odd indices flip.
    # This normalizes all Q values to the same perspective before averaging.
    signs = (-1.0) ** np.arange(n)
    qs_normed = qs * signs

    # Backward EMA: propagate from end of game to start
    st = np.zeros(n, dtype=np.float64)
    st[-1] = qs_normed[-1]
    for i in range(n - 2, -1, -1):
        st[i] = alpha * st[i + 1] + (1 - alpha) * qs_normed[i]

    # Restore per-position perspective
    st_q = st * signs

    # Clamp to [-1, 1] for safety
    st_q = np.clip(st_q, -1.0, 1.0)

    return st_q.tolist()


def compute_game_result(positions: list[LeelaPositionData]) -> list[int]:
    """Compute game result for each position from terminal evaluation.

    The result alternates sign each ply since each position is from the
    perspective of the side to move.

    Args:
        positions: All positions in a game, in order.

    Returns:
        List of game results (-1=loss, 0=draw, 1=win from side-to-move), one per position.
    """
    n = len(positions)
    if n == 0:
        return []

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

    return game_results


def verify_game_integrity(
    game_positions: list[LeelaPositionData],
    game_idx: int,
) -> None:
    """Verify full game integrity: standard start, move connectivity, legal moves.

    Checks that:
    1. The game starts from the standard starting position (not Chess960).
    2. Every consecutive pair of positions is connected by a legal move.
    3. Each position's legal moves match those from python-chess exactly.
    4. No dangerous invariance bits are set.

    Args:
        game_positions: All positions in a game, in order.
        game_idx: Game index for error messages.

    Raises:
        VerificationError: If any integrity check fails.
    """
    if not game_positions:
        raise VerificationError(f"Game {game_idx}: Empty game")

    # 1. Verify first position is standard starting position (not Chess960)
    first_pos = game_positions[0]
    if not is_standard_starting_position(first_pos.fen):
        raise VerificationError(
            f"Game {game_idx}: Chess960 detected - first FEN: {first_pos.fen}"
        )

    # Build boards for all positions
    boards = []
    for pos_idx, pos in enumerate(game_positions):
        try:
            board = chess.Board(pos.fen)
        except ValueError as e:
            raise VerificationError(
                f"Game {game_idx}, pos {pos_idx}: Invalid FEN: {e}"
            ) from e
        boards.append(board)

    # 2. Verify move connectivity: each consecutive pair must be connected by a legal move
    for i in range(len(boards) - 1):
        try:
            find_move_between_positions(boards[i], boards[i + 1])
        except VerificationError as e:
            raise VerificationError(
                f"Game {game_idx}, pos {i}->{i+1}: No legal move connects "
                f"{boards[i].fen()} to {boards[i+1].fen()}"
            ) from e

    # 3. Verify legal moves match python-chess for every position
    for pos_idx, pos in enumerate(game_positions):
        try:
            verify_legal_moves(pos)
        except VerificationError as e:
            raise VerificationError(
                f"Game {game_idx}, pos {pos_idx}: {e} (FEN: {pos.fen})"
            ) from e

    # 4. Verify invariance info for every position
    for pos_idx, pos in enumerate(game_positions):
        try:
            verify_invariance_info(pos)
        except VerificationError as e:
            raise VerificationError(
                f"Game {game_idx}, pos {pos_idx}: {e} (FEN: {pos.fen})"
            ) from e


@dataclass
class TrainingPositionData:
    """Slimmed-down position data for training.

    Contains essential fields from the original position plus the game result.
    """

    # Core position data (from LeelaPositionData)
    fen: str
    legal_moves: list[tuple[str, float]]  # (move_uci, probability)
    root_q: float
    root_d: float
    best_q: float
    best_move_uci: str | None

    # Game result from terminal evaluation
    game_result: int  # -1=loss, 0=draw, 1=win from side-to-move perspective

    # Short-term Q value: backward EMA of bestQ across the game.
    # Used as value target for the optimistic policy head's weighting.
    # In [-1, 1] from the side-to-move perspective.
    st_q: float = 0.0


def encode_training_position(position: TrainingPositionData) -> bytes:
    """Encode a training position to bytes."""
    data = {
        "fen": position.fen,
        "legal_moves": position.legal_moves,
        "root_q": position.root_q,
        "root_d": position.root_d,
        "best_q": position.best_q,
        "best_move_uci": position.best_move_uci,
        "game_result": position.game_result,
        "st_q": position.st_q,
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
        best_q=data.get("best_q", data["root_q"]),
        best_move_uci=data["best_move_uci"],
        game_result=data["game_result"],
        st_q=data.get("st_q", 0.0),
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
    1. Verify full game integrity (standard start, move connectivity, legal moves,
       invariance flags)
    2. Compute game result from terminal evaluation
    3. Filter positions with half-move clock > 90 (near 50-move draw)
    4. Deduplicate by FEN (ignoring half-move and full-move counters)
    5. Write individual positions with only training-essential fields

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

    with BagWriter(str(output_path), compress=False) as writer:
        for game_idx in range(num_games):
            encoded_game = reader[game_idx]
            game_positions = decode_game(encoded_game)

            # Verify full game integrity (start pos, connectivity, legal moves, invariance)
            verify_game_integrity(game_positions, game_idx)

            # Compute game result for all positions
            game_results = compute_game_result(game_positions)

            # Compute short-term Q values from bestQ across the game
            best_qs = [pos.best_q for pos in game_positions]
            st_qs = compute_st_q(best_qs)

            # Filter positions: exclude those with half-move clock > 90 (near 50-move draw)
            selected_positions = [
                (pos_idx, pos, game_results[pos_idx], st_qs[pos_idx])
                for pos_idx, pos in enumerate(game_positions)
                if int(pos.fen.split()[4]) <= 90
            ]

            positions_before_dedup += len(selected_positions)

            # Write in reverse order so that later (rarer) positions win dedup ties
            for _pos_idx, pos, game_result, st_q in reversed(selected_positions):
                dedup_key = fen_dedup_key(pos.fen)

                if dedup_key in seen_fens:
                    continue

                seen_fens.add(dedup_key)

                # Create training position with core fields + game result + st_q
                training_pos = TrainingPositionData(
                    fen=pos.fen,
                    legal_moves=pos.legal_moves,
                    root_q=pos.root_q,
                    root_d=pos.root_d,
                    best_q=pos.best_q,
                    best_move_uci=pos.best_move_uci,
                    game_result=game_result,
                    st_q=st_q,
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
        print(f"  Positions selected (after clock filter): {positions_before_dedup}")
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
        print("  1. Verifies full game integrity (fails fast on any error):")
        print("     - Standard starting position (rejects Chess960)")
        print("     - Move connectivity between consecutive positions")
        print("     - Legal moves match python-chess exactly")
        print("     - No dangerous invariance flags")
        print("  2. Computes game_result from terminal evaluation")
        print("  3. Filters positions with half-move clock > 90 (near 50-move draw)")
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
