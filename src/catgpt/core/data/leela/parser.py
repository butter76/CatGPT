"""
Simple Leela Chess Zero V6 Training Data Parser

A minimal parser for LC0 V6 format training data.
Designed for clarity over generality.
"""

import gzip
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np

from .utils import get_input_format_name, get_leela_move_from_idx

# V6 format constants
V6_VERSION = 6
V6_INPUT_FORMAT = 1  # INPUT_112_WITH_CASTLING_PLANE
V6_RECORD_SIZE = 8356
V6_NUM_MOVES = 1858

# Standard starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ============================================================================
# Move Conversion: Leela notation → Standard UCI
# ============================================================================

def flip_move(move: str) -> str:
    """Flip a move for black's perspective.

    Leela stores moves from the side-to-move's perspective.
    When black is to move, ranks are flipped (1↔8, 2↔7, etc.)

    Args:
        move: Move in Leela notation (e.g., "d2d4")

    Returns:
        Move with ranks flipped (e.g., "d7d5")
    """
    if len(move) < 4:
        return move

    from_file = move[0]
    from_rank = str(9 - int(move[1]))
    to_file = move[2]
    to_rank = str(9 - int(move[3]))

    result = from_file + from_rank + to_file + to_rank

    # Handle promotion suffix
    if len(move) > 4:
        result += move[4:]

    return result


def leela_move_to_uci(move: str, board: chess.Board, flip_for_black: bool = False) -> str:
    """Convert a Leela move to standard UCI notation.

    Handles:
    - Perspective flip: Leela stores black's moves from black's POV (ranks flipped)
    - Promotions: Leela uses knight as default (a7a8 = knight), UCI requires explicit piece
    - Castling: Leela uses "king captures rook" (e1h1), UCI uses king destination (e1g1)

    Args:
        move: Leela move string (e.g., "e2e4", "e1h1", "a7a8")
        board: chess.Board in the current position (for castling detection)
        flip_for_black: If True, flip ranks (for black's perspective)

    Returns:
        Standard UCI move string
    """
    if len(move) < 4:
        return move

    # Flip for black's perspective first
    if flip_for_black:
        move = flip_move(move)

    from_sq = move[:2]
    to_sq = move[2:4]
    promo = move[4:] if len(move) > 4 else ""

    # Check if it's a castling move (king captures rook)
    if _is_castling_move(board, from_sq, to_sq):
        return _convert_castling_to_uci(from_sq, to_sq)

    # Check if it's a pawn promotion
    from_rank = from_sq[1]
    to_rank = to_sq[1]

    is_promotion = False
    if (from_rank == '7' and to_rank == '8') or (from_rank == '2' and to_rank == '1'):
        piece = board.piece_at(chess.parse_square(from_sq))
        is_promotion = piece and piece.piece_type == chess.PAWN

    if is_promotion:
        if not promo:
            # Leela's default: no suffix = knight promotion
            return move + 'n'
        # Already has promotion piece (q, r, b)
        return move

    return move


def _is_castling_move(board: chess.Board, from_sq: str, to_sq: str) -> bool:
    """Check if this is a castling move (king captures rook)."""
    # Must have a king on the from square
    piece = board.piece_at(chess.parse_square(from_sq))
    if not piece or piece.piece_type != chess.KING:
        return False

    # Standard castling positions
    return (from_sq == 'e1' and to_sq in ('a1', 'h1')) or (from_sq == 'e8' and to_sq in ('a8', 'h8'))


def _convert_castling_to_uci(from_sq: str, to_sq: str) -> str:
    """Convert Leela castling (king captures rook) to UCI (king destination)."""
    # White kingside: e1h1 -> e1g1
    # White queenside: e1a1 -> e1c1
    # Black kingside: e8h8 -> e8g8
    # Black queenside: e8a8 -> e8c8

    if from_sq == 'e1':
        if to_sq == 'h1':
            return 'e1g1'
        if to_sq == 'a1':
            return 'e1c1'
    elif from_sq == 'e8':
        if to_sq == 'h8':
            return 'e8g8'
        if to_sq == 'a8':
            return 'e8c8'

    # Fallback (shouldn't happen for standard chess)
    return from_sq + to_sq


# ============================================================================
# Planes to FEN Conversion (for verification)
# ============================================================================

def planes_to_fen_without_ep(
    planes: bytes,
    side_to_move: int,
    castling_us_ooo: bool,
    castling_us_oo: bool,
    castling_them_ooo: bool,
    castling_them_oo: bool,
    rule50_count: int,
) -> str:
    """Convert Leela planes to FEN string (without en passant square).

    The planes are stored from the perspective of the side to move.
    We convert to standard FEN (always from white's perspective).

    Args:
        planes: 832 bytes (104 uint64 bitplanes)
        side_to_move: 0 = white, 1 = black
        castling_*: Castling rights
        rule50_count: Halfmove clock

    Returns:
        FEN string with en passant as '-'
    """
    # Parse the first history frame (current position)
    # 13 planes per history step: 6 our pieces + 6 their pieces + 1 repetition
    # Each plane is 8 bytes (uint64)

    board = [['.' for _ in range(8)] for _ in range(8)]

    # Plane order: P, N, B, R, Q, K (ours), p, n, b, r, q, k (theirs)
    piece_order = "PNBRQKpnbrqk"

    for plane_idx in range(12):  # Skip repetition plane (index 12)
        offset = plane_idx * 8
        bitboard = int.from_bytes(planes[offset:offset + 8], byteorder='little')

        # Determine the piece character
        if side_to_move == 0:  # White to move
            # "Our" pieces are white (uppercase), "their" pieces are black (lowercase)
            piece = piece_order[plane_idx]
        else:  # Black to move
            # "Our" pieces are black, "their" pieces are white
            # Also, the board is flipped vertically!
            if plane_idx < 6:
                piece = piece_order[plane_idx].lower()  # Our pieces -> black
            else:
                piece = piece_order[plane_idx].upper()  # Their pieces -> white

        # Place pieces on the board
        # Note: Leela stores bits with file order reversed (bit 0 = h-file, bit 7 = a-file)
        for sq in range(64):
            if bitboard & (1 << sq):
                file_idx = 7 - (sq % 8)  # Reverse file order
                rank_idx = sq // 8

                if side_to_move == 1:  # Black to move: flip the board
                    rank_idx = 7 - rank_idx

                board[rank_idx][file_idx] = piece

    # Convert board to FEN
    fen_ranks = []
    for rank_idx in range(7, -1, -1):  # FEN goes from rank 8 to rank 1
        rank_str = ""
        empty_count = 0
        for file_idx in range(8):
            piece = board[rank_idx][file_idx]
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece
        if empty_count > 0:
            rank_str += str(empty_count)
        fen_ranks.append(rank_str)

    fen_board = '/'.join(fen_ranks)

    # Side to move
    stm = 'w' if side_to_move == 0 else 'b'

    # Castling rights
    castling = ""
    if side_to_move == 0:  # White to move
        if castling_us_oo:
            castling += 'K'
        if castling_us_ooo:
            castling += 'Q'
        if castling_them_oo:
            castling += 'k'
        if castling_them_ooo:
            castling += 'q'
    else:  # Black to move
        if castling_them_oo:
            castling += 'K'
        if castling_them_ooo:
            castling += 'Q'
        if castling_us_oo:
            castling += 'k'
        if castling_us_ooo:
            castling += 'q'
    castling = castling or '-'

    # En passant unknown (set to '-')
    ep = '-'

    # Fullmove number is not stored, use 1 as placeholder
    fullmove = 1

    return f"{fen_board} {stm} {castling} {ep} {rule50_count} {fullmove}"


def detect_en_passant_square(
    fen_without_ep: str,
    leela_uci_moves: set[str],
    verify: bool = False,
) -> str | None:
    """Detect en passant square by comparing Leela moves with python-chess moves.

    Args:
        fen_without_ep: FEN with '-' for en passant
        leela_uci_moves: Set of legal moves from Leela (in UCI)
        verify: If True, raise errors on inconsistencies

    Returns:
        En passant square (e.g., 'e3') or None
    """
    board = chess.Board(fen_without_ep)

    # Get legal moves from python-chess
    chess_moves = {move.uci() for move in board.legal_moves}

    # Leela moves should be a superset (includes en passant if available)
    # Chess moves should be a subset
    extra_leela_moves = leela_uci_moves - chess_moves
    missing_leela_moves = chess_moves - leela_uci_moves

    if verify and missing_leela_moves:
        raise ValueError(
            f"Python-chess has moves that Leela doesn't!\n"
            f"FEN: {fen_without_ep}\n"
            f"Missing from Leela: {missing_leela_moves}\n"
            f"Chess moves: {len(chess_moves)}, Leela moves: {len(leela_uci_moves)}"
        )

    if not extra_leela_moves:
        return None

    # Try each possible en passant square
    for file in 'abcdefgh':
        rank = '6' if board.turn == chess.WHITE else '3'
        ep_square = file + rank

        # Replace the en passant field (4th field in FEN)
        parts = fen_without_ep.split()
        parts[3] = ep_square
        test_fen = ' '.join(parts)

        try:
            test_board = chess.Board(test_fen)
            test_moves = {move.uci() for move in test_board.legal_moves}

            # Check if this explains the extra moves
            if leela_uci_moves == test_moves:
                return ep_square
        except ValueError:
            continue

    # If we get here, we couldn't find an en passant square that explains the difference
    if verify and extra_leela_moves:
        raise ValueError(
            f"Failed to detect en passant square!\n"
            f"FEN: {fen_without_ep}\n"
            f"Extra Leela moves: {extra_leela_moves}\n"
            f"These moves should be explainable by en passant, but no valid ep square was found."
        )

    return None


def is_standard_starting_position(fen: str) -> bool:
    """Check if a FEN represents the standard chess starting position.

    Used to detect Fischer Random (Chess960) games.
    """
    # Compare just the piece placement
    parts = fen.split()
    start_parts = STARTING_FEN.split()

    # Check piece placement
    return parts[0] == start_parts[0]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InvarianceInfo:
    """Decoded invariance_info bitfield."""

    flip_transform: bool  # bit 0: vertical flip (ranks 1↔8)
    mirror_transform: bool  # bit 1: horizontal flip (files a↔h)
    transpose_transform: bool  # bit 2: diagonal flip
    best_q_proven: bool  # bit 3: best_q from tablebase
    max_game_length_exceeded: bool  # bit 4: game cut short
    game_adjudicated: bool  # bit 5: game ended by adjudication
    marked_for_deletion: bool  # bit 6: flagged by rescorer
    side_to_move_black: bool  # bit 7: for canonicalized formats

    @classmethod
    def from_byte(cls, b: int) -> "InvarianceInfo":
        return cls(
            flip_transform=bool(b & 0x01),
            mirror_transform=bool(b & 0x02),
            transpose_transform=bool(b & 0x04),
            best_q_proven=bool(b & 0x08),
            max_game_length_exceeded=bool(b & 0x10),
            game_adjudicated=bool(b & 0x20),
            marked_for_deletion=bool(b & 0x40),
            side_to_move_black=bool(b & 0x80),
        )

    def to_byte(self) -> int:
        return (
            (self.flip_transform << 0)
            | (self.mirror_transform << 1)
            | (self.transpose_transform << 2)
            | (self.best_q_proven << 3)
            | (self.max_game_length_exceeded << 4)
            | (self.game_adjudicated << 5)
            | (self.marked_for_deletion << 6)
            | (self.side_to_move_black << 7)
        )


@dataclass
class LeelaPosition:
    """A single training position from LC0 V6 data.

    All fields match the official LC0 training data format.
    Moves are converted to standard UCI notation.
    """

    # ===== FEN (derived from previous move) =====
    fen: str  # Standard FEN string

    # ===== Policy (1858 floats) =====
    # List of (move_uci, probability) for legal moves only, sorted by prob desc
    # Moves are in standard UCI notation
    legal_moves: list[tuple[str, float]]

    # ===== Board State =====
    planes: bytes  # 832 bytes: 104 uint64 bitplanes (raw)
    castling_us_ooo: bool
    castling_us_oo: bool
    castling_them_ooo: bool
    castling_them_oo: bool
    side_to_move: int  # 0 = white, 1 = black
    rule50_count: int
    invariance_info: InvarianceInfo

    # ===== Game Result =====
    result: int  # +1=win, 0=draw, -1=loss (from STM perspective)
    result_q: float
    result_d: float

    # ===== Evaluations =====
    root_q: float
    root_d: float
    root_m: float
    best_q: float
    best_d: float
    best_m: float
    played_q: float
    played_d: float
    played_m: float
    orig_q: float
    orig_d: float
    orig_m: float

    # ===== MCTS Statistics =====
    plies_left: float
    visits: int

    # ===== Move Indices =====
    played_idx: int
    best_idx: int

    # ===== Cached UCI moves =====
    _played_move_uci: str | None = field(default=None, repr=False)
    _best_move_uci: str | None = field(default=None, repr=False)

    @property
    def played_move(self) -> str | None:
        """UCI string of the move that was actually played."""
        return self._played_move_uci

    @property
    def best_move(self) -> str | None:
        """UCI string of the best move according to MCTS."""
        return self._best_move_uci

    @property
    def wdl(self) -> tuple[float, float, float]:
        """Win/Draw/Loss probabilities from best_q and best_d."""
        win = 0.5 * (1.0 - self.best_d + self.best_q)
        loss = 0.5 * (1.0 - self.best_d - self.best_q)
        return (win, self.best_d, loss)

    @property
    def result_wdl(self) -> tuple[float, float, float]:
        """Win/Draw/Loss probabilities from game result."""
        win = 0.5 * (1.0 - self.result_d + self.result_q)
        loss = 0.5 * (1.0 - self.result_d - self.result_q)
        return (win, self.result_d, loss)


# ============================================================================
# Parsing
# ============================================================================

def parse_record(
    data: bytes,
    board: chess.Board | None = None,
    verify_with_planes: bool = False,
) -> LeelaPosition:
    """Parse a single V6 training record.

    Args:
        data: Exactly 8356 bytes of V6 record data
        board: chess.Board from the previous position (to generate FEN from played move)
               If None, generates FEN from planes
        verify_with_planes: If True, verify FEN from board matches planes

    Returns:
        Parsed LeelaPosition with all fields populated
    """
    assert len(data) == V6_RECORD_SIZE, f"Expected {V6_RECORD_SIZE} bytes, got {len(data)}"

    # ===== Header (bytes 0-7) =====
    version = struct.unpack_from('<i', data, 0)[0]
    input_format = struct.unpack_from('<i', data, 4)[0]

    assert version == V6_VERSION, f"Expected version {V6_VERSION}, got {version}"
    assert input_format == V6_INPUT_FORMAT, (
        f"Expected input format {V6_INPUT_FORMAT} ({get_input_format_name(V6_INPUT_FORMAT)}), "
        f"got {input_format} ({get_input_format_name(input_format)})"
    )

    # ===== Planes (bytes 7440-8271): 832 bytes =====
    planes_offset = 8 + (V6_NUM_MOVES * 4)  # 7440
    planes = data[planes_offset:planes_offset + 832]

    # ===== Game State (bytes 8272-8279): 8 bytes =====
    state_offset = planes_offset + 832  # 8272
    (
        castling_us_ooo,
        castling_us_oo,
        castling_them_ooo,
        castling_them_oo,
        side_to_move,
        rule50_count,
        invariance_info_byte,
        result,
    ) = struct.unpack_from('<BBBBBBBb', data, state_offset)

    invariance_info = InvarianceInfo.from_byte(invariance_info_byte)

    # ===== Policy (bytes 8-7439): 1858 floats =====
    probs_raw = np.frombuffer(data, dtype='<f4', count=V6_NUM_MOVES, offset=8)

    # Determine if we need to flip moves (black's perspective)
    is_black_to_move = side_to_move == 1

    # If we have a board from previous position, use it to generate FEN
    if board is not None:
        fen = board.fen()
        current_board = board.copy()
    else:
        # Generate FEN from planes (without en passant)
        fen = planes_to_fen_without_ep(
            planes,
            side_to_move,
            bool(castling_us_ooo),
            bool(castling_us_oo),
            bool(castling_them_ooo),
            bool(castling_them_oo),
            rule50_count,
        )
        current_board = chess.Board(fen)

    # Extract legal moves and convert to UCI
    legal_moves = []
    leela_move_set = set()
    for idx, prob in enumerate(probs_raw):
        if prob > -0.99:  # Legal move (not -1.0)
            leela_move = get_leela_move_from_idx(idx)
            uci_move = leela_move_to_uci(leela_move, current_board, flip_for_black=is_black_to_move)
            legal_moves.append((uci_move, float(prob)))
            leela_move_set.add(uci_move)
    legal_moves.sort(key=lambda x: x[1], reverse=True)

    # If FEN was generated from planes, detect en passant
    if board is None:
        ep_square = detect_en_passant_square(fen, leela_move_set, verify=False)
        if ep_square:
            parts = fen.split()
            parts[3] = ep_square
            fen = ' '.join(parts)
            current_board = chess.Board(fen)

    # Verify with planes if requested
    if verify_with_planes and board is not None:
        planes_fen_without_ep = planes_to_fen_without_ep(
            planes,
            side_to_move,
            bool(castling_us_ooo),
            bool(castling_us_oo),
            bool(castling_them_ooo),
            bool(castling_them_oo),
            rule50_count,
        )
        ep_square = detect_en_passant_square(planes_fen_without_ep, leela_move_set, verify=True)
        planes_fen = planes_fen_without_ep
        if ep_square:
            parts = planes_fen_without_ep.split()
            parts[3] = ep_square
            planes_fen = ' '.join(parts)

        # Compare (ignoring fullmove number)
        board_parts = fen.split()
        planes_parts = planes_fen.split()
        if board_parts[:5] != planes_parts[:5]:
            raise ValueError(
                f"FEN mismatch!\n"
                f"From board:  {fen}\n"
                f"From planes: {planes_fen}"
            )

    # ===== Evaluation Floats (bytes 8280-8339): 15 floats = 60 bytes =====
    eval_offset = state_offset + 8  # 8280
    (
        root_q, best_q, root_d, best_d, root_m, best_m, plies_left,
        result_q, result_d, played_q, played_d, played_m,
        orig_q, orig_d, orig_m,
    ) = struct.unpack_from('<fffffffffffffff', data, eval_offset)

    # ===== MCTS Stats (bytes 8340-8347): visits + indices =====
    stats_offset = eval_offset + 60  # 8340
    (visits,) = struct.unpack_from('<I', data, stats_offset)

    indices_offset = stats_offset + 4  # 8344
    (played_idx, best_idx) = struct.unpack_from('<HH', data, indices_offset)

    # Convert played and best moves to UCI
    played_move_uci = None
    best_move_uci = None

    if 0 <= played_idx < V6_NUM_MOVES:
        leela_played = get_leela_move_from_idx(played_idx)
        played_move_uci = leela_move_to_uci(leela_played, current_board, flip_for_black=is_black_to_move)

    if 0 <= best_idx < V6_NUM_MOVES:
        leela_best = get_leela_move_from_idx(best_idx)
        best_move_uci = leela_move_to_uci(leela_best, current_board, flip_for_black=is_black_to_move)

    return LeelaPosition(
        fen=fen,
        legal_moves=legal_moves,
        planes=planes,
        castling_us_ooo=bool(castling_us_ooo),
        castling_us_oo=bool(castling_us_oo),
        castling_them_ooo=bool(castling_them_ooo),
        castling_them_oo=bool(castling_them_oo),
        side_to_move=side_to_move,
        rule50_count=rule50_count,
        invariance_info=invariance_info,
        result=result,
        result_q=result_q,
        result_d=result_d,
        root_q=root_q,
        root_d=root_d,
        root_m=root_m,
        best_q=best_q,
        best_d=best_d,
        best_m=best_m,
        played_q=played_q,
        played_d=played_d,
        played_m=played_m,
        orig_q=orig_q,
        orig_d=orig_d,
        orig_m=orig_m,
        plies_left=plies_left,
        visits=visits,
        played_idx=played_idx,
        best_idx=best_idx,
        _played_move_uci=played_move_uci,
        _best_move_uci=best_move_uci,
    )


def read_chunk(
    filepath: str | Path,
    skip_chess960: bool = True,
    verify_with_planes: bool = False,
) -> Iterator[LeelaPosition]:
    """Read all positions from a Leela chunk file.

    Args:
        filepath: Path to .gz chunk file
        skip_chess960: If True, skip games that start from non-standard positions
        verify_with_planes: If True, verify FEN from moves matches planes

    Yields:
        LeelaPosition for each record in the chunk
    """
    filepath = Path(filepath)

    # Open with gzip if needed
    open_fn = gzip.open if filepath.suffix == '.gz' else open

    with open_fn(filepath, 'rb') as f:
        data = f.read()

    # Validate we have complete records
    num_records = len(data) // V6_RECORD_SIZE
    remainder = len(data) % V6_RECORD_SIZE

    if remainder != 0:
        raise ValueError(
            f"File size {len(data)} is not a multiple of record size {V6_RECORD_SIZE}. "
            f"Remainder: {remainder} bytes"
        )

    if num_records == 0:
        return

    # Parse first position (without board)
    first_record = data[:V6_RECORD_SIZE]
    first_pos = parse_record(first_record, board=None, verify_with_planes=False)

    # Check for Chess960
    if skip_chess960 and not is_standard_starting_position(first_pos.fen):
        return

    # Yield first position
    yield first_pos

    # Create board and apply moves sequentially
    board = chess.Board(first_pos.fen)
    if first_pos.played_move:
        board.push_uci(first_pos.played_move)

    # Parse remaining positions
    for i in range(1, num_records):
        offset = i * V6_RECORD_SIZE
        record_data = data[offset:offset + V6_RECORD_SIZE]
        pos = parse_record(record_data, board=board, verify_with_planes=verify_with_planes)
        yield pos

        # Apply played move for next position
        if pos.played_move:
            board.push_uci(pos.played_move)


def read_chunks_from_tar(
    tar_path: str | Path,
    max_games: int | None = None,
    skip_chess960: bool = True,
    verify_with_planes: bool = False,
) -> Iterator[list[LeelaPosition]]:
    """Read games from a tar archive of chunk files.

    Args:
        tar_path: Path to .tar file containing .gz chunks
        max_games: Maximum number of games to yield (None = all)
        skip_chess960: Skip Fischer Random games
        verify_with_planes: Verify FEN consistency with planes

    Yields:
        List of positions for each game (one .gz file = one game)
    """
    import tarfile

    tar_path = Path(tar_path)
    games_yielded = 0

    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.gz'):
                continue

            if max_games is not None and games_yielded >= max_games:
                return

            # Extract and parse
            f = tar.extractfile(member)
            if f is None:
                continue

            decompressed = gzip.decompress(f.read())
            num_records = len(decompressed) // V6_RECORD_SIZE

            if num_records == 0:
                continue

            # Parse first position
            first_pos = parse_record(decompressed[:V6_RECORD_SIZE], board=None)

            # Check for Chess960
            if skip_chess960 and not is_standard_starting_position(first_pos.fen):
                continue

            # Parse all positions in this game
            positions = [first_pos]
            board = chess.Board(first_pos.fen)
            if first_pos.played_move:
                board.push_uci(first_pos.played_move)

            for i in range(1, num_records):
                offset = i * V6_RECORD_SIZE
                record_data = decompressed[offset:offset + V6_RECORD_SIZE]
                pos = parse_record(record_data, board=board, verify_with_planes=verify_with_planes)
                positions.append(pos)

                if pos.played_move:
                    board.push_uci(pos.played_move)

            yield positions
            games_yielded += 1


# ============================================================================
# Formatting
# ============================================================================

def format_position(pos: LeelaPosition, verbose: bool = False) -> str:
    """Format a position as a human-readable string."""
    lines = []

    # FEN
    lines.append(f"FEN: {pos.fen}")

    # Basic info
    stm = "black" if pos.side_to_move else "white"
    result_str = {1: "win", 0: "draw", -1: "loss"}.get(pos.result, f"?{pos.result}")
    lines.append(f"Side to move: {stm} | Result: {result_str} | Rule50: {pos.rule50_count}")

    # Evaluation
    win, draw, loss = pos.wdl
    lines.append(f"Eval: Q={pos.best_q:+.3f} D={pos.best_d:.3f} (W={win:.1%} D={draw:.1%} L={loss:.1%})")

    # Moves
    lines.append(f"Best move: {pos.best_move} | Played: {pos.played_move}")

    # Top policy moves
    top_moves = pos.legal_moves[:5]
    moves_str = " ".join(f"{m}:{p:.2%}" for m, p in top_moves)
    lines.append(f"Top policy: {moves_str}")

    if verbose:
        lines.append("")
        lines.append(f"MCTS visits: {pos.visits}")
        lines.append(f"Root:   Q={pos.root_q:+.3f} D={pos.root_d:.3f} M={pos.root_m:.1f}")
        lines.append(f"Best:   Q={pos.best_q:+.3f} D={pos.best_d:.3f} M={pos.best_m:.1f}")
        lines.append(f"Played: Q={pos.played_q:+.3f} D={pos.played_d:.3f} M={pos.played_m:.1f}")
        lines.append(f"Orig:   Q={pos.orig_q:+.3f} D={pos.orig_d:.3f} M={pos.orig_m:.1f}")
        lines.append(f"Result: Q={pos.result_q:+.3f} D={pos.result_d:.3f}")
        lines.append(f"Plies left: {pos.plies_left:.0f}")

        inv = pos.invariance_info
        flags = []
        if inv.best_q_proven:
            flags.append("tablebase")
        if inv.game_adjudicated:
            flags.append("adjudicated")
        if inv.max_game_length_exceeded:
            flags.append("max_length")
        if inv.marked_for_deletion:
            flags.append("DELETED")
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")

    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python -m catgpt.core.data.leela.parser <input> [options]")
        print()
        print("Arguments:")
        print("  input              Path to .gz (single game) or .tar (multiple games)")
        print()
        print("Options:")
        print("  --verbose, -v      Show detailed position info")
        print("  --verify           Verify FEN from moves matches planes")
        print("  --max-games N      Process at most N games (tar only)")
        print("  --stats            Show statistics instead of positions")
        print("  --no-skip-960      Include Chess960 games (default: skip)")
        sys.exit(0 if '--help' in sys.argv or '-h' in sys.argv else 1)

    input_file = sys.argv[1]
    input_path = Path(input_file)

    # Parse arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    verify = '--verify' in sys.argv
    show_stats = '--stats' in sys.argv
    skip_chess960 = '--no-skip-960' not in sys.argv

    max_games = None
    if '--max-games' in sys.argv:
        idx = sys.argv.index('--max-games')
        if idx + 1 < len(sys.argv):
            max_games = int(sys.argv[idx + 1])

    # Detect input type
    is_tar = input_path.suffix == '.tar'

    print(f"Parsing: {input_file}")
    print(f"Type: {'TAR archive' if is_tar else 'Single game'}")
    if verify:
        print("Verification: ON (checking FEN from planes)")
    if not skip_chess960:
        print("Chess960: INCLUDED")
    if max_games:
        print(f"Max games: {max_games}")
    print()

    if is_tar:
        # Process tar archive
        if show_stats:
            # Statistics mode
            game_count = 0
            position_count = 0
            ep_count = 0
            skip_count = 0

            try:
                for game in read_chunks_from_tar(
                    input_file,
                    max_games=max_games,
                    skip_chess960=skip_chess960,
                    verify_with_planes=verify
                ):
                    game_count += 1
                    for pos in game:
                        position_count += 1
                        ep_square = pos.fen.split()[3]
                        if ep_square != '-':
                            ep_count += 1

                    if game_count % 100 == 0:
                        print(f"Processed {game_count} games, {position_count} positions...", end='\r')

                print()  # Clear progress line
                print(f"✓ Processed {game_count} games")
                print(f"✓ Total positions: {position_count}")
                print(f"✓ Positions with en passant: {ep_count}")
                if position_count > 0:
                    print(f"✓ Average game length: {position_count / game_count:.1f} positions")

                if verify:
                    print()
                    print("✓ All verifications passed!")

            except KeyboardInterrupt:
                print()
                print(f"\nInterrupted after {game_count} games")
                print(f"Total positions: {position_count}")

        else:
            # Show positions mode
            for game_num, game in enumerate(read_chunks_from_tar(
                input_file,
                max_games=max_games or 5,  # Default to 5 games if not specified
                skip_chess960=skip_chess960,
                verify_with_planes=verify
            ), 1):
                print(f"{'='*60}")
                print(f"Game {game_num}")
                print(f"{'='*60}")
                print()

                for i, pos in enumerate(game):
                    if not verbose and i >= 5:
                        print(f"... ({len(game)} total positions, use --verbose to see all)")
                        break

                    print(f"=== Position {i} ===")
                    print(format_position(pos, verbose=verbose))
                    print()

                if not verbose:
                    break

    else:
        # Process single game
        if show_stats:
            positions = list(read_chunk(input_file, skip_chess960=skip_chess960, verify_with_planes=verify))
            ep_count = sum(1 for pos in positions if pos.fen.split()[3] != '-')

            print(f"✓ Positions: {len(positions)}")
            print(f"✓ Positions with en passant: {ep_count}")

            if verify:
                print()
                print("✓ All verifications passed!")

        else:
            for i, pos in enumerate(read_chunk(input_file, skip_chess960=skip_chess960, verify_with_planes=verify)):
                print(f"=== Position {i} ===")
                print(format_position(pos, verbose=verbose))
                print()

                # Only show first 5 by default
                if i >= 4 and not verbose:
                    print("... (use --verbose to see all positions)")
                    break
