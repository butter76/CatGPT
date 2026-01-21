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

"""Apache Beam coders for Leela Chess training data.

These coders provide efficient serialization for storing chess positions
and games in .bag files using msgpack.
"""

import abc
import math
from dataclasses import dataclass
from typing import Any

import grain.python as pygrain
import msgpack
import numpy as np
from apache_beam import coders

from catgpt.core.utils import (
    POLICY_SHAPE,
    POLICY_TO_DIM,
    TokenizerConfig,
    encode_policy_target,
    flip_square,
    parse_square,
    parse_uci_move,
    tokenizer,
)


def square_to_index(square_name: str | None, flip: bool = False) -> int:
    """Convert square name to 0-63 index, or -1 if None.

    Args:
        square_name: Square name like "e4", or None.
        flip: Whether to flip the square (for black to move).

    Returns:
        Index 0-63 for valid squares, -1 for None.
    """
    if square_name is None:
        return -1
    if flip:
        square_name = flip_square(square_name)
    return parse_square(square_name)


def convert_old_win_prob_to_new(old_win_prob: float) -> float:
    """Convert win probability from old Lichess formula to new formula.

    Old formula: win_prob = 1 / (1 + exp(-0.00368208 * cp))
    New formula: Q = atan(cp / 90) / 1.5637541897, win_prob = (Q + 1) / 2

    Args:
        old_win_prob: Win probability computed with the old formula.

    Returns:
        Win probability computed with the new formula.
    """
    # Clamp to avoid division by zero or log of zero
    old_win_prob = max(1e-10, min(1 - 1e-10, old_win_prob))

    # Inverse of old formula: cp = ln(win_prob / (1 - win_prob)) / 0.00368208
    cp = math.log(old_win_prob / (1 - old_win_prob)) / 0.00368208

    # New formula
    q = math.atan(cp / 90) / 1.5637541897
    new_win_prob = (q + 1) / 2

    return new_win_prob

STATE_VALUE_CODER = coders.TupleCoder((
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
))

class ConvertToSequence(pygrain.MapTransform, abc.ABC):
  """Base class for converting chess data to a sequence of integers."""

  def __init__(self) -> None:
    super().__init__()

class ConvertStateValueDataToSequence(ConvertToSequence):
  """Converts the fen, move, and win probability into a sequence of integers."""

  def __init__(self, tokenizer_config: TokenizerConfig | None = None) -> None:
    """Initialize with tokenizer configuration.

    Args:
      tokenizer_config: Configuration for tokenization. If None, uses default.
    """
    super().__init__()
    self.tokenizer_config = tokenizer_config or TokenizerConfig()

  def map(
      self, element: bytes
  ):
    fen, old_win_prob = STATE_VALUE_CODER.decode(element)
    state = tokenizer.tokenize(fen, self.tokenizer_config)
    win_prob = convert_old_win_prob_to_new(old_win_prob)
    return state, np.array([win_prob])


class ConvertTrainingBagDataToSequence(ConvertToSequence):
  """Converts training .bag data (from bagz_to_bag.py) to a sequence of integers.

  This coder handles the new .bag format produced by bagz_to_bag.py, which
  contains msgpack-encoded TrainingPositionData with rich meta-features.

  Extracts:
  - fen: tokenized board state
  - root_q: converted to win probability
  - legal_moves: converted to policy target (64, 73) tensor
  - next_capture_square: square index of piece to be captured (auxiliary target)
  - next_pawn_move_square: square index of pawn that will move (auxiliary target)

  Future versions can leverage additional fields like:
  - game_result: final game outcome
  - piece_will_move_to: next move predictions
  - square_will_be_occupied_from: forward board state
  """

  def __init__(
      self,
      tokenizer_config: TokenizerConfig | None = None,
      *,
      include_policy: bool = True,
      include_next_capture: bool = False,
      include_next_pawn_move: bool = False,
  ) -> None:
    """Initialize with tokenizer configuration.

    Args:
      tokenizer_config: Configuration for tokenization. If None, uses default.
      include_policy: Whether to include policy target in output.
      include_next_capture: Whether to include next_capture_square target.
      include_next_pawn_move: Whether to include next_pawn_move_square target.
    """
    super().__init__()
    self.tokenizer_config = tokenizer_config or TokenizerConfig()
    self.include_policy = include_policy
    self.include_next_capture = include_next_capture
    self.include_next_pawn_move = include_next_pawn_move

  def map(self, element: bytes):
    """Map a training position to (state_tokens, win_prob, ...).

    Args:
      element: Msgpack-encoded TrainingPositionData.

    Returns:
      Tuple with variable length depending on configuration:
        - state_tokens: np.ndarray of tokenized FEN
        - win_prob: np.ndarray of shape (1,) with win probability in [0, 1]
        - policy_target: np.ndarray of shape (64, 73) (if include_policy)
        - next_capture_idx: int in [-1, 63] (if include_next_capture)
        - next_pawn_move_idx: int in [-1, 63] (if include_next_pawn_move)
    """
    # Decode msgpack data
    data = msgpack.unpackb(element, raw=False)

    # Extract fields
    fen = data["fen"]
    root_q = data["root_q"]  # Q value in [-1, 1]

    # Parse FEN fields
    fen_parts = fen.split()
    side_to_move = fen_parts[1]
    flip = side_to_move == "b"
    # Full move counter is the 6th field (1-indexed move number)
    fullmove = int(fen_parts[5])

    # Tokenize FEN (tokenizer handles flip internally)
    state = tokenizer.tokenize(fen, self.tokenizer_config)

    # Convert Q to win probability: Q ∈ [-1, 1] → win_prob ∈ [0, 1]
    win_prob = (1.0 + root_q) / 2.0

    # Build result tuple
    result = [state, np.array([win_prob])]

    if self.include_policy:
      # Build policy target from legal moves
      legal_moves = data["legal_moves"]
      policy_target = encode_policy_target(legal_moves, flip=flip)
      result.append(policy_target)

    if self.include_next_capture:
      # Convert square name to index (-1 if None)
      # Only train on positions where fullmove > 15 (opening phase is too noisy)
      if fullmove > 15:
        next_capture_sq = data.get("next_capture_square")
        next_capture_idx = square_to_index(next_capture_sq, flip=flip)
      else:
        next_capture_idx = -1  # Masked out
      result.append(next_capture_idx)

    if self.include_next_pawn_move:
      # Convert square name to index (-1 if None)
      # Only train on positions where fullmove > 15 (opening phase is too noisy)
      if fullmove > 15:
        next_pawn_sq = data.get("next_pawn_move_square")
        next_pawn_idx = square_to_index(next_pawn_sq, flip=flip)
      else:
        next_pawn_idx = -1  # Masked out
      result.append(next_pawn_idx)

    return tuple(result)


@dataclass
class LeelaPositionData:
    """Stripped-down position data for training.

    Contains only the fields needed for training, serialized efficiently.
    """

    fen: str
    legal_moves: list[tuple[str, float]]  # (move_uci, probability)
    invariance_info: int  # Stored as byte for efficiency
    result: int  # +1=win, 0=draw, -1=loss
    root_q: float
    root_d: float
    best_q: float
    best_d: float
    played_q: float
    played_d: float
    orig_q: float
    orig_d: float
    best_move_uci: str | None


def position_to_dict(pos: Any) -> dict[str, Any]:
    """Convert a LeelaPosition to a serializable dict.

    Args:
        pos: A LeelaPosition object from the parser.

    Returns:
        Dictionary with the fields needed for training.
    """
    return {
        "fen": pos.fen,
        "legal_moves": pos.legal_moves,
        "invariance_info": pos.invariance_info.to_byte(),
        "result": pos.result,
        "root_q": pos.root_q,
        "root_d": pos.root_d,
        "best_q": pos.best_q,
        "best_d": pos.best_d,
        "played_q": pos.played_q,
        "played_d": pos.played_d,
        "orig_q": pos.orig_q,
        "orig_d": pos.orig_d,
        "best_move_uci": pos._best_move_uci,
    }


def dict_to_position_data(d: dict[str, Any]) -> LeelaPositionData:
    """Convert a dict back to LeelaPositionData.

    Args:
        d: Dictionary from deserialization.

    Returns:
        LeelaPositionData object.
    """
    return LeelaPositionData(
        fen=d["fen"],
        legal_moves=[(m, p) for m, p in d["legal_moves"]],
        invariance_info=d["invariance_info"],
        result=d["result"],
        root_q=d["root_q"],
        root_d=d["root_d"],
        best_q=d["best_q"],
        best_d=d["best_d"],
        played_q=d["played_q"],
        played_d=d["played_d"],
        orig_q=d["orig_q"],
        orig_d=d["orig_d"],
        best_move_uci=d["best_move_uci"],
    )


class LeelaPositionCoder(coders.Coder):
    """Coder for a single LeelaPositionData using msgpack."""

    def encode(self, position: LeelaPositionData) -> bytes:
        """Encode a position to bytes."""
        data = {
            "fen": position.fen,
            "legal_moves": position.legal_moves,
            "invariance_info": position.invariance_info,
            "result": position.result,
            "root_q": position.root_q,
            "root_d": position.root_d,
            "best_q": position.best_q,
            "best_d": position.best_d,
            "played_q": position.played_q,
            "played_d": position.played_d,
            "orig_q": position.orig_q,
            "orig_d": position.orig_d,
            "best_move_uci": position.best_move_uci,
        }
        return msgpack.packb(data, use_bin_type=True)

    def decode(self, encoded: bytes) -> LeelaPositionData:
        """Decode bytes to a position."""
        data = msgpack.unpackb(encoded, raw=False)
        return dict_to_position_data(data)

    def is_deterministic(self) -> bool:
        return True


class LeelaGameCoder(coders.Coder):
    """Coder for a full game (list of positions) using msgpack.

    Each game is encoded as a list of position dictionaries.
    This is the primary coder for storing games in .bag files.
    """

    def encode(self, game: list[LeelaPositionData]) -> bytes:
        """Encode a game (list of positions) to bytes."""
        positions = []
        for pos in game:
            positions.append({
                "fen": pos.fen,
                "legal_moves": pos.legal_moves,
                "invariance_info": pos.invariance_info,
                "result": pos.result,
                "root_q": pos.root_q,
                "root_d": pos.root_d,
                "best_q": pos.best_q,
                "best_d": pos.best_d,
                "played_q": pos.played_q,
                "played_d": pos.played_d,
                "orig_q": pos.orig_q,
                "orig_d": pos.orig_d,
                "best_move_uci": pos.best_move_uci,
            })
        return msgpack.packb(positions, use_bin_type=True)

    def decode(self, encoded: bytes) -> list[LeelaPositionData]:
        """Decode bytes to a game (list of positions)."""
        positions_data = msgpack.unpackb(encoded, raw=False)
        return [dict_to_position_data(d) for d in positions_data]

    def is_deterministic(self) -> bool:
        return True


# Convenience functions for direct encode/decode without coder instances
_position_coder = LeelaPositionCoder()
_game_coder = LeelaGameCoder()


def encode_position(position: LeelaPositionData) -> bytes:
    """Encode a single position to bytes."""
    return _position_coder.encode(position)


def decode_position(encoded: bytes) -> LeelaPositionData:
    """Decode bytes to a single position."""
    return _position_coder.decode(encoded)


def encode_game(game: list[LeelaPositionData]) -> bytes:
    """Encode a game (list of positions) to bytes."""
    return _game_coder.encode(game)


def decode_game(encoded: bytes) -> list[LeelaPositionData]:
    """Decode bytes to a game (list of positions)."""
    return _game_coder.decode(encoded)
