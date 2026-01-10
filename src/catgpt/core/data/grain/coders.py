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

from dataclasses import dataclass
from typing import Any

import msgpack
from apache_beam import coders


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


# Also provide a generic pickle-based coder for arbitrary objects
class PickleCoder(coders.Coder):
    """Generic pickle-based coder for arbitrary Python objects.

    Use this when you need to serialize arbitrary objects and don't
    need deterministic encoding or maximum performance.
    """

    def encode(self, value: Any) -> bytes:
        """Encode a value using msgpack (falls back to pickle for complex types)."""
        import pickle

        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(self, encoded: bytes) -> Any:
        """Decode bytes back to a value."""
        import pickle

        return pickle.loads(encoded)

    def is_deterministic(self) -> bool:
        return False


_pickle_coder = PickleCoder()


def encode_pickle(value: Any) -> bytes:
    """Encode any Python object using pickle."""
    return _pickle_coder.encode(value)


def decode_pickle(encoded: bytes) -> Any:
    """Decode pickle bytes back to a Python object."""
    return _pickle_coder.decode(encoded)
