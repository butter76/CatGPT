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

"""Convert Leela Chess .tar files to .bag format for PyGrain.

Each .tar file contains multiple .gz files, where each .gz is a single game.
This script converts them to a .bag file where each record is a serialized
game (list of positions).
"""

from pathlib import Path

from catgpt.core.data.grain.bagz import BagWriter
from catgpt.core.data.grain.coders import (
    LeelaPositionData,
    encode_game,
    position_to_dict,
)
from catgpt.core.data.leela.parser import read_chunks_from_tar


def convert_tar_to_bag(
    tar_path: str | Path,
    output_path: str | Path | None = None,
    *,
    max_games: int | None = None,
    skip_chess960: bool = True,
    compress: bool = False,
    verbose: bool = False,
) -> int:
    """Convert a Leela Chess .tar file to .bag/.bagz format.

    Each game (originally a .gz file in the tar) becomes a single record
    in the .bag file, serialized using LeelaGameCoder.

    Args:
        tar_path: Path to input .tar file containing .gz game files.
        output_path: Path for output file. If None, uses tar_path with
            .bag or .bagz extension (depending on compress flag).
        max_games: Maximum number of games to convert. None = all games.
        skip_chess960: If True, skip Fischer Random (Chess960) games.
        compress: If True, create compressed .bagz file. Otherwise, create
            uncompressed .bag file for faster training.
        verbose: If True, print progress information.

    Returns:
        Number of games written to the output file.
    """
    tar_path = Path(tar_path)

    if output_path is None:
        suffix = ".bagz" if compress else ".bag"
        output_path = tar_path.with_suffix(suffix)
    else:
        output_path = Path(output_path)

    games_written = 0

    with BagWriter(str(output_path), compress=compress) as writer:
        for game_positions in read_chunks_from_tar(
            tar_path,
            max_games=max_games,
            skip_chess960=skip_chess960,
            verify_with_planes=False,
        ):
            # Convert LeelaPosition objects to LeelaPositionData
            game_data: list[LeelaPositionData] = []
            for pos in game_positions:
                pos_dict = position_to_dict(pos)
                game_data.append(
                    LeelaPositionData(
                        fen=pos_dict["fen"],
                        legal_moves=pos_dict["legal_moves"],
                        invariance_info=pos_dict["invariance_info"],
                        result=pos_dict["result"],
                        root_q=pos_dict["root_q"],
                        root_d=pos_dict["root_d"],
                        best_q=pos_dict["best_q"],
                        best_d=pos_dict["best_d"],
                        played_q=pos_dict["played_q"],
                        played_d=pos_dict["played_d"],
                        orig_q=pos_dict["orig_q"],
                        orig_d=pos_dict["orig_d"],
                        best_move_uci=pos_dict["best_move_uci"],
                    )
                )

            # Encode and write the game
            encoded = encode_game(game_data)
            writer.write(encoded)
            games_written += 1

            if verbose and games_written % 100 == 0:
                print(f"Converted {games_written} games...")

    if verbose:
        print(f"✓ Wrote {games_written} games to {output_path}")

    return games_written


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python -m catgpt.core.data.grain.tar_to_bag <input.tar> [options]")
        print()
        print("Arguments:")
        print("  input.tar          Path to input .tar file containing .gz game files")
        print()
        print("Options:")
        print("  --output PATH      Output file path (default: input.bag or input.bagz)")
        print("  --max-games N      Convert at most N games")
        print("  --compress, -c     Create compressed .bagz file (default: uncompressed .bag)")
        print("  --no-skip-960      Include Chess960 games (default: skip)")
        print("  --verbose, -v      Print progress information")
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)

    input_tar = sys.argv[1]

    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    compress = "--compress" in sys.argv or "-c" in sys.argv
    skip_chess960 = "--no-skip-960" not in sys.argv

    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    max_games = None
    if "--max-games" in sys.argv:
        idx = sys.argv.index("--max-games")
        if idx + 1 < len(sys.argv):
            max_games = int(sys.argv[idx + 1])

    # Run conversion
    games = convert_tar_to_bag(
        input_tar,
        output_path=output_path,
        max_games=max_games,
        skip_chess960=skip_chess960,
        compress=compress,
        verbose=verbose,
    )

    print(f"Converted {games} games from {input_tar}")
