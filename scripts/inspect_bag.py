#!/usr/bin/env python3
"""Inspect .bag/.bagz files containing Leela Chess training data.

This script reads a .bag or .bagz file, decodes the games using LeelaGameCoder,
and displays the contents in a human-readable format.
"""

import sys
from pathlib import Path

from catgpt.core.data.grain import BagReader, decode_game
from catgpt.core.data.leela.parser import InvarianceInfo


def format_position(pos, position_idx: int) -> str:
    """Format a position in a human-readable way."""
    lines = []
    lines.append(f"  Position {position_idx}:")
    lines.append(f"    FEN: {pos.fen}")

    # Result and Q/D values
    result_str = {1: "WIN", 0: "DRAW", -1: "LOSS"}.get(pos.result, f"?{pos.result}")
    lines.append(f"    Result: {result_str}")
    lines.append(f"    Root:   Q={pos.root_q:+.3f}  D={pos.root_d:.3f}")
    lines.append(f"    Best:   Q={pos.best_q:+.3f}  D={pos.best_d:.3f}")
    lines.append(f"    Played: Q={pos.played_q:+.3f}  D={pos.played_d:.3f}")
    lines.append(f"    Orig:   Q={pos.orig_q:+.3f}  D={pos.orig_d:.3f}")

    # Best move
    if pos.best_move_uci:
        lines.append(f"    Best move: {pos.best_move_uci}")

    # Top policy moves
    top_moves = pos.legal_moves[:5]
    if top_moves:
        moves_str = ", ".join(f"{m}:{p:.1%}" for m, p in top_moves)
        lines.append(f"    Top policy: {moves_str}")

    # Invariance info
    inv = InvarianceInfo.from_byte(pos.invariance_info)
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
        lines.append(f"    Flags: {', '.join(flags)}")

    return "\n".join(lines)


def inspect_bag(
    bag_path: str | Path,
    max_games: int = 5,
    max_positions_per_game: int = 10,
    verbose: bool = False,
) -> None:
    """Inspect a .bag or .bagz file and display its contents.

    Args:
        bag_path: Path to .bag or .bagz file
        max_games: Maximum number of games to display
        max_positions_per_game: Maximum positions to show per game
        verbose: If True, show all positions in each game
    """
    bag_path = Path(bag_path)

    if not bag_path.exists():
        print(f"Error: File not found: {bag_path}")
        sys.exit(1)

    print(f"Reading: {bag_path}")
    print(f"Type: {'Compressed (.bagz)' if bag_path.suffix == '.bagz' else 'Uncompressed (.bag)'}")
    print()

    reader = BagReader(str(bag_path))
    total_games = len(reader)

    print(f"Total games in file: {total_games}")
    print()

    games_to_show = min(max_games, total_games)

    for game_idx in range(games_to_show):
        print(f"{'='*70}")
        print(f"Game {game_idx + 1} of {total_games}")
        print(f"{'='*70}")

        # Read and decode the game
        encoded_game = reader[game_idx]
        game = decode_game(encoded_game)

        print(f"Positions in game: {len(game)}")
        print()

        # Show positions
        positions_to_show = len(game) if verbose else min(max_positions_per_game, len(game))

        for pos_idx in range(positions_to_show):
            print(format_position(game[pos_idx], pos_idx))
            print()

        if not verbose and len(game) > max_positions_per_game:
            remaining = len(game) - max_positions_per_game
            print(f"  ... ({remaining} more positions, use --verbose to see all)")
            print()

    if total_games > max_games:
        print(f"\n... ({total_games - max_games} more games, use --max-games to see more)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python scripts/inspect_bag.py <file.bag|file.bagz> [options]")
        print()
        print("Arguments:")
        print("  file.bag|file.bagz    Path to .bag or .bagz file to inspect")
        print()
        print("Options:")
        print("  --max-games N         Show at most N games (default: 5)")
        print("  --max-positions N     Show at most N positions per game (default: 10)")
        print("  --verbose, -v         Show all positions in all games")
        print("  --help, -h            Show this help message")
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)

    bag_path = sys.argv[1]

    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    max_games = 5
    if "--max-games" in sys.argv:
        idx = sys.argv.index("--max-games")
        if idx + 1 < len(sys.argv):
            max_games = int(sys.argv[idx + 1])

    max_positions = 10
    if "--max-positions" in sys.argv:
        idx = sys.argv.index("--max-positions")
        if idx + 1 < len(sys.argv):
            max_positions = int(sys.argv[idx + 1])

    # Run inspection
    inspect_bag(bag_path, max_games=max_games, max_positions_per_game=max_positions, verbose=verbose)


if __name__ == "__main__":
    main()
