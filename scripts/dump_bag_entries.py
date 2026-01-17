#!/usr/bin/env python3
"""Dump FEN + win probability entries from a .bag/.bagz file.

This script reads a .bag or .bagz file and outputs the first few entries
showing the FEN position and associated win probability.

Supports both:
- Old format: (fen, win_prob) tuple via STATE_VALUE_CODER
- New format: TrainingPositionData msgpack via decode_training_position
"""

import sys
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagDataSource
from catgpt.core.data.grain.bagz_to_bag import decode_training_position
from catgpt.core.data.grain.coders import STATE_VALUE_CODER, convert_old_win_prob_to_new


def dump_entries(
    bag_path: str | Path,
    num_entries: int = 10,
) -> None:
    """Dump FEN + win probability entries from a bag file.

    Args:
        bag_path: Path to .bag or .bagz file
        num_entries: Number of entries to display
    """
    bag_path = Path(bag_path)

    if not bag_path.exists():
        print(f"Error: File not found: {bag_path}")
        sys.exit(1)

    print(f"Reading: {bag_path}")
    print(f"Type: {'Compressed (.bagz)' if bag_path.suffix == '.bagz' else 'Uncompressed (.bag)'}")
    print()

    # Open the bag file
    source = BagDataSource(str(bag_path))
    total_entries = len(source)

    print(f"Total entries: {total_entries}")
    print()

    # Detect format by trying to decode the first entry
    raw_data = source[0]
    is_new_format = False
    try:
        # Try new format (msgpack with TrainingPositionData)
        data = msgpack.unpackb(raw_data, raw=False)
        if isinstance(data, dict) and "fen" in data and "root_q" in data:
            is_new_format = True
            print("Detected format: NEW (TrainingPositionData from bagz_to_bag.py)")
        else:
            print("Detected format: OLD (STATE_VALUE_CODER tuple)")
    except Exception:
        print("Detected format: OLD (STATE_VALUE_CODER tuple)")

    print()

    entries_to_show = min(num_entries, total_entries)
    print(f"Showing first {entries_to_show} entries:")
    print("=" * 80)

    for i in range(entries_to_show):
        # Get raw bytes from bag
        raw_data = source[i]

        print(f"Entry {i + 1}:")

        if is_new_format:
            # New format: TrainingPositionData
            pos = decode_training_position(raw_data)
            win_prob = (1.0 + pos.root_q) / 2.0

            print(f"  FEN:             {pos.fen}")
            print(f"  Root Q:          {pos.root_q:.4f}")
            print(f"  Root D:          {pos.root_d:.4f}")
            print(f"  Win Prob:        {win_prob:.4f}")
            print(f"  Game Result:     {pos.game_result} (-1=loss, 0=draw, 1=win)")
            print(f"  Best Move:       {pos.best_move_uci or 'None'}")
            print(f"  Legal Moves:     {len(pos.legal_moves)}")

            # Show meta-features if available
            if pos.next_capture_square:
                print(f"  Next Capture:    {pos.next_capture_square}")
            if pos.next_pawn_move_square:
                print(f"  Next Pawn Move:  {pos.next_pawn_move_square}")
            if pos.piece_will_move_to:
                print(f"  Pieces Moving:   {len(pos.piece_will_move_to)}")
        else:
            # Old format: (fen, win_prob) tuple
            fen, old_win_prob = STATE_VALUE_CODER.decode(raw_data)
            new_win_prob = convert_old_win_prob_to_new(old_win_prob)

            print(f"  FEN:             {fen}")
            print(f"  Win Prob (old):  {old_win_prob:.4f}")
            print(f"  Win Prob (new):  {new_win_prob:.4f}")

        print()

    if total_entries > num_entries:
        print(f"... ({total_entries - num_entries} more entries)")


def main():
    """Main entry point."""
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python scripts/dump_bag_entries.py <file.bag|file.bagz> [options]")
        print()
        print("Arguments:")
        print("  file.bag|file.bagz    Path to .bag or .bagz file")
        print()
        print("Options:")
        print("  --num N, -n N         Show N entries (default: 10)")
        print("  --help, -h            Show this help message")
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)

    bag_path = sys.argv[1]

    # Parse arguments
    num_entries = 10
    if "--num" in sys.argv:
        idx = sys.argv.index("--num")
        if idx + 1 < len(sys.argv):
            num_entries = int(sys.argv[idx + 1])
    elif "-n" in sys.argv:
        idx = sys.argv.index("-n")
        if idx + 1 < len(sys.argv):
            num_entries = int(sys.argv[idx + 1])

    dump_entries(bag_path, num_entries=num_entries)


if __name__ == "__main__":
    main()
