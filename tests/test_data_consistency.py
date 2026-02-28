"""Test that both PyTorch (searchless_chess) and JAX (CatGPT) load data identically.

This script compares the data processing between both codebases:
1. Tokenized FEN strings
2. Policy targets (64, 73)
3. HL-Gauss value distributions

Run with:
    cd ~/CatGPT && uv run python tests/test_data_consistency.py
"""

import glob
import sys
from pathlib import Path

# Add packages to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "searchless_chess"))

import msgpack
import numpy as np
from scipy.stats import norm


# ============================================================================
# PyTorch/searchless_chess implementations (minimal, avoiding torch import)
# ============================================================================

# Tokenizer (from searchless_chess/src/tokenizer.py)
PYTORCH_CHARACTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'p', 'b', 'n', 'r', 'c', 'k', 'q',
    'P', 'B', 'N', 'R', 'C', 'Q', 'K',
    'x', '.',
]
PYTORCH_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(PYTORCH_CHARACTERS)}
PYTORCH_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
PYTORCH_SEQUENCE_LENGTH = 64


def pytorch_parse_square(square: str, flip: bool = False) -> int:
    """Mirrors chess.square_mirror(chess.parse_square(square)) from searchless_chess."""
    col = ord(square[0]) - ord('a')
    rank = int(square[1]) - 1
    chess_sq = rank * 8 + col
    if not flip:
        return chess_sq ^ 56
    else:
        return chess_sq


def pytorch_flip_square(square: str) -> str:
    """From searchless_chess/src/data_loader.py."""
    return square[0] + str(9 - int(square[1]))


def pytorch_tokenize(fen: str) -> np.ndarray:
    """From searchless_chess/src/tokenizer.py."""
    raw_board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
    raw_board = raw_board.replace('/', '')
    board = ''
    flip = side == 'b'
    for char in raw_board:
        if char in PYTORCH_SPACES_CHARACTERS:
            board += '.' * int(char)
        else:
            board += char
    for char in castling:
        if char == 'K':
            sq = pytorch_parse_square("h1")
            assert board[sq] == 'R', f"Expected R at h1 (index {sq}), got {board[sq]}"
            board = board[:sq] + 'C' + board[sq + 1:]
        elif char == 'Q':
            sq = pytorch_parse_square("a1")
            assert board[sq] == 'R', f"Expected R at a1 (index {sq}), got {board[sq]}"
            board = board[:sq] + 'C' + board[sq + 1:]
        elif char == 'k':
            sq = pytorch_parse_square("h8")
            assert board[sq] == 'r', f"Expected r at h8 (index {sq}), got {board[sq]}"
            board = board[:sq] + 'c' + board[sq + 1:]
        elif char == 'q':
            sq = pytorch_parse_square("a8")
            assert board[sq] == 'r', f"Expected r at a8 (index {sq}), got {board[sq]}"
            board = board[:sq] + 'c' + board[sq + 1:]
    if flip:
        board = board[56:64] + board[48:56] + board[40:48] + board[32:40] + board[24:32] + board[16:24] + board[8:16] + board[0:8]
        board = board.swapcase()
    if en_passant != '-':
        en_sq = pytorch_parse_square(en_passant, flip=flip)
        assert board[en_sq] == '.', f"Expected . at {en_passant} (index {en_sq}), got {board[en_sq]}"
        board = board[:en_sq] + 'x' + board[en_sq + 1:]

    indices = [PYTORCH_CHARACTERS_INDEX[char] for char in board]
    assert len(indices) == PYTORCH_SEQUENCE_LENGTH
    return np.asarray(indices, dtype=np.uint8)


def pytorch_process_prob(win_prob: float, num_bins: int = 81) -> np.ndarray:
    """From searchless_chess/src/data_loader.py."""
    bin_width = 1.0 / num_bins
    sigma = bin_width * 0.75
    bin_starts = np.arange(0.0, 1.0, bin_width)
    bin_ends = bin_starts + bin_width
    probs = norm.cdf(bin_ends, loc=win_prob, scale=sigma) - norm.cdf(bin_starts, loc=win_prob, scale=sigma)
    probs = probs / probs.sum(keepdims=True)
    return probs


# Policy encoding (from searchless_chess/src/data_loader.py ConvertTrainingBagDataToSequence)
PYTORCH_UNDERPROMO_PIECE_OFFSET = {"n": 0, "b": 1, "r": 2}


def pytorch_encode_policy(legal_moves: list, flip: bool) -> np.ndarray:
    """From searchless_chess/src/data_loader.py _encode_policy_target."""
    target = np.zeros((64, 73), dtype=np.float32)

    for uci_move, prob in legal_moves:
        from_sq = uci_move[:2]
        to_sq = uci_move[2:4]
        promo = uci_move[4:].lower() if len(uci_move) > 4 else None

        if flip:
            from_sq = pytorch_flip_square(from_sq)
            to_sq = pytorch_flip_square(to_sq)

        from_idx = pytorch_parse_square(from_sq, flip=False)

        if promo and promo != 'q':
            file_diff = ord(to_sq[0]) - ord(from_sq[0])
            to_idx = 64 + PYTORCH_UNDERPROMO_PIECE_OFFSET[promo] * 3 + (file_diff + 1)
        else:
            to_idx = pytorch_parse_square(to_sq, flip=False)

        target[from_idx, to_idx] = prob

    return target


# ============================================================================
# JAX/CatGPT implementations
# ============================================================================

from catgpt.core.utils.tokenizer import TokenizerConfig, tokenize as jax_tokenize
from catgpt.core.utils.policy import encode_policy_target as jax_encode_policy
from catgpt.jax.data.dataloader import _hl_gauss_transform_numpy as jax_hl_gauss


# ============================================================================
# Test harness
# ============================================================================

def load_raw_samples(bag_path: str, n_samples: int = 8) -> list[dict]:
    """Load raw samples from a bag file."""
    from catgpt.core.data.grain.bagz import BagDataSource

    source = BagDataSource(bag_path)
    samples = []
    for i in range(min(n_samples, len(source))):
        raw = source[i]
        data = msgpack.unpackb(raw, raw=False)
        samples.append(data)
    return samples


def process_pytorch(sample: dict) -> dict:
    """Process using PyTorch/searchless_chess logic."""
    fen = sample["fen"]
    root_q = sample["root_q"]
    legal_moves = sample["legal_moves"]

    side = fen.split()[1]
    flip = side == "b"

    state = pytorch_tokenize(fen)
    win_prob = (1.0 + root_q) / 2.0
    hl = pytorch_process_prob(win_prob, num_bins=81)
    policy = pytorch_encode_policy(legal_moves, flip)

    return {"state": state, "win_prob": win_prob, "hl": hl, "policy": policy}


def process_jax(sample: dict) -> dict:
    """Process using JAX/CatGPT logic."""
    fen = sample["fen"]
    root_q = sample["root_q"]
    legal_moves = sample["legal_moves"]

    side = fen.split()[1]
    flip = side == "b"

    config = TokenizerConfig(sequence_length=64, include_halfmove=False)
    state = jax_tokenize(fen, config)
    win_prob = (1.0 + root_q) / 2.0
    hl = jax_hl_gauss(np.array([win_prob]), num_bins=81, sigma_ratio=0.75)[0]
    policy = jax_encode_policy(legal_moves, flip)

    return {"state": state, "win_prob": win_prob, "hl": hl, "policy": policy}


def compare_arrays(name: str, pytorch_arr: np.ndarray, jax_arr: np.ndarray, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Compare arrays and print results."""
    if pytorch_arr.shape != jax_arr.shape:
        print(f"  ❌ {name}: Shape mismatch! PyTorch={pytorch_arr.shape}, JAX={jax_arr.shape}")
        return False

    if np.allclose(pytorch_arr, jax_arr, rtol=rtol, atol=atol):
        print(f"  ✅ {name}: MATCH (shape={pytorch_arr.shape})")
        return True
    else:
        diff = np.abs(pytorch_arr.astype(np.float64) - jax_arr.astype(np.float64))
        max_diff = diff.max()
        mean_diff = diff.mean()
        mismatch_count = (diff > atol).sum()
        print(f"  ❌ {name}: MISMATCH (shape={pytorch_arr.shape})")
        print(f"      max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, mismatch_count={mismatch_count}")

        if mismatch_count > 0 and pytorch_arr.ndim >= 1:
            first_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"      First max diff at {first_diff_idx}: PyTorch={pytorch_arr[first_diff_idx]}, JAX={jax_arr[first_diff_idx]}")

        return False


def main():
    # Find a training bag file
    bag_pattern = Path("~/training_bag/training-run1-test80-202507*.bag").expanduser()
    bag_files = sorted(glob.glob(str(bag_pattern)))

    if not bag_files:
        bag_pattern = Path("~/training_bag/*.bag").expanduser()
        bag_files = sorted(glob.glob(str(bag_pattern)))

    if not bag_files:
        print("No bag files found. Please check data paths.")
        return 1

    bag_path = bag_files[0]
    n_samples = 32

    print(f"\n{'='*70}")
    print(f"Testing data consistency: {Path(bag_path).name}")
    print(f"{'='*70}\n")

    print(f"Loading {n_samples} raw samples...")
    samples = load_raw_samples(bag_path, n_samples)
    print(f"Loaded {len(samples)} samples\n")

    all_match = True
    state_match = 0
    hl_match = 0
    policy_match = 0

    for i, sample in enumerate(samples):
        fen = sample["fen"]
        side = fen.split()[1]
        print(f"[{i:2d}] {fen[:45]}... ({side} to move)")

        try:
            pytorch_result = process_pytorch(sample)
            jax_result = process_jax(sample)

            s_ok = compare_arrays("state", pytorch_result["state"], jax_result["state"])
            h_ok = compare_arrays("hl", pytorch_result["hl"], jax_result["hl"], rtol=1e-4, atol=1e-5)
            p_ok = compare_arrays("policy", pytorch_result["policy"], jax_result["policy"], rtol=1e-5, atol=1e-6)

            all_match &= s_ok and h_ok and p_ok
            state_match += int(s_ok)
            hl_match += int(h_ok)
            policy_match += int(p_ok)

            if abs(pytorch_result["win_prob"] - jax_result["win_prob"]) > 1e-8:
                print(f"  ❌ win_prob: {pytorch_result['win_prob']:.8f} vs {jax_result['win_prob']:.8f}")
                all_match = False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_match = False

        print()

    # Summary
    n = len(samples)
    print(f"{'='*70}")
    print(f"Summary: state={state_match}/{n}, hl={hl_match}/{n}, policy={policy_match}/{n}")
    if all_match:
        print("✅ ALL DATA MATCHES - Both codebases process data identically!")
    else:
        print("❌ DATA MISMATCH - Check differences above!")
    print(f"{'='*70}\n")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
