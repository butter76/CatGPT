/**
 * Policy encoding utilities for chess move distributions.
 *
 * This module provides utilities for encoding/decoding moves to/from the policy
 * tensor format used by the model's policy head.
 *
 * Policy tensor shape: (64, 73)
 * - 64 = from_square (source square, row-major a8=0 to h1=63)
 * - 73 = to_square encoding:
 *   - 0-63: Normal destination squares (including queen promotions)
 *   - 64-66: Knight underpromotions (left capture, straight, right capture)
 *   - 67-69: Bishop underpromotions (left capture, straight, right capture)
 *   - 70-72: Rook underpromotions (left capture, straight, right capture)
 *
 * Note: The tokenizer flips positions when black is to move, so policy indices
 * must also be flipped accordingly.
 */

#ifndef CATGPT_ENGINE_POLICY_HPP
#define CATGPT_ENGINE_POLICY_HPP

#include <array>
#include <cstdint>
#include <utility>

#include "../../external/chess-library/include/chess.hpp"

namespace catgpt {

// Policy tensor dimensions
inline constexpr int POLICY_FROM_DIM = 64;
inline constexpr int POLICY_TO_DIM = 73;
inline constexpr int POLICY_SIZE = POLICY_FROM_DIM * POLICY_TO_DIM;  // 4672

namespace detail {

// Files and ranks for square parsing
inline constexpr std::array<char, 8> FILES = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
inline constexpr std::array<char, 8> RANKS = {'1', '2', '3', '4', '5', '6', '7', '8'};

/**
 * Convert algebraic notation to board index (0-63, row-major from a8 to h1).
 *
 * Board indexing convention (row-major from a8):
 *   a8=0,  b8=1,  c8=2,  d8=3,  e8=4,  f8=5,  g8=6,  h8=7
 *   a7=8,  b7=9,  c7=10, d7=11, e7=12, f7=13, g7=14, h7=15
 *   ...
 *   a1=56, b1=57, c1=58, d1=59, e1=60, f1=61, g1=62, h1=63
 */
constexpr int parse_square(char file_char, char rank_char) noexcept {
    int file_idx = file_char - 'a';  // 0-7 (a-h)
    int rank_idx = rank_char - '1';  // 0-7 (1-8)

    // Convert to row-major index (a8=0, h8=7, a7=8, ..., h1=63)
    // rank 8 is row 0, rank 1 is row 7
    int row = 7 - rank_idx;
    return row * 8 + file_idx;
}

/**
 * Flip a square vertically (swap ranks 1↔8, 2↔7, etc.).
 */
constexpr int flip_square_index(int idx) noexcept {
    int file = idx % 8;
    int row = idx / 8;
    return (7 - row) * 8 + file;
}

/**
 * Convert chess library square to policy index.
 * Chess library uses a1=0, h8=63 (rank-major from a1).
 * Policy uses a8=0, h1=63 (row-major from a8).
 */
constexpr int chess_sq_to_policy_idx(int chess_sq) noexcept {
    int file = chess_sq % 8;
    int rank = chess_sq / 8;  // 0=rank1, 7=rank8
    // Policy: row 0 = rank 8, row 7 = rank 1
    int row = 7 - rank;
    return row * 8 + file;
}

}  // namespace detail

/**
 * Convert a chess::Move to policy tensor indices (from_idx, to_idx).
 *
 * @param move A chess::Move object.
 * @param flip Whether to flip squares (for black to move, to match tokenizer).
 * @return Pair of (from_idx, to_idx) for indexing into policy tensor.
 *         from_idx is in [0, 63], to_idx is in [0, 72].
 */
inline std::pair<int, int> encode_move_to_policy_index(const chess::Move& move, bool flip = false) {
    // Get from and to squares from chess library (a1=0 to h8=63)
    int from_chess = move.from().index();
    int to_chess = move.to().index();

    // Convert to policy indices (a8=0 to h1=63)
    int from_idx = detail::chess_sq_to_policy_idx(from_chess);
    int to_idx = detail::chess_sq_to_policy_idx(to_chess);

    // Handle underpromotions (non-queen promotions)
    if (move.typeOf() == chess::Move::PROMOTION) {
        auto promo_type = move.promotionType();

        // Queen promotions use normal destination square
        if (promo_type != chess::PieceType::QUEEN) {
            // Calculate file difference for underpromotion direction
            int from_file = from_chess % 8;
            int to_file = to_chess % 8;
            int file_diff = to_file - from_file;  // -1, 0, or +1

            // Map to underpromotion indices 64-72
            // Knight: 64-66, Bishop: 67-69, Rook: 70-72
            int piece_offset = 0;
            if (promo_type == chess::PieceType::KNIGHT) {
                piece_offset = 0;
            } else if (promo_type == chess::PieceType::BISHOP) {
                piece_offset = 1;
            } else if (promo_type == chess::PieceType::ROOK) {
                piece_offset = 2;
            }

            to_idx = 64 + piece_offset * 3 + (file_diff + 1);
        }
    }

    // Flip if black to move (to match tokenizer's board orientation)
    if (flip) {
        from_idx = detail::flip_square_index(from_idx);
        // Only flip to_idx if it's a normal square (0-63)
        if (to_idx < 64) {
            to_idx = detail::flip_square_index(to_idx);
        }
        // Underpromotion indices (64-72) don't need flipping since
        // they encode relative direction, not absolute square
    }

    return {from_idx, to_idx};
}

/**
 * Get the flat policy index from (from_idx, to_idx).
 */
constexpr int policy_flat_index(int from_idx, int to_idx) noexcept {
    return from_idx * POLICY_TO_DIM + to_idx;
}

}  // namespace catgpt

#endif  // CATGPT_ENGINE_POLICY_HPP
