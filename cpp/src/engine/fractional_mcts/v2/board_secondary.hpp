/**
 * Board → secondary hash bridge for v2::SearchArena.
 *
 * Lives in a separate header from `tt_arena.hpp` so the TT data
 * structure remains chess-library-free (and re-usable from the
 * standalone TT bench/test binaries that don't link chess-library).
 *
 * Anything that wants to talk to `v2::SearchArena::find` /
 * `find_or_claim` / `publish_info` from chess-search code should
 * include this header and pass `v2::secondary_hash(board)` as the
 * `key_secondary` argument.
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_V2_BOARD_SECONDARY_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_V2_BOARD_SECONDARY_HPP

#include <cstdint>

#include "../../../../external/chess-library/include/chess.hpp"

namespace catgpt::v2 {

/**
 * Independent 32-bit hash of a chess::Board, used as the secondary
 * half of the (key, key_secondary) match in v2::SearchArena.
 *
 * Critically, this MUST NOT be a deterministic function of
 * `board.hash()` alone — if it were, an actual primary Zobrist
 * collision would also collide the secondary and we'd gain zero
 * additional collision resistance. We hash the position state
 * directly: piece bitboards (per (PieceType, Color)), side-to-move,
 * castling rights, and en-passant square. The mixing constants are
 * disjoint from `chess::Zobrist::RANDOM_ARRAY`, so two boards that
 * happen to collide on Zobrist are uncorrelated here.
 *
 * Halfmove clock is intentionally NOT mixed in: `chess::Board::hash()`
 * doesn't include it either, and we want the secondary to be
 * invariant across the same equivalence class as the primary.
 * (50-move-rule draws are handled separately at descent time.)
 *
 * Cost: ~12 dependent loads of bitboards (cache-warm, all members of
 * the Board) + a SplitMix-style finalizer; ~5–10 ns warm.
 */
[[nodiscard]] inline uint32_t secondary_hash(const chess::Board& b) noexcept {
    constexpr uint64_t k1 = 0x9E3779B97F4A7C15ULL;
    constexpr uint64_t k2 = 0xBF58476D1CE4E5B9ULL;
    constexpr uint64_t k3 = 0x94D049BB133111EBULL;
    constexpr uint64_t k4 = 0xC2B2AE3D27D4EB4FULL;
    constexpr uint64_t k5 = 0x165667B19E3779F9ULL;
    constexpr uint64_t k6 = 0x85EBCA77C2B2AE63ULL;

    uint64_t h = 0xCBF29CE484222325ULL;  // FNV-ish seed
    h ^= b.pieces(chess::PieceType::PAWN,   chess::Color::WHITE).getBits() * k1;
    h ^= b.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE).getBits() * k2;
    h ^= b.pieces(chess::PieceType::BISHOP, chess::Color::WHITE).getBits() * k3;
    h ^= b.pieces(chess::PieceType::ROOK,   chess::Color::WHITE).getBits() * k4;
    h ^= b.pieces(chess::PieceType::QUEEN,  chess::Color::WHITE).getBits() * k5;
    h ^= b.pieces(chess::PieceType::KING,   chess::Color::WHITE).getBits() * k6;
    h *= k1;
    h ^= b.pieces(chess::PieceType::PAWN,   chess::Color::BLACK).getBits() * k6;
    h ^= b.pieces(chess::PieceType::KNIGHT, chess::Color::BLACK).getBits() * k5;
    h ^= b.pieces(chess::PieceType::BISHOP, chess::Color::BLACK).getBits() * k4;
    h ^= b.pieces(chess::PieceType::ROOK,   chess::Color::BLACK).getBits() * k3;
    h ^= b.pieces(chess::PieceType::QUEEN,  chess::Color::BLACK).getBits() * k2;
    h ^= b.pieces(chess::PieceType::KING,   chess::Color::BLACK).getBits() * k1;

    h ^= (static_cast<uint64_t>(static_cast<int>(b.sideToMove()))     << 1);
    h ^= (static_cast<uint64_t>(b.castlingRights().hashIndex())       << 4);
    h ^= (static_cast<uint64_t>(b.enpassantSq().index())              << 9);

    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<uint32_t>(h);
}

}  // namespace catgpt::v2

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_V2_BOARD_SECONDARY_HPP
