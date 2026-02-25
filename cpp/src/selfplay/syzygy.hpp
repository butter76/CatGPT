/**
 * Syzygy Tablebase Probing — thin C++ wrapper around Fathom.
 *
 * Bridges the Disservin chess-library's Board representation to
 * Fathom's C API (tb_init / tb_probe_wdl).  Thread-safe by default
 * (Fathom is compiled without TB_NO_THREADS).
 *
 * Usage:
 *   SyzygyProber prober("/path/to/syzygy/tables");
 *   auto wdl = prober.probe_wdl(board);
 *   if (wdl) { ... }
 */

#ifndef CATGPT_SELFPLAY_SYZYGY_HPP
#define CATGPT_SELFPLAY_SYZYGY_HPP

#include <cstdint>
#include <optional>
#include <print>
#include <string>

#include "../../external/chess-library/include/chess.hpp"

// Fathom C headers
extern "C" {
#include "../../external/fathom/src/tbprobe.h"
}

namespace catgpt {

/**
 * WDL result from Syzygy probe (from side-to-move perspective).
 */
enum class SyzygyWDL {
    LOSS,           // Side to move loses (with best play)
    BLESSED_LOSS,   // Loss but salvageable via 50-move rule
    DRAW,           // Theoretical draw
    CURSED_WIN,     // Win but drawable via 50-move rule
    WIN,            // Side to move wins (with best play)
};

/**
 * RAII wrapper around Fathom's tablebase initialization.
 */
class SyzygyProber {
public:
    /**
     * Initialize Syzygy tablebases from the given path.
     * @param path  Directory containing .rtbw/.rtbz files.
     */
    explicit SyzygyProber(const std::string& path) {
        if (!tb_init(path.c_str())) {
            std::println(stderr, "[Syzygy] WARNING: Failed to initialize tablebases from: {}", path);
            max_pieces_ = 0;
            return;
        }
        max_pieces_ = TB_LARGEST;
        std::println(stderr, "[Syzygy] Initialized: max {} pieces, path: {}", max_pieces_, path);
    }

    ~SyzygyProber() {
        tb_free();
    }

    // Non-copyable (Fathom has global state)
    SyzygyProber(const SyzygyProber&) = delete;
    SyzygyProber& operator=(const SyzygyProber&) = delete;
    SyzygyProber(SyzygyProber&&) = delete;
    SyzygyProber& operator=(SyzygyProber&&) = delete;

    /**
     * Probe the WDL table for the given position.
     *
     * Returns std::nullopt if:
     *   - Too many pieces on the board (> TB_LARGEST)
     *   - Position has castling rights
     *   - Probe failed for any other reason
     *
     * @param board  The position to probe.
     * @return WDL from the side-to-move's perspective.
     */
    [[nodiscard]] std::optional<SyzygyWDL> probe_wdl(const chess::Board& board) const {
        if (max_pieces_ == 0) return std::nullopt;

        // Piece count check
        int piece_count = board.occ().count();
        if (piece_count > static_cast<int>(max_pieces_)) return std::nullopt;

        // Castling rights → not in tablebases
        if (board.castlingRights().has(chess::Color::WHITE) ||
            board.castlingRights().has(chess::Color::BLACK)) {
            return std::nullopt;
        }

        // Extract bitboards
        uint64_t white   = board.us(chess::Color::WHITE).getBits();
        uint64_t black   = board.us(chess::Color::BLACK).getBits();
        uint64_t kings   = board.pieces(chess::PieceType::KING).getBits();
        uint64_t queens  = board.pieces(chess::PieceType::QUEEN).getBits();
        uint64_t rooks   = board.pieces(chess::PieceType::ROOK).getBits();
        uint64_t bishops = board.pieces(chess::PieceType::BISHOP).getBits();
        uint64_t knights = board.pieces(chess::PieceType::KNIGHT).getBits();
        uint64_t pawns   = board.pieces(chess::PieceType::PAWN).getBits();

        // En passant square (0 = none for Fathom; a1 is never a legal ep target)
        auto ep_sq = board.enpassantSq();
        unsigned ep = (ep_sq == chess::Square::NO_SQ) ? 0 : static_cast<unsigned>(ep_sq.index());

        // Side to move
        bool turn = (board.sideToMove() == chess::Color::WHITE);

        // Probe — pass rule50=0 and castling=0 to get theoretical WDL
        unsigned result = tb_probe_wdl(
            white, black, kings, queens, rooks, bishops, knights, pawns,
            /*rule50=*/0, /*castling=*/0, ep, turn);

        if (result == TB_RESULT_FAILED) return std::nullopt;

        switch (result) {
            case TB_LOSS:         return SyzygyWDL::LOSS;
            case TB_BLESSED_LOSS: return SyzygyWDL::BLESSED_LOSS;
            case TB_DRAW:         return SyzygyWDL::DRAW;
            case TB_CURSED_WIN:   return SyzygyWDL::CURSED_WIN;
            case TB_WIN:          return SyzygyWDL::WIN;
            default:              return std::nullopt;
        }
    }

    /** Maximum number of pieces supported by loaded tablebases. */
    [[nodiscard]] unsigned max_pieces() const noexcept { return max_pieces_; }

    /** Whether tablebases were loaded successfully. */
    [[nodiscard]] bool is_available() const noexcept { return max_pieces_ > 0; }

private:
    unsigned max_pieces_ = 0;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_SYZYGY_HPP
