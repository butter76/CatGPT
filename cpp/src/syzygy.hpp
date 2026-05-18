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

#ifndef CATGPT_SYZYGY_HPP
#define CATGPT_SYZYGY_HPP

#include <cstdint>
#include <optional>
#include <print>
#include <string>

#include "../external/chess-library/include/chess.hpp"

// Fathom C headers
extern "C" {
#include "../external/fathom/src/tbprobe.h"
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

inline constexpr std::string_view to_string(SyzygyWDL w) noexcept {
    switch (w) {
        case SyzygyWDL::LOSS:         return "loss";
        case SyzygyWDL::BLESSED_LOSS: return "blessed-loss";
        case SyzygyWDL::DRAW:         return "draw";
        case SyzygyWDL::CURSED_WIN:   return "cursed-win";
        case SyzygyWDL::WIN:          return "win";
    }
    return "?";
}

/**
 * Result of probing the Syzygy DTZ tables at the root.
 *
 * `move`: best move (WDL-preserving, minimum |DTZ|) decoded against the
 *         board's legal movelist so en-passant / promotion flags match
 *         the chess-library's representation exactly.
 * `wdl`:  WDL value at the root (STM perspective).
 * `dtz`:  Distance-To-Zero in plies. Signed: positive on a win, negative
 *         on a loss, zero on a draw. Magnitude is Fathom's raw DTZ for
 *         the chosen move, useful for `info string` telemetry.
 */
struct SyzygyRootResult {
    chess::Move move;
    SyzygyWDL   wdl;
    int         dtz;
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
        ProbeInputs in;
        if (!extract_inputs(board, in)) return std::nullopt;

        // Pass rule50=0 to get theoretical WDL (ignore 50-move drift).
        unsigned result = tb_probe_wdl(
            in.white, in.black, in.kings, in.queens, in.rooks, in.bishops,
            in.knights, in.pawns,
            /*rule50=*/0, /*castling=*/0, in.ep, in.turn);

        return decode_wdl(result);
    }

    /**
     * Probe the Syzygy DTZ tables at the root.
     *
     * Returns the best move (WDL-preserving, minimum |DTZ|) decoded
     * against the board's legal movelist so en-passant / promotion flags
     * match the chess-library's representation exactly.
     *
     * NOTE: `tb_probe_root` is documented as NOT thread-safe; this method
     * is intended to be called once per root, before any concurrent
     * search workers spawn.
     *
     * Returns std::nullopt if the position is not Syzygy-eligible (too
     * many pieces, castling rights, or any other failure mode).
     */
    [[nodiscard]] std::optional<SyzygyRootResult> probe_root_dtz(
        const chess::Board& board) const
    {
        ProbeInputs in;
        if (!extract_inputs(board, in)) return std::nullopt;

        // Pass the real half-move clock so the 50-move rule is honored
        // (matters for cursed-win / blessed-loss handling).
        const unsigned rule50 = static_cast<unsigned>(board.halfMoveClock());

        unsigned r = tb_probe_root(
            in.white, in.black, in.kings, in.queens, in.rooks, in.bishops,
            in.knights, in.pawns,
            rule50, /*castling=*/0, in.ep, in.turn,
            /*results=*/nullptr);

        if (r == TB_RESULT_FAILED) return std::nullopt;
        // Stalemate / checkmate are root states where there are 0 or 1
        // legal moves; the caller's single-legal-move fast path handles
        // those without us. Bail so the caller falls through naturally.
        if (r == TB_RESULT_STALEMATE || r == TB_RESULT_CHECKMATE) {
            return std::nullopt;
        }

        const unsigned wdl_code = TB_GET_WDL(r);
        const unsigned from     = TB_GET_FROM(r);
        const unsigned to       = TB_GET_TO(r);
        const unsigned promo    = TB_GET_PROMOTES(r);
        const unsigned dtz_mag  = TB_GET_DTZ(r);

        chess::Move picked = match_tb_move(board, from, to, promo);
        if (picked == chess::Move::NO_MOVE) return std::nullopt;

        SyzygyWDL wdl;
        switch (wdl_code) {
            case TB_LOSS:         wdl = SyzygyWDL::LOSS;         break;
            case TB_BLESSED_LOSS: wdl = SyzygyWDL::BLESSED_LOSS; break;
            case TB_DRAW:         wdl = SyzygyWDL::DRAW;         break;
            case TB_CURSED_WIN:   wdl = SyzygyWDL::CURSED_WIN;   break;
            case TB_WIN:          wdl = SyzygyWDL::WIN;          break;
            default:              return std::nullopt;
        }

        // Sign DTZ from STM perspective so callers don't have to redo the
        // wdl_code switch: +ve wins, 0 draws, -ve losses.
        int dtz_signed = 0;
        if (wdl == SyzygyWDL::WIN || wdl == SyzygyWDL::CURSED_WIN) {
            dtz_signed = static_cast<int>(dtz_mag);
        } else if (wdl == SyzygyWDL::LOSS || wdl == SyzygyWDL::BLESSED_LOSS) {
            dtz_signed = -static_cast<int>(dtz_mag);
        }

        return SyzygyRootResult{picked, wdl, dtz_signed};
    }

    /** Maximum number of pieces supported by loaded tablebases. */
    [[nodiscard]] unsigned max_pieces() const noexcept { return max_pieces_; }

    /** Whether tablebases were loaded successfully. */
    [[nodiscard]] bool is_available() const noexcept { return max_pieces_ > 0; }

    /**
     * Whether `board` clears the structural gates for a Syzygy probe:
     * piece count within `TB_LARGEST` and no castling rights for either
     * side. Does not actually open the TB files.
     */
    [[nodiscard]] bool is_eligible(const chess::Board& board) const noexcept {
        if (max_pieces_ == 0) return false;
        if (static_cast<unsigned>(board.occ().count()) > max_pieces_) return false;
        if (board.castlingRights().has(chess::Color::WHITE) ||
            board.castlingRights().has(chess::Color::BLACK)) {
            return false;
        }
        return true;
    }

private:
    struct ProbeInputs {
        uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
        unsigned ep;
        bool     turn;
    };

    /**
     * Run the structural gates and extract Fathom-shaped bitboards.
     * Returns false if the position is not Syzygy-eligible.
     */
    [[nodiscard]] bool extract_inputs(const chess::Board& board,
                                      ProbeInputs& out) const noexcept {
        if (!is_eligible(board)) return false;

        out.white   = board.us(chess::Color::WHITE).getBits();
        out.black   = board.us(chess::Color::BLACK).getBits();
        out.kings   = board.pieces(chess::PieceType::KING).getBits();
        out.queens  = board.pieces(chess::PieceType::QUEEN).getBits();
        out.rooks   = board.pieces(chess::PieceType::ROOK).getBits();
        out.bishops = board.pieces(chess::PieceType::BISHOP).getBits();
        out.knights = board.pieces(chess::PieceType::KNIGHT).getBits();
        out.pawns   = board.pieces(chess::PieceType::PAWN).getBits();

        const auto ep_sq = board.enpassantSq();
        out.ep = (ep_sq == chess::Square::NO_SQ)
            ? 0
            : static_cast<unsigned>(ep_sq.index());

        out.turn = (board.sideToMove() == chess::Color::WHITE);
        return true;
    }

    static std::optional<SyzygyWDL> decode_wdl(unsigned result) noexcept {
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

    /**
     * Resolve Fathom's (from, to, promo) triple to a chess::Move by
     * matching against the legal movelist for `board`. This is more
     * robust than reconstructing Move::ENPASSANT / Move::PROMOTION flags
     * manually because the chess library is the source of truth for the
     * move's type field.
     */
    [[nodiscard]] static chess::Move match_tb_move(const chess::Board& board,
                                                   unsigned from,
                                                   unsigned to,
                                                   unsigned promo) noexcept
    {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);

        const chess::PieceType promo_pt = [promo]() {
            switch (promo) {
                case TB_PROMOTES_QUEEN:  return chess::PieceType::QUEEN;
                case TB_PROMOTES_ROOK:   return chess::PieceType::ROOK;
                case TB_PROMOTES_BISHOP: return chess::PieceType::BISHOP;
                case TB_PROMOTES_KNIGHT: return chess::PieceType::KNIGHT;
                default:                 return chess::PieceType::NONE;
            }
        }();

        for (const chess::Move& m : legal) {
            if (m.from().index() != static_cast<int>(from)) continue;
            if (m.to().index()   != static_cast<int>(to))   continue;
            if (m.typeOf() == chess::Move::PROMOTION) {
                if (promo == TB_PROMOTES_NONE) continue;
                if (m.promotionType() != promo_pt) continue;
            } else if (promo != TB_PROMOTES_NONE) {
                continue;
            }
            return m;
        }
        return chess::Move::NO_MOVE;
    }

    unsigned max_pieces_ = 0;
};

}  // namespace catgpt

#endif  // CATGPT_SYZYGY_HPP
