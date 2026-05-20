/**
 * Dummy evaluation network for alpha-beta search.
 *
 * Stand-in for a real neural network: material balance plus a small tempo
 * bonus for the side to move. Outputs both centipawns (for search) and a
 * [0, 1] value in the same spirit as the production WDL-derived Q.
 */

#ifndef CATGPT_ALPHA_BETA_DUMMY_EVALUATOR_HPP
#define CATGPT_ALPHA_BETA_DUMMY_EVALUATOR_HPP

#include <algorithm>
#include <cmath>

#include "../../external/chess-library/include/chess.hpp"

namespace catgpt::alpha_beta {

inline constexpr int kMateScore = 30'000;

/// Leaf / heuristic evaluation from the side-to-move perspective.
struct EvalResult {
    float value = 0.5f;  // [0, 1], higher is better for stm
    int centipawns = 0;
};

/**
 * Placeholder network — no GPU, no policy head.
 */
class DummyEvalNetwork {
public:
    [[nodiscard]] EvalResult evaluate(const chess::Board& board) const noexcept {
        const int white_adv = material_balance_cp(board);
        const int stm_material =
            board.sideToMove() == chess::Color::WHITE ? white_adv : -white_adv;
        const int stm_cp = stm_material + 10;  // tempo for side to move

        EvalResult out;
        out.centipawns = stm_cp;
        // Map roughly [-1000, +1000] cp to [0, 1] with tanh squashing.
        out.value = 0.5f + 0.5f * std::tanh(static_cast<float>(stm_cp) / 400.0f);
        out.value = std::clamp(out.value, 0.0f, 1.0f);
        return out;
    }

private:
    [[nodiscard]] static int piece_value(chess::PieceType pt) noexcept {
        if (pt == chess::PieceType::PAWN) return 100;
        if (pt == chess::PieceType::KNIGHT) return 320;
        if (pt == chess::PieceType::BISHOP) return 330;
        if (pt == chess::PieceType::ROOK) return 500;
        if (pt == chess::PieceType::QUEEN) return 900;
        return 0;
    }

    /// White-positive material balance (white pieces minus black).
    [[nodiscard]] static int material_balance_cp(const chess::Board& board) noexcept {
        int score = 0;
        for (auto pt :
             {chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
              chess::PieceType::ROOK, chess::PieceType::QUEEN}) {
            const int v = piece_value(pt);
            score += v * static_cast<int>(board.pieces(pt, chess::Color::WHITE).count());
            score -= v * static_cast<int>(board.pieces(pt, chess::Color::BLACK).count());
        }
        return score;
    }
};

}  // namespace catgpt::alpha_beta

#endif  // CATGPT_ALPHA_BETA_DUMMY_EVALUATOR_HPP
