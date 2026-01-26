/**
 * SearchResult - Return type for SearchAlgo::search().
 *
 * Contains the best move found along with optional metadata
 * for UCI info output.
 */

#ifndef CATGPT_ENGINE_SEARCH_RESULT_HPP
#define CATGPT_ENGINE_SEARCH_RESULT_HPP

#include <cstdint>
#include <optional>

#include "../../external/chess-library/include/chess.hpp"

namespace catgpt {

/**
 * Score type: either centipawns or mate distance.
 */
struct Score {
    enum class Type { CENTIPAWNS, MATE };

    Type type = Type::CENTIPAWNS;
    int value = 0;  // Centipawns, or moves to mate (positive = winning)

    [[nodiscard]] static constexpr Score cp(int centipawns) noexcept {
        return Score{Type::CENTIPAWNS, centipawns};
    }

    [[nodiscard]] static constexpr Score mate(int moves) noexcept {
        return Score{Type::MATE, moves};
    }

    [[nodiscard]] constexpr bool is_mate() const noexcept {
        return type == Type::MATE;
    }
};

struct SearchResult {
    // The best move found (required)
    chess::Move best_move = chess::Move::NO_MOVE;

    // Evaluation score
    std::optional<Score> score;

    // Search statistics
    std::optional<int> depth;
    std::optional<int> seldepth;  // Selective depth
    std::optional<std::int64_t> nodes;
    std::optional<std::int64_t> time_ms;
    std::optional<std::int64_t> nps;  // Nodes per second

    // Principal variation (best line found)
    std::optional<std::vector<chess::Move>> pv;

    /**
     * Check if a valid move was found.
     */
    [[nodiscard]] bool has_move() const noexcept {
        return best_move != chess::Move::NO_MOVE;
    }

    /**
     * Create a result with just a best move.
     */
    [[nodiscard]] static SearchResult from_move(chess::Move move) noexcept {
        SearchResult result;
        result.best_move = move;
        return result;
    }
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_SEARCH_RESULT_HPP
