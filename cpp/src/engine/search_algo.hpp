/**
 * SearchAlgo - Abstract interface for chess search algorithms.
 *
 * This class defines the contract that all search implementations must follow.
 * The UCI handler owns a SearchAlgo and delegates position management and
 * search execution to it.
 *
 * Threading model:
 * - reset() and makemove() are called from the UCI/main thread
 * - search() runs on a dedicated search thread
 * - stop() is called from the UCI thread to signal early termination
 */

#ifndef CATGPT_ENGINE_SEARCH_ALGO_HPP
#define CATGPT_ENGINE_SEARCH_ALGO_HPP

#include <string_view>

#include "../../external/chess-library/include/chess.hpp"
#include "search_limits.hpp"
#include "search_result.hpp"

namespace catgpt {

// Standard starting position FEN
inline constexpr std::string_view STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/**
 * Abstract base class for search algorithms.
 */
class SearchAlgo {
public:
    virtual ~SearchAlgo() = default;

    // Non-copyable, non-movable (due to internal threading state)
    SearchAlgo(const SearchAlgo&) = delete;
    SearchAlgo& operator=(const SearchAlgo&) = delete;
    SearchAlgo(SearchAlgo&&) = delete;
    SearchAlgo& operator=(SearchAlgo&&) = delete;

    /**
     * Reset the internal position to the starting position or a given FEN.
     *
     * Called from the UCI thread when a "position" command is received.
     * Must not be called while a search is in progress.
     *
     * @param fen The FEN string to set, or STARTPOS_FEN if omitted.
     */
    virtual void reset(std::string_view fen = STARTPOS_FEN) = 0;

    /**
     * Apply a move to the current position.
     *
     * Called from the UCI thread to replay moves after "position ... moves".
     * Must not be called while a search is in progress.
     *
     * @param move The move to apply (in internal representation).
     */
    virtual void makemove(const chess::Move& move) = 0;

    /**
     * Search for the best move from the current position.
     *
     * This method blocks until the search completes or stop() is called.
     * It should be invoked on a dedicated search thread.
     *
     * @param limits The search limits (time, depth, nodes, etc.).
     * @return The search result containing the best move and optional info.
     */
    virtual SearchResult search(const SearchLimits& limits) = 0;

    /**
     * Signal the search to stop as soon as possible.
     *
     * Called from the UCI thread when a "stop" command is received.
     * This is thread-safe and may be called while search() is running.
     * The search should exit gracefully and return its best move so far.
     */
    virtual void stop() = 0;

    /**
     * Get the current board position (for debugging/info).
     */
    [[nodiscard]] virtual const chess::Board& board() const = 0;

protected:
    // Protected constructor - only derived classes can instantiate
    SearchAlgo() = default;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_SEARCH_ALGO_HPP
