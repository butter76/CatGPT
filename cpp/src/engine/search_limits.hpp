/**
 * SearchLimits - Encapsulates UCI "go" command parameters.
 *
 * This struct captures all the time control and search limit parameters
 * that can be specified in a UCI "go" command.
 */

#ifndef CATGPT_ENGINE_SEARCH_LIMITS_HPP
#define CATGPT_ENGINE_SEARCH_LIMITS_HPP

#include <cstdint>
#include <optional>

namespace catgpt {

struct SearchLimits {
    // Time remaining (milliseconds)
    std::optional<std::int64_t> wtime;
    std::optional<std::int64_t> btime;

    // Increment per move (milliseconds)
    std::optional<std::int64_t> winc;
    std::optional<std::int64_t> binc;

    // Moves until next time control
    std::optional<int> movestogo;

    // Hard limits
    std::optional<int> depth;
    std::optional<std::int64_t> nodes;

    // Fixed time per move (milliseconds)
    std::optional<std::int64_t> movetime;

    // Search until "stop" is received
    bool infinite = false;

    /**
     * Check if any time control is set.
     */
    [[nodiscard]] constexpr bool has_time_control() const noexcept {
        return wtime.has_value() || btime.has_value() || movetime.has_value();
    }

    /**
     * Check if any hard limit is set.
     */
    [[nodiscard]] constexpr bool has_hard_limit() const noexcept {
        return depth.has_value() || nodes.has_value();
    }

    /**
     * Create limits for infinite search.
     */
    [[nodiscard]] static constexpr SearchLimits make_infinite() noexcept {
        SearchLimits limits;
        limits.infinite = true;
        return limits;
    }

    /**
     * Create limits for fixed depth search.
     */
    [[nodiscard]] static constexpr SearchLimits make_depth(int d) noexcept {
        SearchLimits limits;
        limits.depth = d;
        return limits;
    }

    /**
     * Create limits for fixed time search.
     */
    [[nodiscard]] static constexpr SearchLimits make_movetime(std::int64_t ms) noexcept {
        SearchLimits limits;
        limits.movetime = ms;
        return limits;
    }
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_SEARCH_LIMITS_HPP
