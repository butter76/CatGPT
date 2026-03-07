/**
 * Hash function for chess::Move.
 *
 * Shared by all search node types so that chess::Move can be used as a key
 * in std::unordered_map without duplicate definitions.
 */

#ifndef CATGPT_ENGINE_MOVE_HASH_HPP
#define CATGPT_ENGINE_MOVE_HASH_HPP

#include <cstdint>
#include <functional>

#include "../../external/chess-library/include/chess.hpp"

namespace catgpt {

/**
 * Hash function for chess::Move so it can be used as an unordered_map key.
 */
struct MoveHash {
    std::size_t operator()(const chess::Move& move) const noexcept {
        return std::hash<std::uint16_t>{}(move.move());
    }
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MOVE_HASH_HPP
