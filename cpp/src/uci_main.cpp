/**
 * CatGPT UCI Engine - Main Entry Point
 *
 * This is the main executable for the UCI chess engine.
 * It creates a UCIHandler with a DummySearch implementation
 * and runs the UCI protocol loop.
 */

#include <iostream>
#include <memory>

#include "engine/dummy_search.hpp"
#include "uci/uci_handler.hpp"

int main() {
    // Disable stdio synchronization for better performance
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Create UCI handler with DummySearch factory
    catgpt::UCIHandler handler([]() {
        return std::make_unique<catgpt::DummySearch>();
    });

    // Run the UCI loop
    handler.run();

    return 0;
}
