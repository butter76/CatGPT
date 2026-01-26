/**
 * CatGPT Policy UCI Engine - Main Entry Point
 *
 * This is the main executable for the policy-based UCI chess engine.
 * It creates a UCIHandler with a PolicySearch implementation backed
 * by TensorRT for neural network inference.
 *
 * The engine uses pure policy selection: it runs the model once to get
 * policy logits and selects the legal move with the highest probability.
 * No lookahead or value evaluation is performed.
 *
 * This is useful for evaluating raw policy quality without search.
 *
 * Usage: catgpt_policy [engine_path]
 *   engine_path: Path to TensorRT engine file (default: ./catgpt.trt)
 */

#include <filesystem>
#include <iostream>
#include <memory>
#include <print>

#include "engine/policy_search.hpp"
#include "engine/trt_evaluator.hpp"
#include "uci/uci_handler.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Disable stdio synchronization for better performance
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Parse command line arguments
    fs::path engine_path = "/home/shadeform/CatGPT/sample.trt";
    if (argc > 1) {
        engine_path = argv[1];
    }

    // Check if engine file exists
    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}", engine_path.string());
        std::println(stderr, "Usage: {} [engine_path]", argv[0]);
        return 1;
    }

    try {
        // Create shared TensorRT evaluator
        std::println(stderr, "Loading TensorRT engine: {}", engine_path.string());
        auto evaluator = std::make_shared<catgpt::TrtEvaluator>(engine_path);
        std::println(stderr, "Engine loaded successfully");

        // Create UCI handler with PolicySearch factory
        catgpt::UCIHandler handler([evaluator]() {
            return std::make_unique<catgpt::PolicySearch>(evaluator);
        });

        // Run the UCI loop
        handler.run();

    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
