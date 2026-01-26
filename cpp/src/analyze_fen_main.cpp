/**
 * CatGPT FEN Analyzer - Human-readable value & policy output
 *
 * Takes a FEN string as input and displays:
 * - Value: win probability with interpretation
 * - Policy: top moves ranked by probability
 *
 * Usage:
 *   catgpt_analyze [engine_path] < FEN      # Read FEN from stdin
 *   catgpt_analyze [engine_path] "FEN"      # FEN as argument
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "../external/chess-library/include/chess.hpp"
#include "engine/policy.hpp"
#include "engine/trt_evaluator.hpp"
#include "tokenizer.hpp"

namespace fs = std::filesystem;

namespace {

/**
 * Apply softmax to policy logits for the given legal moves only.
 * Returns probabilities normalized over legal moves.
 */
std::vector<std::pair<chess::Move, float>> get_move_probabilities(
    const chess::Board& board,
    const std::array<float, catgpt::POLICY_SIZE>& policy_logits) {

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    bool flip = (board.sideToMove() == chess::Color::BLACK);

    // Collect logits for legal moves
    std::vector<std::pair<chess::Move, float>> move_logits;
    move_logits.reserve(moves.size());

    float max_logit = -std::numeric_limits<float>::infinity();

    for (const auto& move : moves) {
        auto [from_idx, to_idx] = catgpt::encode_move_to_policy_index(move, flip);
        int flat_idx = catgpt::policy_flat_index(from_idx, to_idx);
        float logit = policy_logits[flat_idx];
        move_logits.emplace_back(move, logit);
        max_logit = std::max(max_logit, logit);
    }

    // Compute softmax
    float sum_exp = 0.0f;
    for (auto& [move, logit] : move_logits) {
        logit = std::exp(logit - max_logit);  // Subtract max for numerical stability
        sum_exp += logit;
    }

    // Normalize
    for (auto& [move, prob] : move_logits) {
        prob /= sum_exp;
    }

    // Sort by probability descending
    std::sort(move_logits.begin(), move_logits.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return move_logits;
}

/**
 * Format a move in UCI notation with additional info.
 */
std::string format_move_uci(const chess::Move& move) {
    return chess::uci::moveToUci(move);
}

/**
 * Get a human-readable interpretation of the value.
 */
std::string interpret_value(float value) {
    if (value > 0.9f) return "Winning (very high confidence)";
    if (value > 0.75f) return "Winning";
    if (value > 0.6f) return "Slightly better";
    if (value > 0.55f) return "Small edge";
    if (value >= 0.45f) return "Roughly equal";
    if (value >= 0.4f) return "Small disadvantage";
    if (value >= 0.25f) return "Worse";
    if (value >= 0.1f) return "Losing";
    return "Losing (very high confidence)";
}

/**
 * Print a visual bar representation of probability.
 */
void print_prob_bar(float prob, int width = 20) {
    int filled = static_cast<int>(prob * width + 0.5f);
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        std::cout << (i < filled ? "█" : "░");
    }
    std::cout << "]";
}

/**
 * Print the board in ASCII format.
 */
void print_board(const chess::Board& board) {
    std::cout << "\n  ┌───┬───┬───┬───┬───┬───┬───┬───┐\n";

    for (int rank = 7; rank >= 0; --rank) {
        std::cout << (rank + 1) << " │";
        for (int file = 0; file < 8; ++file) {
            // Square index = rank * 8 + file (a1=0, h8=63)
            auto sq = chess::Square(rank * 8 + file);
            auto piece = board.at(sq);

            char c = '.';
            if (piece != chess::Piece::NONE) {
                static const char* piece_chars = "PNBRQKpnbrqk";
                int idx = static_cast<int>(piece.internal());
                if (idx >= 0 && idx < 12) {
                    c = piece_chars[idx];
                }
            }

            std::cout << " " << c << " │";
        }
        std::cout << "\n";

        if (rank > 0) {
            std::cout << "  ├───┼───┼───┼───┼───┼───┼───┼───┤\n";
        }
    }

    std::cout << "  └───┴───┴───┴───┴───┴───┴───┴───┘\n";
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
}

}  // namespace

/**
 * Check if string looks like a FEN (contains spaces and typical FEN patterns).
 */
bool looks_like_fen(const std::string& s) {
    // FEN strings contain spaces (parts separator) and typically start with ranks
    // containing pieces like r, n, b, q, k, p and "/" separators
    if (s.find(' ') != std::string::npos && s.find('/') != std::string::npos) {
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    fs::path engine_path = "/home/shadeform/CatGPT/sample.trt";
    std::string fen;

    // Check for arguments
    int arg_idx = 1;

    // First non-FEN arg could be engine path
    if (argc > 1) {
        std::string first_arg = argv[1];
        // If it looks like a FEN (has spaces and slashes), treat as FEN not path
        // Otherwise, if it ends with .trt or is a valid file path, treat as engine path
        if (!looks_like_fen(first_arg) &&
            (first_arg.find(".trt") != std::string::npos || fs::exists(first_arg))) {
            engine_path = first_arg;
            arg_idx = 2;
        }
    }

    // Check for FEN argument
    if (argc > arg_idx) {
        fen = argv[arg_idx];
    } else {
        // Read FEN from stdin
        std::cout << "Enter FEN (or press Enter for starting position): ";
        std::getline(std::cin, fen);
        if (fen.empty()) {
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    }

    // Check engine file exists
    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}", engine_path.string());
        std::println(stderr, "Usage: {} [engine_path] [FEN]", argv[0]);
        return 1;
    }

    try {
        // Create board from FEN
        chess::Board board(fen);

        // Load TensorRT engine
        std::println(stderr, "Loading TensorRT engine: {}", engine_path.string());
        auto evaluator = std::make_shared<catgpt::TrtEvaluator>(engine_path);
        std::println(stderr, "Engine loaded successfully");
        std::cerr.flush();  // Flush stderr before stdout output

        // Tokenize position
        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // Run inference
        auto output = evaluator->evaluate(tokens);

        // === Display Results ===

        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "                       CatGPT Position Analysis\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n\n";

        // Board visualization
        std::cout << "Position:\n";
        print_board(board);

        // FEN
        std::cout << "FEN: " << board.getFen() << "\n\n";

        // Side to move
        std::cout << "Side to move: "
                  << (board.sideToMove() == chess::Color::WHITE ? "White" : "Black")
                  << "\n\n";

        // === Value ===
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                            VALUE\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        float win_prob = output.value;
        std::cout << "Win probability (for side to move): ";
        std::cout << std::fixed << std::setprecision(1) << (win_prob * 100.0f) << "%\n";

        std::cout << "\n";
        print_prob_bar(win_prob, 40);
        std::cout << "\n";

        std::cout << "\nInterpretation: " << interpret_value(win_prob) << "\n\n";

        // === Policy ===
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                           POLICY\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        auto move_probs = get_move_probabilities(board, output.policy);

        std::cout << "Top moves (sorted by probability):\n\n";

        int display_count = std::min(static_cast<int>(move_probs.size()), 15);

        std::cout << "  Rank │  Move  │  Prob  │ Visual\n";
        std::cout << "  ─────┼────────┼────────┼────────────────────────\n";

        for (int i = 0; i < display_count; ++i) {
            const auto& [move, prob] = move_probs[i];

            std::cout << "  " << std::setw(4) << (i + 1) << " │ ";
            std::cout << std::setw(6) << format_move_uci(move) << " │ ";
            std::cout << std::setw(5) << std::fixed << std::setprecision(1)
                      << (prob * 100.0f) << "% │ ";
            print_prob_bar(prob, 20);
            std::cout << "\n";
        }

        if (move_probs.size() > 15) {
            std::cout << "\n  ... and " << (move_probs.size() - 15) << " more moves\n";
        }

        std::cout << "\nTotal legal moves: " << move_probs.size() << "\n";

        std::cout << "\n═══════════════════════════════════════════════════════════════\n";

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
