/**
 * CatGPT FEN Analyzer - Human-readable value & policy output
 *
 * Takes a FEN string as input and displays:
 * - WDL: Win/Draw/Loss probabilities
 * - BestQ: distribution histogram and expected value
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
 * Print a horizontal histogram of the 81-bin value distribution.
 *
 * Each bin covers a range in [0, 1]. The bar length is proportional to
 * the bin's probability relative to the maximum bin. Uses block characters
 * for sub-character precision (eighths).
 */
void print_value_histogram(const std::array<float, catgpt::VALUE_NUM_BINS>& probs,
                           int max_bar_width = 50) {
    constexpr int N = catgpt::VALUE_NUM_BINS;

    // Find max probability for scaling
    float max_prob = *std::max_element(probs.begin(), probs.end());
    if (max_prob <= 0.0f) max_prob = 1.0f;  // Avoid division by zero

    // Unicode block elements for sub-character precision (eighths)
    static const char* eighths[] = {
        " ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"
    };

    // Compute expected value from distribution (in [0, 1])
    float ev = 0.0f;
    for (int i = 0; i < N; ++i) {
        float center = static_cast<float>(i) / static_cast<float>(N - 1);
        ev += probs[i] * center;
    }

    // Compute mean and variance in [-1, 1] scale (matching MCTS node convention)
    // Bin centers: c_i = -1 + (i + 0.5) * (2 / N)
    constexpr float bin_width = 2.0f / N;
    float mean_11 = 0.0f;
    for (int i = 0; i < N; ++i) {
        float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
        mean_11 += probs[i] * center;
    }
    float variance = 0.0f;
    for (int i = 0; i < N; ++i) {
        float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
        float diff = center - mean_11;
        variance += probs[i] * diff * diff;
    }
    float stddev = std::sqrt(variance);

    // Find the peak bin
    int peak_bin = static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin());
    float peak_center = static_cast<float>(peak_bin) / static_cast<float>(N - 1);

    std::cout << "  E[v] = " << std::fixed << std::setprecision(3) << ev
              << "  |  Peak at bin " << peak_bin
              << " (v ≈ " << std::setprecision(3) << peak_center << ")\n";
    std::cout << "  Variance ([-1,1]) = " << std::setprecision(4) << variance
              << "  |  Std Dev = " << std::setprecision(4) << stddev
              << "  |  Mean ([-1,1]) = " << std::setprecision(4) << mean_11 << "\n\n";

    // Helper to repeat a multi-byte UTF-8 string
    auto repeat_str = [](const std::string& s, int n) {
        std::string result;
        result.reserve(s.size() * n);
        for (int i = 0; i < n; ++i) result += s;
        return result;
    };

    // Print axis label
    std::cout << "  Bin  Value │ Prob   │ Distribution\n";
    std::cout << "  ──── ───── ┼ ────── ┼ " << repeat_str("─", max_bar_width) << "\n";

    for (int i = 0; i < N; ++i) {
        float center = static_cast<float>(i) / static_cast<float>(N - 1);
        float prob = probs[i];
        float fraction = prob / max_prob;
        float bar_exact = fraction * max_bar_width;

        int full_blocks = static_cast<int>(bar_exact);
        int eighth = static_cast<int>((bar_exact - full_blocks) * 8.0f);
        if (eighth > 8) eighth = 8;

        // Color hint: highlight the peak bin
        bool is_peak = (i == peak_bin);

        std::cout << "  " << std::setw(3) << i << "  "
                  << std::fixed << std::setprecision(3) << center << " │ "
                  << std::setw(5) << std::setprecision(1) << (prob * 100.0f) << "% │ ";

        if (is_peak) std::cout << "\033[1;33m";  // Bold yellow for peak

        for (int b = 0; b < full_blocks; ++b) {
            std::cout << "█";
        }
        if (full_blocks < max_bar_width && eighth > 0) {
            std::cout << eighths[eighth];
        }

        if (is_peak) std::cout << "\033[0m";  // Reset

        std::cout << "\n";
    }

    // Bottom axis with scale markers
    std::cout << "  ──── ───── ┼ ────── ┼ " << repeat_str("─", max_bar_width) << "\n";
    std::cout << "                        0%";
    int mid_pos = max_bar_width / 2 - 3;
    int end_pos = max_bar_width - 7;
    if (mid_pos > 3) {
        std::cout << std::string(mid_pos - 2, ' ')
                  << std::fixed << std::setprecision(1) << (max_prob * 50.0f) << "%";
        std::cout << std::string(end_pos - mid_pos - 3, ' ')
                  << std::setprecision(1) << (max_prob * 100.0f) << "%";
    }
    std::cout << "\n";
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

        // === WDL ===
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                           WDL\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        float win  = output.wdl[0];
        float draw = output.wdl[1];
        float loss = output.wdl[2];
        float wdl_value = win + 0.5f * draw;  // Q in [0, 1]

        std::cout << "  Win:  " << std::fixed << std::setprecision(1)
                  << (win * 100.0f) << "%   ";
        print_prob_bar(win, 25);
        std::cout << "\n";

        std::cout << "  Draw: " << std::fixed << std::setprecision(1)
                  << (draw * 100.0f) << "%   ";
        print_prob_bar(draw, 25);
        std::cout << "\n";

        std::cout << "  Loss: " << std::fixed << std::setprecision(1)
                  << (loss * 100.0f) << "%   ";
        print_prob_bar(loss, 25);
        std::cout << "\n\n";

        // Q value from WDL (converted to [-1, 1])
        float q_11 = 2.0f * wdl_value - 1.0f;
        int cp = static_cast<int>(90.0f * std::tan(q_11 * 1.5637541897f));
        std::cout << "  Q (WDL): " << std::showpos << std::setprecision(3) << q_11
                  << std::noshowpos << "  (" << interpret_value(wdl_value) << ", "
                  << std::showpos << cp << std::noshowpos << " cp)\n\n";

        // === BestQ Distribution ===
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                    BESTQ DISTRIBUTION\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        print_value_histogram(output.value_probs);

        std::cout << "\n";

        // === Policy ===
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                           POLICY\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        auto move_probs = get_move_probabilities(board, output.policy);

        std::cout << "All legal moves (sorted by probability):\n\n";

        std::cout << "  Rank │  Move  │  Prob  │ Visual\n";
        std::cout << "  ─────┼────────┼────────┼────────────────────────\n";

        for (int i = 0; i < static_cast<int>(move_probs.size()); ++i) {
            const auto& [move, prob] = move_probs[i];

            std::cout << "  " << std::setw(4) << (i + 1) << " │ ";
            std::cout << std::setw(6) << format_move_uci(move) << " │ ";
            std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                      << (prob * 100.0f) << "% │ ";
            print_prob_bar(prob, 20);
            std::cout << "\n";
        }

        std::cout << "\nTotal legal moves: " << move_probs.size() << "\n";

        std::cout << "\n═══════════════════════════════════════════════════════════════\n";

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
