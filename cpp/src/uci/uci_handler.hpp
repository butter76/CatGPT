/**
 * UCIHandler - UCI protocol implementation.
 *
 * This class handles parsing UCI commands from stdin, managing the search
 * thread, and outputting UCI responses to stdout.
 *
 * Threading model:
 * - The main loop runs on the main thread, reading from stdin
 * - Search runs on a dedicated std::jthread
 * - stop() can be called from the main thread to interrupt search
 */

#ifndef CATGPT_UCI_HANDLER_HPP
#define CATGPT_UCI_HANDLER_HPP

#include <functional>
#include <iostream>
#include <memory>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../../external/chess-library/include/chess.hpp"
#include "../engine/search_algo.hpp"

namespace catgpt {

// Engine identification
inline constexpr std::string_view ENGINE_NAME = "CatGPT";
inline constexpr std::string_view ENGINE_AUTHOR = "CatGPT Team";

/**
 * Factory function type for creating search algorithms.
 */
using SearchAlgoFactory = std::function<std::unique_ptr<SearchAlgo>()>;

class UCIHandler {
public:
    /**
     * Construct a UCI handler with a search algorithm factory.
     *
     * @param factory Function that creates a new SearchAlgo instance.
     */
    explicit UCIHandler(SearchAlgoFactory factory)
        : factory_(std::move(factory))
        , search_algo_(factory_())
    {}

    /**
     * Run the UCI main loop.
     * Reads commands from stdin until "quit" is received.
     */
    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!process_command(line)) {
                break;  // quit was received
            }
        }

        // Ensure search is stopped before exiting
        stop_search();
    }

private:
    SearchAlgoFactory factory_;
    std::unique_ptr<SearchAlgo> search_algo_;
    std::jthread search_thread_;

    /**
     * Process a single UCI command.
     * @return false if the engine should quit, true otherwise.
     */
    bool process_command(std::string_view line) {
        // Trim leading/trailing whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string_view::npos) {
            return true;  // Empty line
        }
        line = line.substr(start);
        auto end = line.find_last_not_of(" \t\r\n");
        if (end != std::string_view::npos) {
            line = line.substr(0, end + 1);
        }

        // Parse the command
        auto tokens = tokenize(line);
        if (tokens.empty()) {
            return true;
        }

        const auto& cmd = tokens[0];

        if (cmd == "uci") {
            handle_uci();
        } else if (cmd == "isready") {
            handle_isready();
        } else if (cmd == "ucinewgame") {
            handle_ucinewgame();
        } else if (cmd == "position") {
            handle_position(tokens);
        } else if (cmd == "go") {
            handle_go(tokens);
        } else if (cmd == "stop") {
            handle_stop();
        } else if (cmd == "quit") {
            return false;
        } else if (cmd == "d") {
            // Debug command: print current board
            handle_debug();
        }
        // Unknown commands are silently ignored per UCI spec

        return true;
    }

    void handle_uci() {
        std::println("id name {}", ENGINE_NAME);
        std::println("id author {}", ENGINE_AUTHOR);
        // TODO: Add UCI options here when needed
        std::println("uciok");
    }

    void handle_isready() {
        // Ensure any pending operations complete
        stop_search();
        std::println("readyok");
    }

    void handle_ucinewgame() {
        stop_search();
        // Recreate the search algorithm for a fresh game
        search_algo_ = factory_();
    }

    void handle_position(const std::vector<std::string>& tokens) {
        stop_search();

        if (tokens.size() < 2) {
            return;  // Invalid command
        }

        std::size_t moves_idx = 0;
        std::string fen;

        if (tokens[1] == "startpos") {
            fen = std::string(STARTPOS_FEN);
            moves_idx = 2;
        } else if (tokens[1] == "fen") {
            // FEN has 6 parts: position, side, castling, ep, halfmove, fullmove
            std::ostringstream fen_stream;
            std::size_t i = 2;
            for (; i < tokens.size() && tokens[i] != "moves"; ++i) {
                if (i > 2) fen_stream << ' ';
                fen_stream << tokens[i];
            }
            fen = fen_stream.str();
            moves_idx = i;
        } else {
            return;  // Invalid command
        }

        // Reset to the base position
        search_algo_->reset(fen);

        // Apply moves if present
        if (moves_idx < tokens.size() && tokens[moves_idx] == "moves") {
            for (std::size_t i = moves_idx + 1; i < tokens.size(); ++i) {
                auto move = chess::uci::uciToMove(search_algo_->board(), tokens[i]);
                if (move != chess::Move::NO_MOVE) {
                    search_algo_->makemove(move);
                }
            }
        }
    }

    void handle_go(const std::vector<std::string>& tokens) {
        stop_search();

        SearchLimits limits = parse_go_params(tokens);

        // Start search on a new thread
        search_thread_ = std::jthread([this, limits]() {
            auto result = search_algo_->search(limits);
            output_bestmove(result);
        });
    }

    void handle_stop() {
        stop_search();
    }

    void handle_debug() {
        std::cout << search_algo_->board() << '\n';
        std::println("FEN: {}", search_algo_->board().getFen());
    }

    void stop_search() {
        if (search_thread_.joinable()) {
            search_algo_->stop();
            search_thread_.join();
        }
    }

    SearchLimits parse_go_params(const std::vector<std::string>& tokens) {
        SearchLimits limits;

        for (std::size_t i = 1; i < tokens.size(); ++i) {
            const auto& token = tokens[i];

            if (token == "infinite") {
                limits.infinite = true;
            } else if (token == "wtime" && i + 1 < tokens.size()) {
                limits.wtime = std::stoll(tokens[++i]);
            } else if (token == "btime" && i + 1 < tokens.size()) {
                limits.btime = std::stoll(tokens[++i]);
            } else if (token == "winc" && i + 1 < tokens.size()) {
                limits.winc = std::stoll(tokens[++i]);
            } else if (token == "binc" && i + 1 < tokens.size()) {
                limits.binc = std::stoll(tokens[++i]);
            } else if (token == "movestogo" && i + 1 < tokens.size()) {
                limits.movestogo = std::stoi(tokens[++i]);
            } else if (token == "depth" && i + 1 < tokens.size()) {
                limits.depth = std::stoi(tokens[++i]);
            } else if (token == "nodes" && i + 1 < tokens.size()) {
                limits.nodes = std::stoll(tokens[++i]);
            } else if (token == "movetime" && i + 1 < tokens.size()) {
                limits.movetime = std::stoll(tokens[++i]);
            }
        }

        return limits;
    }

    void output_bestmove(const SearchResult& result) {
        if (!result.has_move()) {
            // No legal move (shouldn't happen in normal play)
            std::println("bestmove 0000");
            return;
        }

        std::string bestmove_str = chess::uci::moveToUci(result.best_move);
        std::println("bestmove {}", bestmove_str);
    }

    static std::vector<std::string> tokenize(std::string_view line) {
        std::vector<std::string> tokens;
        std::string token;

        for (char c : line) {
            if (c == ' ' || c == '\t') {
                if (!token.empty()) {
                    tokens.push_back(std::move(token));
                    token.clear();
                }
            } else {
                token += c;
            }
        }

        if (!token.empty()) {
            tokens.push_back(std::move(token));
        }

        return tokens;
    }
};

}  // namespace catgpt

#endif  // CATGPT_UCI_HANDLER_HPP
