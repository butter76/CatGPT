/**
 * GameSlot — per-game state and adjudication for self-play.
 *
 * Each GameSlot tracks one ongoing game: the board, move list,
 * opening FEN, and adjudication counters for draws and resignations.
 * The chess-library handles checkmate, stalemate, repetition, and
 * fifty-move-rule detection natively.
 */

#ifndef CATGPT_SELFPLAY_GAME_SLOT_HPP
#define CATGPT_SELFPLAY_GAME_SLOT_HPP

#include <cmath>
#include <string>
#include <vector>

#include "../../external/chess-library/include/chess.hpp"
#include "selfplay_config.hpp"
#include "syzygy.hpp"

namespace catgpt {

/**
 * How a game ended.
 */
enum class GameTermination {
    ONGOING,                // Game still in progress
    CHECKMATE,
    STALEMATE,
    INSUFFICIENT_MATERIAL,
    THREEFOLD_REPETITION,
    FIFTY_MOVE_RULE,
    DRAW_ADJUDICATED,
    RESIGN_ADJUDICATED,
    SYZYGY_ADJUDICATED,
    MAX_MOVES,
};

/**
 * Game result from white's perspective.
 */
enum class GameOutcome {
    WHITE_WIN,
    BLACK_WIN,
    DRAW,
};

/**
 * Completed game record for PGN output.
 */
struct GameRecord {
    std::string opening_fen;
    std::vector<chess::Move> moves;
    GameTermination termination = GameTermination::ONGOING;
    GameOutcome outcome = GameOutcome::DRAW;
    int total_gpu_evals = 0;

    /** True if the baseline engine played White in this game. */
    bool baseline_white = true;

    /** Convert outcome to PGN result string. */
    [[nodiscard]] std::string result_string() const {
        switch (outcome) {
            case GameOutcome::WHITE_WIN: return "1-0";
            case GameOutcome::BLACK_WIN: return "0-1";
            case GameOutcome::DRAW:      return "1/2-1/2";
        }
        return "*";
    }

    /**
     * Result from the baseline engine's perspective.
     * 1.0 = baseline win, 0.5 = draw, 0.0 = baseline loss.
     */
    [[nodiscard]] float baseline_score() const {
        if (outcome == GameOutcome::DRAW) return 0.5f;
        bool white_won = (outcome == GameOutcome::WHITE_WIN);
        bool baseline_won = (white_won == baseline_white);
        return baseline_won ? 1.0f : 0.0f;
    }
};

/**
 * A single game slot tracking an ongoing game.
 */
class GameSlot {
public:
    GameSlot() = default;

    /**
     * Start a new game from the given opening position.
     */
    void start(const std::string& opening_fen) {
        board_ = chess::Board(opening_fen);
        opening_fen_ = opening_fen;
        moves_.clear();
        total_gpu_evals_ = 0;
        consecutive_draw_scores_ = 0;
        consecutive_resign_scores_ = 0;
        terminated_ = false;
        termination_ = GameTermination::ONGOING;
        outcome_ = GameOutcome::DRAW;
    }

    /** Apply a move and update adjudication state. */
    void apply_move(chess::Move move, int cp_score, int gpu_evals) {
        board_.makeMove<true>(move);
        moves_.push_back(move);
        total_gpu_evals_ += gpu_evals;

        // Update draw adjudication counter
        if (std::abs(cp_score) < draw_cp_threshold_) {
            ++consecutive_draw_scores_;
        } else {
            consecutive_draw_scores_ = 0;
        }

        // Update resign adjudication counter (cp is from side-to-move perspective BEFORE the move)
        // A very negative cp from the side that just moved means they're losing badly
        if (cp_score < -resign_cp_threshold_) {
            ++consecutive_resign_scores_;
        } else {
            consecutive_resign_scores_ = 0;
        }
    }

    /**
     * Check if the game is over (after applying the last move).
     * Sets termination_ and outcome_ if the game has ended.
     *
     * @param config  Selfplay configuration (adjudication thresholds).
     * @param syzygy  Optional Syzygy prober (nullptr to skip tablebase adjudication).
     * @return true if the game is over.
     */
    bool check_game_over(const SelfPlayConfig& config, const SyzygyProber* syzygy = nullptr) {
        // Set adjudication thresholds
        draw_cp_threshold_ = config.draw_cp_threshold;
        resign_cp_threshold_ = config.resign_cp_threshold;

        // Natural endings (from the chess library)
        auto [reason, game_result] = board_.isGameOver();
        if (game_result != chess::GameResult::NONE) {
            if (game_result == chess::GameResult::LOSE) {
                // Side to move lost (got checkmated)
                if (board_.sideToMove() == chess::Color::WHITE) {
                    set_result(GameTermination::CHECKMATE, GameOutcome::BLACK_WIN);
                } else {
                    set_result(GameTermination::CHECKMATE, GameOutcome::WHITE_WIN);
                }
            } else {
                // Draw (stalemate, insufficient material, etc.)
                if (reason == chess::GameResultReason::STALEMATE) {
                    set_result(GameTermination::STALEMATE, GameOutcome::DRAW);
                } else {
                    set_result(GameTermination::INSUFFICIENT_MATERIAL, GameOutcome::DRAW);
                }
            }
            return true;
        }

        // Threefold repetition
        if (board_.isRepetition(2)) {
            set_result(GameTermination::THREEFOLD_REPETITION, GameOutcome::DRAW);
            return true;
        }

        // Fifty-move rule
        if (board_.isHalfMoveDraw()) {
            set_result(GameTermination::FIFTY_MOVE_RULE, GameOutcome::DRAW);
            return true;
        }

        // Syzygy tablebase adjudication
        if (syzygy != nullptr && syzygy->is_available()) {
            auto wdl = syzygy->probe_wdl(board_);
            if (wdl.has_value()) {
                // Adjudicate wins/losses (WDL = WIN or LOSS)
                // Draws are also adjudicated.
                // Cursed wins / blessed losses are treated as draws
                // (the 50-move rule can save the losing side).
                switch (*wdl) {
                    case SyzygyWDL::WIN:
                        if (board_.sideToMove() == chess::Color::WHITE) {
                            set_result(GameTermination::SYZYGY_ADJUDICATED, GameOutcome::WHITE_WIN);
                        } else {
                            set_result(GameTermination::SYZYGY_ADJUDICATED, GameOutcome::BLACK_WIN);
                        }
                        return true;
                    case SyzygyWDL::LOSS:
                        if (board_.sideToMove() == chess::Color::WHITE) {
                            set_result(GameTermination::SYZYGY_ADJUDICATED, GameOutcome::BLACK_WIN);
                        } else {
                            set_result(GameTermination::SYZYGY_ADJUDICATED, GameOutcome::WHITE_WIN);
                        }
                        return true;
                    case SyzygyWDL::DRAW:
                    case SyzygyWDL::CURSED_WIN:
                    case SyzygyWDL::BLESSED_LOSS:
                        set_result(GameTermination::SYZYGY_ADJUDICATED, GameOutcome::DRAW);
                        return true;
                }
            }
        }

        // Max moves
        if (static_cast<int>(moves_.size()) >= config.max_moves) {
            set_result(GameTermination::MAX_MOVES, GameOutcome::DRAW);
            return true;
        }

        // Draw adjudication
        if (consecutive_draw_scores_ >= config.draw_adjudicate_moves) {
            set_result(GameTermination::DRAW_ADJUDICATED, GameOutcome::DRAW);
            return true;
        }

        // Resign adjudication
        if (consecutive_resign_scores_ >= config.resign_adjudicate_moves) {
            // The side that has been consistently losing should lose.
            // Since cp_score was from the side-to-move's perspective when the move was made,
            // negative scores mean that side was losing. The last N moves all had the moving
            // side admitting they're losing, so the current side to move is losing.
            if (board_.sideToMove() == chess::Color::WHITE) {
                set_result(GameTermination::RESIGN_ADJUDICATED, GameOutcome::BLACK_WIN);
            } else {
                set_result(GameTermination::RESIGN_ADJUDICATED, GameOutcome::WHITE_WIN);
            }
            return true;
        }

        return false;
    }

    /** Build a GameRecord from the completed game. */
    [[nodiscard]] GameRecord to_record() const {
        return GameRecord{
            .opening_fen = opening_fen_,
            .moves = moves_,
            .termination = termination_,
            .outcome = outcome_,
            .total_gpu_evals = total_gpu_evals_,
        };
    }

    // ─── Accessors ──────────────────────────────────────────────────────

    [[nodiscard]] const chess::Board& board() const { return board_; }
    [[nodiscard]] chess::Board& board() { return board_; }
    [[nodiscard]] bool is_terminated() const { return terminated_; }
    [[nodiscard]] const std::string& opening_fen() const { return opening_fen_; }
    [[nodiscard]] int total_gpu_evals() const { return total_gpu_evals_; }

private:
    void set_result(GameTermination term, GameOutcome out) {
        terminated_ = true;
        termination_ = term;
        outcome_ = out;
    }

    chess::Board board_;
    std::string opening_fen_;
    std::vector<chess::Move> moves_;
    int total_gpu_evals_ = 0;

    // Adjudication state
    int consecutive_draw_scores_ = 0;
    int consecutive_resign_scores_ = 0;
    int draw_cp_threshold_ = 10;
    int resign_cp_threshold_ = 400;

    // Result
    bool terminated_ = false;
    GameTermination termination_ = GameTermination::ONGOING;
    GameOutcome outcome_ = GameOutcome::DRAW;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_GAME_SLOT_HPP
