/**
 * Tokenization of chess positions for neural network input.
 *
 * This module provides utilities for converting chess::Board positions
 * into a fixed-length sequence of tokens suitable for TensorRT evaluation,
 * equivalent to the Python tokenizer in src/catgpt/core/utils/tokenizer.py.
 *
 * The tokenization scheme represents each square on the board as a single token,
 * with special handling for:
 * - Castling rights: Rooks that can castle are marked with 'C'/'c' instead of 'R'/'r'
 * - En passant: The en passant target square is marked with 'x'
 * - Board orientation: When it's black's turn, the board is flipped and pieces swapped
 * - Halfmove clock: Optionally appended as 2 digits (left-padded with '.')
 *
 * IMPORTANT: For correct tokenization, the Board must have been created using
 * makeMove<true>() (EXACT mode) so that en passant squares only appear when legal.
 */

#ifndef CATGPT_TOKENIZER_HPP
#define CATGPT_TOKENIZER_HPP

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#include "../external/chess-library/include/chess.hpp"

namespace catgpt {

// =============================================================================
// Vocabulary Constants
// =============================================================================

/**
 * Character vocabulary for tokenization.
 * - Digits 0-9 are used for the halfmove clock
 * - Lowercase letters are non-side-to-move pieces, uppercase are side-to-move pieces
 * - 'c'/'C' marks rooks with castling rights
 * - 'x' marks the en passant target square
 * - '.' is used for empty squares and padding
 */
inline constexpr std::array<char, 26> CHARACTERS = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  // 0-9
    'p', 'b', 'n', 'r', 'c', 'k', 'q',  // 10-16: Non-side-to-move pieces (c = castling rook)
    'P', 'B', 'N', 'R', 'C', 'Q', 'K',  // 17-23: Side-to-move pieces (C = castling rook)
    'x',  // 24: En passant target square
    '.',  // 25: Empty square / padding
};

inline constexpr std::size_t VOCAB_SIZE = CHARACTERS.size();  // 26

// Padding/empty token index
inline constexpr std::uint8_t PAD_TOKEN = 25;

// Character to token index lookup (constexpr)
namespace detail {

consteval std::array<std::uint8_t, 128> make_char_to_token() {
    std::array<std::uint8_t, 128> table{};
    // Initialize all to padding token (index 25 for '.')
    for (auto& v : table) v = PAD_TOKEN;

    for (std::size_t i = 0; i < CHARACTERS.size(); ++i) {
        table[static_cast<std::size_t>(CHARACTERS[i])] = static_cast<std::uint8_t>(i);
    }
    return table;
}

inline constexpr auto CHAR_TO_TOKEN = make_char_to_token();

}  // namespace detail

/**
 * Convert a character to its token index.
 */
constexpr std::uint8_t char_to_token(char c) {
    return detail::CHAR_TO_TOKEN[static_cast<std::size_t>(c)];
}

// =============================================================================
// Configuration
// =============================================================================

/**
 * Configuration for board tokenization.
 */
struct TokenizerConfig {
    std::size_t sequence_length = 67;  // Total output length
    bool include_halfmove = true;       // Include 2-digit halfmove clock

    constexpr TokenizerConfig() = default;
    constexpr TokenizerConfig(std::size_t len, bool halfmove)
        : sequence_length(len), include_halfmove(halfmove) {}

    constexpr void validate() const {
        std::size_t min_length = include_halfmove ? 66 : 64;
        if (sequence_length < min_length) {
            throw std::invalid_argument(
                "sequence_length must be at least " + std::to_string(min_length));
        }
    }
};

// Commonly used configurations
inline constexpr TokenizerConfig DEFAULT_CONFIG{67, true};
inline constexpr TokenizerConfig NO_HALFMOVE_CONFIG{64, false};

// =============================================================================
// Square Index Conversion
// =============================================================================

namespace detail {

/**
 * Convert chess library square index (a1=0, h8=63) to tokenizer index (a8=0, h1=63).
 *
 * Chess library: rank * 8 + file where rank 0 = rank 1
 * Tokenizer:     (7 - rank) * 8 + file
 */
constexpr int chess_sq_to_tokenizer_idx(int chess_idx) {
    int file = chess_idx % 8;
    int rank = chess_idx / 8;
    return (7 - rank) * 8 + file;
}

/**
 * Flip a tokenizer index vertically (swap ranks).
 * idx = row * 8 + file, flipped = (7 - row) * 8 + file
 */
constexpr int flip_tokenizer_idx(int idx) {
    int file = idx % 8;
    int row = idx / 8;
    return (7 - row) * 8 + file;
}

// Precomputed lookup table: chess square index -> tokenizer index
consteval std::array<int, 64> make_sq_to_idx_table() {
    std::array<int, 64> table{};
    for (int i = 0; i < 64; ++i) {
        table[i] = chess_sq_to_tokenizer_idx(i);
    }
    return table;
}

inline constexpr auto SQ_TO_IDX = make_sq_to_idx_table();

}  // namespace detail

// =============================================================================
// Piece Character Conversion
// =============================================================================

namespace detail {

/**
 * Convert a chess::Piece to its character representation.
 * Side-to-move pieces are uppercase, opponent pieces are lowercase.
 * Castling rooks are marked separately after this conversion.
 */
constexpr char piece_to_char(chess::Piece piece, bool is_side_to_move_piece) {
    using P = chess::Piece::underlying;

    // Get the piece type character
    char c;
    switch (piece.internal()) {
        case P::WHITEPAWN:   c = 'P'; break;
        case P::WHITEKNIGHT: c = 'N'; break;
        case P::WHITEBISHOP: c = 'B'; break;
        case P::WHITEROOK:   c = 'R'; break;
        case P::WHITEQUEEN:  c = 'Q'; break;
        case P::WHITEKING:   c = 'K'; break;
        case P::BLACKPAWN:   c = 'p'; break;
        case P::BLACKKNIGHT: c = 'n'; break;
        case P::BLACKBISHOP: c = 'b'; break;
        case P::BLACKROOK:   c = 'r'; break;
        case P::BLACKQUEEN:  c = 'q'; break;
        case P::BLACKKING:   c = 'k'; break;
        default:             c = '.'; break;
    }

    // Swap case based on side to move
    // Side-to-move pieces should be uppercase, opponent pieces lowercase
    if (c != '.') {
        bool is_upper = (c >= 'A' && c <= 'Z');
        if (is_side_to_move_piece && !is_upper) {
            c = c - 'a' + 'A';  // to upper
        } else if (!is_side_to_move_piece && is_upper) {
            c = c - 'A' + 'a';  // to lower
        }
    }

    return c;
}

/**
 * Check if a chess::Piece belongs to the given color.
 */
constexpr bool piece_is_color(chess::Piece piece, chess::Color color) {
    using P = chess::Piece::underlying;
    if (piece == P::NONE) return false;

    bool is_white = static_cast<int>(piece.internal()) < 6;
    return (color == chess::Color::WHITE) == is_white;
}

}  // namespace detail

// =============================================================================
// Tokenization Functions
// =============================================================================

/**
 * Tokenize a chess::Board into a fixed-length token sequence.
 * The side-to-move pawns move up the board in the resulting representation.
 *
 * @tparam N Output array size (default: 67 for board + 2 halfmove digits + 1 padding)
 * @param board The chess board to tokenize
 * @param config Tokenization configuration
 * @return Array of token indices
 */
template <std::size_t N = 67>
std::array<std::uint8_t, N> tokenize(const chess::Board& board,
                                      const TokenizerConfig& config = DEFAULT_CONFIG) {
    static_assert(N >= 64, "Output array must be at least 64 elements");

    const auto stm = board.sideToMove();
    const bool is_black_to_move = (stm == chess::Color::BLACK);

    // Initialize output with padding tokens
    std::array<std::uint8_t, N> result;
    result.fill(char_to_token('.'));

    // Build the board representation (64 squares)
    std::array<char, 64> board_chars;
    board_chars.fill('.');

    // Iterate over all squares and populate board_chars in tokenizer order (a8=0, h1=63)
    for (int chess_sq = 0; chess_sq < 64; ++chess_sq) {
        auto sq = chess::Square(chess_sq);
        auto piece = board.at(sq);

        if (piece == chess::Piece::NONE) continue;

        bool is_stm_piece = detail::piece_is_color(piece, stm);
        char c = detail::piece_to_char(piece, is_stm_piece);

        // Convert to tokenizer index
        int tok_idx = detail::SQ_TO_IDX[chess_sq];

        // If black to move, flip the board vertically
        if (is_black_to_move) {
            tok_idx = detail::flip_tokenizer_idx(tok_idx);
        }

        board_chars[tok_idx] = c;
    }

    // Mark castling rooks with 'C' (side-to-move) or 'c' (opponent)
    auto cr = board.castlingRights();

    // Helper to mark a castling rook
    auto mark_castling_rook = [&](chess::Color color, chess::Board::CastlingRights::Side side) {
        if (!cr.has(color, side)) return;

        // Get the rook file and construct the square
        auto rook_file = cr.getRookFile(color, side);
        auto rank = (color == chess::Color::WHITE) ? chess::Rank::RANK_1 : chess::Rank::RANK_8;
        auto sq = chess::Square(rook_file, rank);

        int tok_idx = detail::SQ_TO_IDX[sq.index()];
        if (is_black_to_move) {
            tok_idx = detail::flip_tokenizer_idx(tok_idx);
        }

        // Determine if this rook belongs to side-to-move
        bool is_stm_rook = (color == stm);
        board_chars[tok_idx] = is_stm_rook ? 'C' : 'c';
    };

    mark_castling_rook(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE);
    mark_castling_rook(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    mark_castling_rook(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE);
    mark_castling_rook(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);

    // Mark en passant target square with 'x'
    auto ep_sq = board.enpassantSq();
    if (ep_sq != chess::Square::NO_SQ) {
        int tok_idx = detail::SQ_TO_IDX[ep_sq.index()];
        if (is_black_to_move) {
            tok_idx = detail::flip_tokenizer_idx(tok_idx);
        }
        board_chars[tok_idx] = 'x';
    }

    // Convert board chars to tokens
    for (int i = 0; i < 64; ++i) {
        result[i] = char_to_token(board_chars[i]);
    }

    // Append halfmove clock if configured
    if (config.include_halfmove && N >= 66) {
        auto hfm = board.halfMoveClock();
        if (hfm > 99) {
            // Clamp to 99 (Python would throw, but we'll be lenient)
            hfm = 99;
        }

        // Left-pad with '.' for single digits
        if (hfm < 10) {
            result[64] = char_to_token('.');
            result[65] = char_to_token(static_cast<char>('0' + hfm));
        } else {
            result[64] = char_to_token(static_cast<char>('0' + hfm / 10));
            result[65] = char_to_token(static_cast<char>('0' + hfm % 10));
        }
    }

    return result;
}

/**
 * Tokenize directly from a FEN string.
 *
 * @tparam N Output array size
 * @param fen FEN string (all 6 fields required)
 * @param config Tokenization configuration
 * @return Array of token indices
 */
template <std::size_t N = 67>
std::array<std::uint8_t, N> tokenize_fen(std::string_view fen,
                                          const TokenizerConfig& config = DEFAULT_CONFIG) {
    chess::Board board(fen);
    return tokenize<N>(board, config);
}

/**
 * Tokenize to a dynamically-sized output (for TensorRT input preparation).
 * Returns tokens as int32_t for direct use with TensorRT.
 *
 * @param board The chess board to tokenize
 * @param config Tokenization configuration
 * @return Vector of int32_t token indices
 */
inline std::vector<std::int32_t> tokenize_for_trt(const chess::Board& board,
                                                   const TokenizerConfig& config = NO_HALFMOVE_CONFIG) {
    auto tokens = tokenize<64>(board, config);
    std::vector<std::int32_t> result(tokens.begin(), tokens.end());
    return result;
}

}  // namespace catgpt

#endif  // CATGPT_TOKENIZER_HPP
