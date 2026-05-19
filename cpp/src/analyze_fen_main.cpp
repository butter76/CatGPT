/**
 * CatGPT FEN Analyzer — Human-readable NN output for a single position.
 *
 * Successor to the old `catgpt_analyze` binary that was deleted in the
 * libfork PR (commit cb383ad) along with the `TrtEvaluator` class it
 * depended on. This version is fully self-contained — no `BatchEvaluator`,
 * no libcoro, no libfork. Supports two input formats:
 *
 *   - Packed `.network` file (CATGPT_NETWORK magic): a bundle of
 *     per-bucket sub-engines. We pick the smallest bucket sub-engine
 *     (typically 1) and instantiate it directly.
 *   - Raw serialized TensorRT engine (typically `.trt`, "ftrt" magic):
 *     one engine with N optimization profiles. We pick the profile whose
 *     opt batch is smallest and bind that.
 *
 * Runs ONE synchronous inference, and pretty-prints:
 *
 *   - The board
 *   - Scalar Q (value head) with cp conversion
 *   - 81-bin BestQ distribution as a horizontal histogram
 *   - Approximate W/D/L derived from the bin distribution
 *   - All legal moves ranked by policy probability, with the optimistic
 *     policy alongside if the engine exposes an `optimistic_policy`
 *     output tensor.
 *
 * Usage:
 *   catgpt_analyze [network_path] [FEN]
 *   catgpt_analyze [network_path] < fen.txt
 *
 * Defaults: engine_path = $CATGPT_NETWORK or /home/shadeform/CatGPT/main.trt
 */

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../external/chess-library/include/chess.hpp"
#include "engine/network_file.hpp"
#include "engine/nn_constants.hpp"
#include "engine/policy.hpp"
#include "engine/trt_runtime.hpp"
#include "tokenizer.hpp"

namespace fs = std::filesystem;

namespace {

// ─── Single-position TRT evaluator ──────────────────────────────────────

constexpr int SEQ_LENGTH = 64;

/**
 * Outputs of one NN evaluation.
 *
 * `has_optimistic_policy` is true iff the loaded engine actually exposed
 * an `optimistic_policy` output tensor; older engines simply omit it.
 */
struct AnalysisOutput {
    float                                          value;           // Q in [0, 1]
    std::array<float, catgpt::VALUE_NUM_BINS>      value_probs;     // 81-bin BestQ
    std::array<float, catgpt::POLICY_SIZE>         policy;          // logits
    std::array<float, catgpt::POLICY_SIZE>         optimistic_policy;
    bool                                           has_optimistic_policy = false;
};

class SinglePositionEvaluator {
public:
    explicit SinglePositionEvaluator(const fs::path& engine_path) {
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");

        // Sniff magic to decide format. CATGPT_NETWORK bundles start with
        // the 16-byte ASCII magic; raw TRT engines start with the 4-byte
        // ASCII "ftrt" tag.
        std::array<char, 16> magic{};
        {
            std::ifstream f(engine_path, std::ios::binary);
            if (!f) throw std::runtime_error(
                std::format("cannot open {}", engine_path.string()));
            f.read(magic.data(), magic.size());
        }

        if (std::memcmp(magic.data(),
                        catgpt::NetworkFile::kMagic,
                        catgpt::NetworkFile::kMagicSize) == 0) {
            load_from_network_bundle(engine_path);
        } else {
            load_from_raw_trt_engine(engine_path);
        }

        discover_io();
        allocate_buffers();
        setup_context();
    }

    ~SinglePositionEvaluator() {
        // Tear down before freeing buffers it references.
        context_.reset();
        engine_.reset();
        runtime_.reset();

        if (stream_) cudaStreamDestroy(stream_);
        if (d_input_)             cudaFree(d_input_);
        if (d_value_)             cudaFree(d_value_);
        if (d_value_probs_)       cudaFree(d_value_probs_);
        if (d_policy_)            cudaFree(d_policy_);
        if (d_optimistic_policy_) cudaFree(d_optimistic_policy_);
        if (h_input_)             cudaFreeHost(h_input_);
        if (h_value_)             cudaFreeHost(h_value_);
        if (h_value_probs_)       cudaFreeHost(h_value_probs_);
        if (h_policy_)            cudaFreeHost(h_policy_);
        if (h_optimistic_policy_) cudaFreeHost(h_optimistic_policy_);
    }

    SinglePositionEvaluator(const SinglePositionEvaluator&) = delete;
    SinglePositionEvaluator& operator=(const SinglePositionEvaluator&) = delete;

    bool has_optimistic_policy() const noexcept {
        return !optimistic_policy_output_name_.empty();
    }

    AnalysisOutput evaluate(const std::array<std::uint8_t, SEQ_LENGTH>& tokens) {
        // Pack into the first slot of the pinned input buffer; trailing
        // slots stay zero (no other batch entries are read by the model).
        for (int i = 0; i < SEQ_LENGTH; ++i) {
            h_input_[i] = static_cast<std::int32_t>(tokens[i]);
        }

        CATGPT_CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_,
                                          bucket_ * SEQ_LENGTH * sizeof(std::int32_t),
                                          cudaMemcpyHostToDevice, stream_));
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_value_, d_value_,
                                          bucket_ * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream_));
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_value_probs_, d_value_probs_,
                                          bucket_ * catgpt::VALUE_NUM_BINS * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream_));
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_policy_, d_policy_,
                                          bucket_ * catgpt::POLICY_SIZE * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream_));
        if (has_optimistic_policy()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(
                h_optimistic_policy_, d_optimistic_policy_,
                bucket_ * catgpt::POLICY_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost, stream_));
        }
        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        AnalysisOutput out;
        out.value = h_value_[0];
        std::memcpy(out.value_probs.data(), h_value_probs_,
                    catgpt::VALUE_NUM_BINS * sizeof(float));
        std::memcpy(out.policy.data(), h_policy_,
                    catgpt::POLICY_SIZE * sizeof(float));
        if (has_optimistic_policy()) {
            std::memcpy(out.optimistic_policy.data(), h_optimistic_policy_,
                        catgpt::POLICY_SIZE * sizeof(float));
            out.has_optimistic_policy = true;
        }
        return out;
    }

private:
    void load_from_network_bundle(const fs::path& path) {
        catgpt::NetworkFile file(path);

        const catgpt::NetworkFile::SubEngine* chosen = nullptr;
        for (const auto& sub : file.sub_engines()) {
            if (!chosen || sub.bucket_size < chosen->bucket_size) chosen = &sub;
        }
        if (!chosen) throw std::runtime_error("network file has no sub-engines");

        bucket_      = chosen->bucket_size;
        profile_idx_ = 0;  // sub-engines have exactly one profile (min==opt==max==bucket)

        std::println(stderr,
                     "[analyze_fen] Loaded .network bundle {} ({:.1f} MB, "
                     "{} sub-engine(s); using bucket={})",
                     path.string(),
                     static_cast<double>(file.file_size()) / (1024.0 * 1024.0),
                     file.sub_engines().size(), bucket_);

        engine_.reset(runtime_->deserializeCudaEngine(
            chosen->blob.data(), chosen->blob.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
    }

    void load_from_raw_trt_engine(const fs::path& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error(
            std::format("cannot open {}", path.string()));
        const auto size = static_cast<std::size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        std::vector<char> buf(size);
        if (!f.read(buf.data(), static_cast<std::streamsize>(size))) {
            throw std::runtime_error("short read");
        }

        std::println(stderr,
                     "[analyze_fen] Loaded .trt engine {} ({:.1f} MB)",
                     path.string(),
                     static_cast<double>(size) / (1024.0 * 1024.0));

        engine_.reset(runtime_->deserializeCudaEngine(buf.data(), buf.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

        // Find the input tensor name first (needed to query profile shapes).
        std::string input_name;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_name = name;
                break;
            }
        }
        if (input_name.empty()) throw std::runtime_error("no input tensor in engine");

        // Pick the optimization profile with the smallest opt batch.
        const int n_profiles = engine_->getNbOptimizationProfiles();
        if (n_profiles < 1) throw std::runtime_error("engine has no profiles");
        int best_profile = 0;
        int best_batch   = std::numeric_limits<int>::max();
        for (int p = 0; p < n_profiles; ++p) {
            auto opt = engine_->getProfileShape(
                input_name.c_str(), p, nvinfer1::OptProfileSelector::kOPT);
            if (opt.nbDims < 1) continue;
            const int b = static_cast<int>(opt.d[0]);
            if (b > 0 && b < best_batch) {
                best_batch   = b;
                best_profile = p;
            }
        }
        if (best_batch == std::numeric_limits<int>::max()) {
            throw std::runtime_error("could not determine batch size from profiles");
        }
        profile_idx_ = best_profile;
        bucket_      = best_batch;

        std::println(stderr,
                     "[analyze_fen] {} optimization profile(s); using profile {} (batch={})",
                     n_profiles, profile_idx_, bucket_);
    }

    void discover_io() {
        const int num_io = engine_->getNbIOTensors();
        for (int i = 0; i < num_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            auto dims = engine_->getTensorShape(name);
            std::string name_str(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
                continue;
            }

            // Specific-name matches first; "optimistic_policy" must come
            // before "policy" because the former is a superstring.
            if (name_str.find("optimistic_policy") != std::string::npos) {
                optimistic_policy_output_name_ = name;
            } else if (name_str.find("policy") != std::string::npos ||
                       name_str.find("Policy") != std::string::npos) {
                policy_output_name_ = name;
            } else if (name_str.find("value_probs") != std::string::npos ||
                       name_str.find("bestq_probs") != std::string::npos) {
                value_probs_output_name_ = name;
            } else if (name_str.find("value") != std::string::npos ||
                       name_str.find("Value") != std::string::npos) {
                if (value_output_name_.empty()) value_output_name_ = name;
            } else {
                // Shape-based fallback (matches BatchEvaluator's logic).
                if (dims.nbDims == 1 || (dims.nbDims == 2 && dims.d[1] == 1)) {
                    if (value_output_name_.empty()) value_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == catgpt::VALUE_NUM_BINS) {
                    if (value_probs_output_name_.empty()) value_probs_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == catgpt::POLICY_SIZE) {
                    if (policy_output_name_.empty()) policy_output_name_ = name_str;
                }
            }
        }

        if (input_name_.empty())          throw std::runtime_error("no input tensor");
        if (value_output_name_.empty())   throw std::runtime_error("no value tensor");
        if (policy_output_name_.empty())  throw std::runtime_error("no policy tensor");

        std::println(stderr,
                     "[analyze_fen] IO: input='{}', value='{}', value_probs='{}', "
                     "policy='{}', optimistic_policy='{}'",
                     input_name_, value_output_name_,
                     value_probs_output_name_.empty() ? "<none>" : value_probs_output_name_,
                     policy_output_name_,
                     optimistic_policy_output_name_.empty() ? "<none>" : optimistic_policy_output_name_);
    }

    void allocate_buffers() {
        CATGPT_CUDA_CHECK(cudaStreamCreate(&stream_));

        CATGPT_CUDA_CHECK(cudaMalloc(&d_input_,       bucket_ * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_,       bucket_ * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_probs_, bucket_ * catgpt::VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_policy_,      bucket_ * catgpt::POLICY_SIZE * sizeof(float)));
        if (has_optimistic_policy()) {
            CATGPT_CUDA_CHECK(cudaMalloc(&d_optimistic_policy_,
                                          bucket_ * catgpt::POLICY_SIZE * sizeof(float)));
        }

        CATGPT_CUDA_CHECK(cudaMallocHost(&h_input_,       bucket_ * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_,       bucket_ * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_probs_, bucket_ * catgpt::VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_policy_,      bucket_ * catgpt::POLICY_SIZE * sizeof(float)));
        if (has_optimistic_policy()) {
            CATGPT_CUDA_CHECK(cudaMallocHost(&h_optimistic_policy_,
                                              bucket_ * catgpt::POLICY_SIZE * sizeof(float)));
        }

        // Zero the input slack so any trailing batch slots are deterministic
        // (the model only reads slot 0 here, but be tidy).
        std::memset(h_input_, 0, bucket_ * SEQ_LENGTH * sizeof(std::int32_t));
    }

    void setup_context() {
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");

        // For .network sub-engines profile_idx_ is always 0 (single profile,
        // min==opt==max==bucket). For raw .trt engines we picked the
        // smallest-batch profile in load_from_raw_trt_engine().
        if (!context_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            throw std::runtime_error("setOptimizationProfileAsync failed");
        }
        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        nvinfer1::Dims in_dims;
        in_dims.nbDims = 2;
        in_dims.d[0] = bucket_;
        in_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), in_dims);

        context_->setTensorAddress(input_name_.c_str(),         d_input_);
        context_->setTensorAddress(value_output_name_.c_str(),  d_value_);
        if (!value_probs_output_name_.empty()) {
            context_->setTensorAddress(value_probs_output_name_.c_str(), d_value_probs_);
        }
        context_->setTensorAddress(policy_output_name_.c_str(), d_policy_);
        if (has_optimistic_policy()) {
            context_->setTensorAddress(optimistic_policy_output_name_.c_str(),
                                       d_optimistic_policy_);
        }
    }

    catgpt::TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t   stream_              = nullptr;
    std::int32_t*  d_input_             = nullptr;
    float*         d_value_             = nullptr;
    float*         d_value_probs_       = nullptr;
    float*         d_policy_            = nullptr;
    float*         d_optimistic_policy_ = nullptr;
    std::int32_t*  h_input_             = nullptr;
    float*         h_value_             = nullptr;
    float*         h_value_probs_       = nullptr;
    float*         h_policy_            = nullptr;
    float*         h_optimistic_policy_ = nullptr;

    int           bucket_       = 1;
    int           profile_idx_  = 0;
    std::string   input_name_;
    std::string   value_output_name_;
    std::string   value_probs_output_name_;
    std::string   policy_output_name_;
    std::string   optimistic_policy_output_name_;
};

// ─── Pretty-printing helpers ────────────────────────────────────────────

std::vector<std::pair<chess::Move, float>> softmax_legal_moves(
    const chess::Board& board,
    const std::array<float, catgpt::POLICY_SIZE>& logits)
{
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    const bool flip = (board.sideToMove() == chess::Color::BLACK);

    std::vector<std::pair<chess::Move, float>> out;
    out.reserve(moves.size());
    float max_logit = -std::numeric_limits<float>::infinity();
    for (const auto& m : moves) {
        auto [from_idx, to_idx] = catgpt::encode_move_to_policy_index(m, flip);
        float logit = logits[catgpt::policy_flat_index(from_idx, to_idx)];
        out.emplace_back(m, logit);
        max_logit = std::max(max_logit, logit);
    }
    float sum_exp = 0.0f;
    for (auto& [m, v] : out) {
        v = std::exp(v - max_logit);
        sum_exp += v;
    }
    if (sum_exp > 0.0f) {
        for (auto& [m, v] : out) v /= sum_exp;
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    return out;
}

std::string interpret_value(float v_unit) {
    // v_unit in [0, 1] (Q from side-to-move's perspective)
    if (v_unit > 0.9f)  return "Winning (very high confidence)";
    if (v_unit > 0.75f) return "Winning";
    if (v_unit > 0.6f)  return "Clearly better";
    if (v_unit > 0.55f) return "Small edge";
    if (v_unit >= 0.45f) return "Roughly equal";
    if (v_unit >= 0.4f)  return "Small disadvantage";
    if (v_unit >= 0.25f) return "Worse";
    if (v_unit >= 0.1f)  return "Losing";
    return "Losing (very high confidence)";
}

void print_prob_bar(float prob, int width = 25) {
    int filled = static_cast<int>(prob * width + 0.5f);
    if (filled < 0) filled = 0;
    if (filled > width) filled = width;
    std::cout << "[";
    for (int i = 0; i < width; ++i) std::cout << (i < filled ? "█" : "░");
    std::cout << "]";
}

/**
 * 81-bin histogram of the BestQ distribution.
 *
 * Bin i covers Q value in [-1, 1] split into VALUE_NUM_BINS bins, so
 * bin centers are c_i = -1 + (i + 0.5) * (2 / N). We also display the
 * implied [0, 1] center for parity with the scalar `value` head.
 */
void print_value_histogram(const std::array<float, catgpt::VALUE_NUM_BINS>& probs,
                           int max_bar_width = 50)
{
    constexpr int N = catgpt::VALUE_NUM_BINS;
    constexpr float bin_width = 2.0f / static_cast<float>(N);

    float max_prob = *std::max_element(probs.begin(), probs.end());
    if (max_prob <= 0.0f) max_prob = 1.0f;

    static const char* eighths[] = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};

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

    int peak_bin = static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin());
    float peak_center_11 = -1.0f + (static_cast<float>(peak_bin) + 0.5f) * bin_width;

    std::cout << "  E[Q] = " << std::fixed << std::setprecision(3) << mean_11
              << "  |  Std = " << std::setprecision(3) << stddev
              << "  |  Peak bin " << peak_bin
              << " (Q ≈ " << std::showpos << std::setprecision(3) << peak_center_11
              << std::noshowpos << ")\n\n";

    auto repeat_str = [](const std::string& s, int n) {
        std::string r; r.reserve(s.size() * n);
        for (int i = 0; i < n; ++i) r += s;
        return r;
    };

    std::cout << "  Bin    Q    │  Prob   │ Distribution\n";
    std::cout << "  ──── ────── ┼ ─────── ┼ " << repeat_str("─", max_bar_width) << "\n";

    for (int i = 0; i < N; ++i) {
        float center_11 = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
        float prob = probs[i];
        float fraction = prob / max_prob;
        float bar_exact = fraction * static_cast<float>(max_bar_width);
        int full_blocks = static_cast<int>(bar_exact);
        int eighth = static_cast<int>((bar_exact - static_cast<float>(full_blocks)) * 8.0f);
        if (eighth > 8) eighth = 8;
        if (full_blocks > max_bar_width) full_blocks = max_bar_width;

        bool is_peak = (i == peak_bin);

        std::cout << "  " << std::setw(3) << i << "  "
                  << std::showpos << std::fixed << std::setprecision(3) << center_11
                  << std::noshowpos << " │ "
                  << std::setw(5) << std::setprecision(1) << (prob * 100.0f) << "%  │ ";

        if (is_peak) std::cout << "\033[1;33m";
        for (int b = 0; b < full_blocks; ++b) std::cout << "█";
        if (full_blocks < max_bar_width && eighth > 0) std::cout << eighths[eighth];
        if (is_peak) std::cout << "\033[0m";
        std::cout << "\n";
    }
    std::cout << "  ──── ────── ┴ ─────── ┴ " << repeat_str("─", max_bar_width) << "\n";
}

/**
 * Approximate WDL from the 81-bin value distribution.
 *
 * The model no longer emits a 3-tuple WDL head — only a scalar value
 * (= P(W) + 0.5 * P(D)) and the full 81-bin distribution. We synthesize
 * a WDL view by partitioning the bins:
 *   Loss = mass with Q < -0.5
 *   Draw = mass with |Q| <= 0.5
 *   Win  = mass with Q >  0.5
 *
 * Q = +1 is decisive win, Q = -1 is decisive loss, Q = 0 is a draw.
 */
struct ApproxWDL { float w, d, l; };
ApproxWDL approx_wdl_from_distribution(
    const std::array<float, catgpt::VALUE_NUM_BINS>& probs)
{
    constexpr int N = catgpt::VALUE_NUM_BINS;
    constexpr float bin_width = 2.0f / static_cast<float>(N);
    ApproxWDL out{0.f, 0.f, 0.f};
    for (int i = 0; i < N; ++i) {
        float c = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
        if (c > 0.5f)        out.w += probs[i];
        else if (c < -0.5f)  out.l += probs[i];
        else                 out.d += probs[i];
    }
    return out;
}

void print_board(const chess::Board& board) {
    std::cout << "\n  ┌───┬───┬───┬───┬───┬───┬───┬───┐\n";
    for (int rank = 7; rank >= 0; --rank) {
        std::cout << "  " << (rank + 1) << " │";
        for (int file = 0; file < 8; ++file) {
            auto sq = chess::Square(rank * 8 + file);
            auto piece = board.at(sq);
            char c = '.';
            if (piece != chess::Piece::NONE) {
                static const char* piece_chars = "PNBRQKpnbrqk";
                int idx = static_cast<int>(piece.internal());
                if (idx >= 0 && idx < 12) c = piece_chars[idx];
            }
            std::cout << " " << c << " │";
        }
        std::cout << "\n";
        if (rank > 0) std::cout << "    ├───┼───┼───┼───┼───┼───┼───┼───┤\n";
    }
    std::cout << "    └───┴───┴───┴───┴───┴───┴───┴───┘\n";
    std::cout << "      a   b   c   d   e   f   g   h\n\n";
}

bool looks_like_fen(const std::string& s) {
    return s.find(' ') != std::string::npos && s.find('/') != std::string::npos;
}

}  // namespace

int main(int argc, char* argv[]) {
    fs::path network_path;
    if (const char* env = std::getenv("CATGPT_NETWORK")) {
        network_path = env;
    } else {
        network_path = "/home/shadeform/CatGPT/main.trt";
    }
    std::string fen;

    int arg_idx = 1;
    if (argc > 1) {
        std::string first = argv[1];
        // If first arg looks like a FEN, the user is using the implicit
        // default network path; otherwise treat it as the network path.
        if (!looks_like_fen(first) &&
            (first.ends_with(".network") || first.ends_with(".trt") ||
             fs::exists(first))) {
            network_path = first;
            arg_idx = 2;
        }
    }

    if (argc > arg_idx) {
        fen = argv[arg_idx];
    } else {
        std::cout << "Enter FEN (blank = starting position): ";
        std::getline(std::cin, fen);
        if (fen.empty()) {
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    }

    if (!fs::exists(network_path)) {
        std::println(stderr, "Error: network file not found: {}", network_path.string());
        std::println(stderr, "Usage: {} [network_path] [FEN]", argv[0]);
        return 1;
    }

    try {
        chess::Board board(fen);

        SinglePositionEvaluator evaluator(network_path);
        std::cerr.flush();

        auto tokens = catgpt::tokenize<SEQ_LENGTH>(board, catgpt::NO_HALFMOVE_CONFIG);
        auto output = evaluator.evaluate(tokens);

        // ── Header & position ──
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "                    CatGPT Position Analysis\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";

        print_board(board);
        std::cout << "FEN: " << board.getFen() << "\n";
        std::cout << "Side to move: "
                  << (board.sideToMove() == chess::Color::WHITE ? "White" : "Black")
                  << "\n\n";

        // ── Value head (scalar) ──
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                       VALUE  (side-to-move perspective)\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        const float v_unit = output.value;          // [0, 1]
        const float q_11   = 2.0f * v_unit - 1.0f;  // [-1, 1]
        const int   cp     = static_cast<int>(100.7066f * std::tan(q_11 * 1.5637541897f));

        std::cout << "  value head:  " << std::fixed << std::setprecision(4) << v_unit
                  << "   (Q = " << std::showpos << std::setprecision(3) << q_11
                  << std::noshowpos << ", " << std::showpos << cp << std::noshowpos
                  << " cp)\n";
        std::cout << "  verdict   :  " << interpret_value(v_unit) << "\n\n";

        // Approximate WDL synthesized from the 81-bin distribution.
        auto wdl = approx_wdl_from_distribution(output.value_probs);
        std::cout << "  Approx WDL (from BestQ bins, |Q|>0.5 = decisive):\n";
        std::cout << "    Win:  " << std::fixed << std::setprecision(1)
                  << (wdl.w * 100.0f) << "%   "; print_prob_bar(wdl.w); std::cout << "\n";
        std::cout << "    Draw: " << std::fixed << std::setprecision(1)
                  << (wdl.d * 100.0f) << "%   "; print_prob_bar(wdl.d); std::cout << "\n";
        std::cout << "    Loss: " << std::fixed << std::setprecision(1)
                  << (wdl.l * 100.0f) << "%   "; print_prob_bar(wdl.l); std::cout << "\n\n";

        // ── BestQ distribution ──
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                    BESTQ DISTRIBUTION (81 bins)\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";
        print_value_histogram(output.value_probs);
        std::cout << "\n";

        // ── Policy ──
        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                            POLICY\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        auto policy_probs = softmax_legal_moves(board, output.policy);

        if (output.has_optimistic_policy) {
            auto opt_probs = softmax_legal_moves(board, output.optimistic_policy);

            std::cout << "All legal moves (sorted by vanilla policy):\n\n";
            std::cout << "  Rank │  Move  │ Policy │ OptPol │ Visual (Policy / OptPol)\n";
            std::cout << "  ─────┼────────┼────────┼────────┼──────────────────────────────────────\n";

            for (std::size_t i = 0; i < policy_probs.size(); ++i) {
                const auto& [move, prob] = policy_probs[i];
                float opt_prob = 0.0f;
                for (const auto& [m, p] : opt_probs) {
                    if (m == move) { opt_prob = p; break; }
                }

                std::cout << "  " << std::setw(4) << (i + 1) << " │ "
                          << std::setw(6) << chess::uci::moveToUci(move) << " │ "
                          << std::setw(5) << std::fixed << std::setprecision(2)
                          << (prob * 100.0f) << "% │ "
                          << std::setw(5) << std::fixed << std::setprecision(2)
                          << (opt_prob * 100.0f) << "% │ ";
                print_prob_bar(prob, 15);
                std::cout << " ";
                print_prob_bar(opt_prob, 15);
                std::cout << "\n";
            }
        } else {
            std::cout << "All legal moves (sorted by probability):\n\n";
            std::cout << "  Rank │  Move  │  Prob  │ Visual\n";
            std::cout << "  ─────┼────────┼────────┼────────────────────────────\n";
            for (std::size_t i = 0; i < policy_probs.size(); ++i) {
                const auto& [move, prob] = policy_probs[i];
                std::cout << "  " << std::setw(4) << (i + 1) << " │ "
                          << std::setw(6) << chess::uci::moveToUci(move) << " │ "
                          << std::setw(5) << std::fixed << std::setprecision(2)
                          << (prob * 100.0f) << "% │ ";
                print_prob_bar(prob, 22);
                std::cout << "\n";
            }
        }

        std::cout << "\nTotal legal moves: " << policy_probs.size() << "\n";
        std::cout << "\n═══════════════════════════════════════════════════════════════\n";

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
    return 0;
}
