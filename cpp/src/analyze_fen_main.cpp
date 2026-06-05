/**
 * CatGPT FEN Analyzer — Human-readable NN output for a single position.
 *
 * Self-contained: loads a packed .network or raw .trt engine, runs ONE
 * synchronous TRT inference, pretty-prints WDL + BestQ + optimistic policy
 * over legal moves. No libcoro, libfork, or search.
 *
 * Model I/O (gather-aware export):
 *   in_0  tokens         int32 (batch, 64)
 *   in_1  legal_indices  int32 (batch, MAX_LEGAL_MOVES)
 *   out   wdl_logit, bestq_probs, optimistic_policy_legal_logit
 *
 * Usage:
 *   catgpt_analyze [network_path] [FEN]
 *   catgpt_analyze [network_path] < fen.txt
 *
 * Default network: $CATGPT_NETWORK or /home/shadeform/CatGPT/S4.network
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
#include <fstream>
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

constexpr int SEQ_LENGTH = 64;
constexpr std::size_t WDL_NUM_CLASSES = 3;

inline float wdl_logits_to_q(const std::array<float, WDL_NUM_CLASSES>& logits) noexcept
{
    const float m = std::max({logits[0], logits[1], logits[2]});
    const float ew = std::exp(logits[0] - m);
    const float ed = std::exp(logits[1] - m);
    const float el = std::exp(logits[2] - m);
    const float inv_z = 1.0f / (ew + ed + el);
    return (ew - el) * inv_z;
}

inline float wdl_logits_to_value(const std::array<float, WDL_NUM_CLASSES>& logits) noexcept
{
    return 0.5f * (wdl_logits_to_q(logits) + 1.0f);
}

struct WDLProbs {
    float w, d, l;
};

WDLProbs softmax_wdl(const std::array<float, WDL_NUM_CLASSES>& logits)
{
    const float m = std::max({logits[0], logits[1], logits[2]});
    const float ew = std::exp(logits[0] - m);
    const float ed = std::exp(logits[1] - m);
    const float el = std::exp(logits[2] - m);
    const float z = ew + ed + el;
    return {ew / z, ed / z, el / z};
}

struct AnalysisOutput {
    std::array<float, WDL_NUM_CLASSES>                     wdl_logits;
    std::array<float, catgpt::VALUE_NUM_BINS>            value_probs;
    std::array<float, catgpt::MAX_LEGAL_MOVES>           legal_policy_logits;
};

class SinglePositionEvaluator {
public:
    explicit SinglePositionEvaluator(const fs::path& engine_path) {
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");

        std::array<char, 16> magic{};
        {
            std::ifstream f(engine_path, std::ios::binary);
            if (!f) {
                throw std::runtime_error(
                    std::format("cannot open {}", engine_path.string()));
            }
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
        context_.reset();
        engine_.reset();
        runtime_.reset();

        if (stream_) cudaStreamDestroy(stream_);
        if (d_input_) cudaFree(d_input_);
        if (d_legal_indices_) cudaFree(d_legal_indices_);
        if (d_wdl_) cudaFree(d_wdl_);
        if (d_value_probs_) cudaFree(d_value_probs_);
        if (d_legal_policy_) cudaFree(d_legal_policy_);
        if (h_input_) cudaFreeHost(h_input_);
        if (h_legal_indices_) cudaFreeHost(h_legal_indices_);
        if (h_wdl_) cudaFreeHost(h_wdl_);
        if (h_value_probs_) cudaFreeHost(h_value_probs_);
        if (h_legal_policy_) cudaFreeHost(h_legal_policy_);
    }

    SinglePositionEvaluator(const SinglePositionEvaluator&) = delete;
    SinglePositionEvaluator& operator=(const SinglePositionEvaluator&) = delete;

    AnalysisOutput evaluate(
        const std::array<std::uint8_t, SEQ_LENGTH>& tokens,
        const std::array<std::int32_t, catgpt::MAX_LEGAL_MOVES>& legal_indices)
    {
        for (int i = 0; i < SEQ_LENGTH; ++i) {
            h_input_[i] = static_cast<std::int32_t>(tokens[i]);
        }
        for (int i = 0; i < catgpt::MAX_LEGAL_MOVES; ++i) {
            h_legal_indices_[i] = legal_indices[i];
        }

        CATGPT_CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_,
                                          bucket_ * SEQ_LENGTH * sizeof(std::int32_t),
                                          cudaMemcpyHostToDevice, stream_));
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(
            d_legal_indices_, h_legal_indices_,
            bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, stream_));

        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 failed");
        }

        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_wdl_, d_wdl_,
                                          bucket_ * WDL_NUM_CLASSES * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream_));
        if (!value_probs_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(
                h_value_probs_, d_value_probs_,
                bucket_ * catgpt::VALUE_NUM_BINS * sizeof(float),
                cudaMemcpyDeviceToHost, stream_));
        }
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(
            h_legal_policy_, d_legal_policy_,
            bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        AnalysisOutput out{};
        std::memcpy(out.wdl_logits.data(), h_wdl_,
                    WDL_NUM_CLASSES * sizeof(float));
        if (!value_probs_output_name_.empty()) {
            std::memcpy(out.value_probs.data(), h_value_probs_,
                        catgpt::VALUE_NUM_BINS * sizeof(float));
        }
        std::memcpy(out.legal_policy_logits.data(), h_legal_policy_,
                    catgpt::MAX_LEGAL_MOVES * sizeof(float));
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

        bucket_ = chosen->bucket_size;
        profile_idx_ = 0;

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
        if (!f) {
            throw std::runtime_error(
                std::format("cannot open {}", path.string()));
        }
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

        std::string input_name;
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                auto dims = engine_->getTensorShape(name);
                if (dims.nbDims == 2 && dims.d[1] == SEQ_LENGTH) {
                    input_name = name;
                    break;
                }
            }
        }
        if (input_name.empty()) throw std::runtime_error("no tokens input in engine");

        const int n_profiles = engine_->getNbOptimizationProfiles();
        if (n_profiles < 1) throw std::runtime_error("engine has no profiles");
        int best_profile = 0;
        int best_batch = std::numeric_limits<int>::max();
        for (int p = 0; p < n_profiles; ++p) {
            auto opt = engine_->getProfileShape(
                input_name.c_str(), p, nvinfer1::OptProfileSelector::kOPT);
            if (opt.nbDims < 1) continue;
            const int b = static_cast<int>(opt.d[0]);
            if (b > 0 && b < best_batch) {
                best_batch = b;
                best_profile = p;
            }
        }
        if (best_batch == std::numeric_limits<int>::max()) {
            throw std::runtime_error("could not determine batch size from profiles");
        }
        profile_idx_ = best_profile;
        bucket_ = best_batch;

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
                if (dims.nbDims == 2 && dims.d[1] == SEQ_LENGTH) {
                    input_name_ = name;
                } else if (dims.nbDims == 2 &&
                           dims.d[1] == catgpt::MAX_LEGAL_MOVES) {
                    legal_indices_input_name_ = name;
                }
                continue;
            }

            if (name_str.find("policy") != std::string::npos ||
                name_str.find("Policy") != std::string::npos) {
                policy_output_name_ = name;
            } else if (name_str.find("value_probs") != std::string::npos ||
                       name_str.find("bestq_probs") != std::string::npos) {
                value_probs_output_name_ = name;
            } else if (name_str.find("wdl_logit") != std::string::npos ||
                       name_str.find("wdl") != std::string::npos) {
                wdl_output_name_ = name;
            } else {
                if (dims.nbDims == 2 &&
                    dims.d[1] == static_cast<int64_t>(WDL_NUM_CLASSES)) {
                    if (wdl_output_name_.empty()) wdl_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == catgpt::VALUE_NUM_BINS) {
                    if (value_probs_output_name_.empty()) value_probs_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == catgpt::MAX_LEGAL_MOVES) {
                    if (policy_output_name_.empty()) policy_output_name_ = name;
                }
            }
        }

        if (input_name_.empty()) throw std::runtime_error("no tokens input tensor");
        if (legal_indices_input_name_.empty()) {
            throw std::runtime_error(
                "no legal_indices input tensor — rebuild engine with gather export");
        }
        if (wdl_output_name_.empty()) throw std::runtime_error("no WDL output tensor");
        if (policy_output_name_.empty()) {
            throw std::runtime_error("no optimistic policy output tensor");
        }

        std::println(stderr,
                     "[analyze_fen] IO: tokens='{}', legal_indices='{}', "
                     "wdl='{}', bestq_probs='{}', policy='{}'",
                     input_name_, legal_indices_input_name_,
                     wdl_output_name_,
                     value_probs_output_name_.empty() ? "<none>" : value_probs_output_name_,
                     policy_output_name_);
    }

    void allocate_buffers() {
        CATGPT_CUDA_CHECK(cudaStreamCreate(&stream_));

        CATGPT_CUDA_CHECK(cudaMalloc(&d_input_,
                                     bucket_ * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMalloc(
            &d_legal_indices_, bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_wdl_,
                                     bucket_ * WDL_NUM_CLASSES * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_probs_,
                                     bucket_ * catgpt::VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(
            &d_legal_policy_, bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(float)));

        CATGPT_CUDA_CHECK(cudaMallocHost(&h_input_,
                                         bucket_ * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMallocHost(
            &h_legal_indices_, bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_wdl_,
                                         bucket_ * WDL_NUM_CLASSES * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_probs_,
                                         bucket_ * catgpt::VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(
            &h_legal_policy_, bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(float)));

        std::memset(h_input_, 0, bucket_ * SEQ_LENGTH * sizeof(std::int32_t));
        std::memset(h_legal_indices_, 0,
                    bucket_ * catgpt::MAX_LEGAL_MOVES * sizeof(std::int32_t));
    }

    void setup_context() {
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");

        if (!context_->setOptimizationProfileAsync(profile_idx_, stream_)) {
            throw std::runtime_error("setOptimizationProfileAsync failed");
        }
        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        nvinfer1::Dims tokens_dims;
        tokens_dims.nbDims = 2;
        tokens_dims.d[0] = bucket_;
        tokens_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), tokens_dims);

        nvinfer1::Dims legal_dims;
        legal_dims.nbDims = 2;
        legal_dims.d[0] = bucket_;
        legal_dims.d[1] = catgpt::MAX_LEGAL_MOVES;
        context_->setInputShape(legal_indices_input_name_.c_str(), legal_dims);

        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(legal_indices_input_name_.c_str(), d_legal_indices_);
        context_->setTensorAddress(wdl_output_name_.c_str(), d_wdl_);
        if (!value_probs_output_name_.empty()) {
            context_->setTensorAddress(value_probs_output_name_.c_str(), d_value_probs_);
        }
        context_->setTensorAddress(policy_output_name_.c_str(), d_legal_policy_);
    }

    catgpt::TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    cudaStream_t stream_ = nullptr;
    std::int32_t* d_input_ = nullptr;
    std::int32_t* d_legal_indices_ = nullptr;
    float* d_wdl_ = nullptr;
    float* d_value_probs_ = nullptr;
    float* d_legal_policy_ = nullptr;
    std::int32_t* h_input_ = nullptr;
    std::int32_t* h_legal_indices_ = nullptr;
    float* h_wdl_ = nullptr;
    float* h_value_probs_ = nullptr;
    float* h_legal_policy_ = nullptr;

    int bucket_ = 1;
    int profile_idx_ = 0;
    std::string input_name_;
    std::string legal_indices_input_name_;
    std::string wdl_output_name_;
    std::string value_probs_output_name_;
    std::string policy_output_name_;
};

void build_legal_indices(
    const chess::Movelist& legal,
    bool flip_for_black,
    std::array<std::int32_t, catgpt::MAX_LEGAL_MOVES>& indices) noexcept
{
    const int n = legal.size();
    for (int i = 0; i < n; ++i) {
        const auto [from_idx, to_idx] =
            catgpt::encode_move_to_policy_index(legal[i], flip_for_black);
        indices[i] = static_cast<std::int32_t>(
            catgpt::policy_flat_index(from_idx, to_idx));
    }
    for (int i = n; i < catgpt::MAX_LEGAL_MOVES; ++i) indices[i] = 0;
}

std::vector<std::pair<chess::Move, float>> softmax_legal_policy(
    const chess::Movelist& moves,
    const std::array<float, catgpt::MAX_LEGAL_MOVES>& logits)
{
    std::vector<std::pair<chess::Move, float>> out;
    out.reserve(moves.size());
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < moves.size(); ++i) {
        float logit = logits[i];
        out.emplace_back(moves[i], logit);
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
    if (v_unit > 0.9f) return "Winning (very high confidence)";
    if (v_unit > 0.75f) return "Winning";
    if (v_unit > 0.6f) return "Clearly better";
    if (v_unit > 0.55f) return "Small edge";
    if (v_unit >= 0.45f) return "Roughly equal";
    if (v_unit >= 0.4f) return "Small disadvantage";
    if (v_unit >= 0.25f) return "Worse";
    if (v_unit >= 0.1f) return "Losing";
    return "Losing (very high confidence)";
}

void print_prob_bar(float prob, int width = 25) {
    int filled = static_cast<int>(prob * width + 0.5f);
    filled = std::clamp(filled, 0, width);
    std::cout << "[";
    for (int i = 0; i < width; ++i) std::cout << (i < filled ? "█" : "░");
    std::cout << "]";
}

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
    float peak_center_11 =
        -1.0f + (static_cast<float>(peak_bin) + 0.5f) * bin_width;

    std::cout << "  E[Q] = " << std::fixed << std::setprecision(3) << mean_11
              << "  |  Std = " << std::setprecision(3) << stddev
              << "  |  Peak bin " << peak_bin
              << " (Q ≈ " << std::showpos << std::setprecision(3) << peak_center_11
              << std::noshowpos << ")\n\n";

    auto repeat_str = [](const std::string& s, int n) {
        std::string r;
        r.reserve(s.size() * static_cast<std::size_t>(n));
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
        network_path = "/home/shadeform/CatGPT/S4.network";
    }
    std::string fen;

    int arg_idx = 1;
    if (argc > 1) {
        std::string first = argv[1];
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

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        const bool flip = (board.sideToMove() == chess::Color::BLACK);

        std::array<std::int32_t, catgpt::MAX_LEGAL_MOVES> legal_indices{};
        build_legal_indices(legal, flip, legal_indices);

        auto tokens = catgpt::tokenize<SEQ_LENGTH>(board, catgpt::NO_HALFMOVE_CONFIG);
        auto output = evaluator.evaluate(tokens, legal_indices);

        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "                    CatGPT Position Analysis\n";
        std::cout << "═══════════════════════════════════════════════════════════════\n";

        print_board(board);
        std::cout << "FEN: " << board.getFen() << "\n";
        std::cout << "Side to move: "
                  << (board.sideToMove() == chess::Color::WHITE ? "White" : "Black")
                  << "\n\n";

        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                       VALUE  (side-to-move perspective)\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        const WDLProbs wdl = softmax_wdl(output.wdl_logits);
        const float v_unit = wdl_logits_to_value(output.wdl_logits);
        const float q_11 = wdl_logits_to_q(output.wdl_logits);
        const int cp = static_cast<int>(100.7066f * std::tan(q_11 * 1.5637541897f));

        std::cout << "  WDL head (softmax of logits W,D,L):\n";
        std::cout << "    Win:  " << std::fixed << std::setprecision(1)
                  << (wdl.w * 100.0f) << "%   ";
        print_prob_bar(wdl.w);
        std::cout << "\n";
        std::cout << "    Draw: " << std::fixed << std::setprecision(1)
                  << (wdl.d * 100.0f) << "%   ";
        print_prob_bar(wdl.d);
        std::cout << "\n";
        std::cout << "    Loss: " << std::fixed << std::setprecision(1)
                  << (wdl.l * 100.0f) << "%   ";
        print_prob_bar(wdl.l);
        std::cout << "\n\n";

        std::cout << "  scalar:  " << std::fixed << std::setprecision(4) << v_unit
                  << "   (Q = " << std::showpos << std::setprecision(3) << q_11
                  << std::noshowpos << ", " << std::showpos << cp << std::noshowpos
                  << " cp)\n";
        std::cout << "  verdict: " << interpret_value(v_unit) << "\n\n";

        std::cout << "  raw WDL logits: "
                  << std::fixed << std::setprecision(3)
                  << "W=" << output.wdl_logits[0] << " D=" << output.wdl_logits[1]
                  << " L=" << output.wdl_logits[2] << "\n\n";

        if (!output.value_probs.empty() &&
            std::any_of(output.value_probs.begin(), output.value_probs.end(),
                        [](float p) { return p != 0.0f; })) {
            std::cout << "───────────────────────────────────────────────────────────────\n";
            std::cout << "                    BESTQ DISTRIBUTION (81 bins)\n";
            std::cout << "───────────────────────────────────────────────────────────────\n\n";
            print_value_histogram(output.value_probs);
            std::cout << "\n";
        }

        std::cout << "───────────────────────────────────────────────────────────────\n";
        std::cout << "                   OPTIMISTIC POLICY (legal moves)\n";
        std::cout << "───────────────────────────────────────────────────────────────\n\n";

        auto policy_probs = softmax_legal_policy(legal, output.legal_policy_logits);

        std::cout << "All legal moves (softmax over gathered optimistic logits):\n\n";
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

        std::cout << "\nTotal legal moves: " << policy_probs.size() << "\n";
        std::cout << "\n═══════════════════════════════════════════════════════════════\n";

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
    return 0;
}
