/**
 * TensorRT Evaluator for neural network inference.
 *
 * Provides a simple interface for evaluating chess positions using a
 * TensorRT engine. This is a simplified version focused on single-position
 * inference for MCTS.
 */

#ifndef CATGPT_ENGINE_TRT_EVALUATOR_HPP
#define CATGPT_ENGINE_TRT_EVALUATOR_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include "policy.hpp"

namespace fs = std::filesystem;

namespace catgpt {

// CUDA error checking macro
#define CATGPT_CUDA_CHECK(call)                                                   \
    do {                                                                          \
        cudaError_t status = (call);                                              \
        if (status != cudaSuccess) {                                              \
            throw std::runtime_error(                                             \
                std::string("CUDA error: ") + cudaGetErrorString(status));        \
        }                                                                         \
    } while (0)

/**
 * TensorRT logger implementation.
 */
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            const char* level = severity == Severity::kERROR     ? "ERROR"
                                : severity == Severity::kWARNING ? "WARNING"
                                                                 : "INFO";
            std::println(stderr, "[TRT {}] {}", level, msg);
        }
    }
};

// Number of bins in the HL-Gauss value distribution
constexpr int VALUE_NUM_BINS = 81;

/**
 * Result of neural network evaluation.
 *
 * The model exports five heads:
 *   value            — WDL-derived Q value P(W)+0.5*P(D) in [0, 1]
 *   value_probs      — BestQ HL-Gauss distribution (81 bins over [0, 1])
 *   wdl              — Win/Draw/Loss probabilities [W, D, L]
 *   policy           — Move distribution logits (64 * 73 = 4672)
 *   optimistic_policy — Optimistic policy logits (64 * 73 = 4672), trained with value-surprise weighting
 */
struct NNOutput {
    float value;                                    // WDL-derived Q value [0, 1]
    std::array<float, VALUE_NUM_BINS> value_probs;  // BestQ distribution (81 bins over [0, 1])
    std::array<float, 3> wdl;                       // Win/Draw/Loss probabilities [W, D, L]
    std::array<float, POLICY_SIZE> policy;          // Policy logits (64 * 73 = 4672)
    std::array<float, POLICY_SIZE> optimistic_policy;  // Optimistic policy logits (64 * 73 = 4672)
    bool has_optimistic_policy = false;             // Whether optimistic policy is available
};

/**
 * TensorRT evaluator for single-position inference.
 *
 * This class loads a TensorRT engine and provides inference for chess positions.
 * It uses batch size 1 with CUDA graphs for low-latency evaluation.
 */
class TrtEvaluator {
public:
    static constexpr int SEQ_LENGTH = 64;
    static constexpr int VOCAB_SIZE = 26;

    explicit TrtEvaluator(const fs::path& engine_path)
        : logger_()
    {
        load_engine(engine_path);
        setup_io();
        allocate_buffers();
        capture_cuda_graph();
    }

    ~TrtEvaluator() {
        // Free CUDA resources in reverse order
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (stream_) cudaStreamDestroy(stream_);
        if (d_input_) cudaFree(d_input_);
        if (d_value_) cudaFree(d_value_);
        if (d_value_probs_) cudaFree(d_value_probs_);
        if (d_wdl_) cudaFree(d_wdl_);
        if (d_policy_) cudaFree(d_policy_);
        if (d_optimistic_policy_) cudaFree(d_optimistic_policy_);
        if (h_input_) cudaFreeHost(h_input_);
        if (h_value_) cudaFreeHost(h_value_);
        if (h_value_probs_) cudaFreeHost(h_value_probs_);
        if (h_wdl_) cudaFreeHost(h_wdl_);
        if (h_policy_) cudaFreeHost(h_policy_);
        if (h_optimistic_policy_) cudaFreeHost(h_optimistic_policy_);
    }

    // Non-copyable, non-movable
    TrtEvaluator(const TrtEvaluator&) = delete;
    TrtEvaluator& operator=(const TrtEvaluator&) = delete;
    TrtEvaluator(TrtEvaluator&&) = delete;
    TrtEvaluator& operator=(TrtEvaluator&&) = delete;

    /**
     * Evaluate a single position.
     *
     * @param tokens Array of 64 token indices representing the position.
     * @return NNOutput containing value and policy logits.
     */
    NNOutput evaluate(const std::array<std::uint8_t, SEQ_LENGTH>& tokens) {
        // Copy tokens to pinned input buffer (as int32)
        for (int i = 0; i < SEQ_LENGTH; ++i) {
            h_input_[i] = static_cast<std::int32_t>(tokens[i]);
        }

        // Launch the CUDA graph (H2D -> inference -> D2H)
        CATGPT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Build output
        NNOutput output;
        output.value = h_value_[0];
        std::memcpy(output.value_probs.data(), h_value_probs_, VALUE_NUM_BINS * sizeof(float));
        if (h_wdl_) {
            std::memcpy(output.wdl.data(), h_wdl_, 3 * sizeof(float));
        } else {
            output.wdl = {0.0f, 1.0f, 0.0f};  // Default to draw if WDL not available
        }
        std::memcpy(output.policy.data(), h_policy_, POLICY_SIZE * sizeof(float));
        if (h_optimistic_policy_) {
            std::memcpy(output.optimistic_policy.data(), h_optimistic_policy_, POLICY_SIZE * sizeof(float));
            output.has_optimistic_policy = true;
        }

        return output;
    }

private:
    void load_engine(const fs::path& engine_path) {
        // Read engine file
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open engine file: " + engine_path.string());
        }

        auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        if (!file.read(engine_data.data(), size)) {
            throw std::runtime_error("Failed to read engine file");
        }

        // Create runtime and deserialize engine
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize CUDA engine");
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create execution context");
        }
    }

    void setup_io() {
        int num_io = engine_->getNbIOTensors();

        // Collect all policy-shaped outputs (shape batch x 4672)
        std::vector<std::string> policy_outputs;

        for (int i = 0; i < num_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            auto dims = engine_->getTensorShape(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
                continue;
            }

            std::string name_str(name);

            // Try to match by name first (order matters: check specific names before generic)
            if (name_str.find("wdl") != std::string::npos ||
                name_str.find("WDL") != std::string::npos) {
                // Distinguish wdl_probs from wdl_value by shape
                if (dims.nbDims == 2 && dims.d[1] == 3) {
                    wdl_output_name_ = name;
                } else {
                    // wdl_value is a scalar — treat as the value output
                    value_output_name_ = name;
                }
            } else if (name_str.find("value_probs") != std::string::npos ||
                       name_str.find("bestq_probs") != std::string::npos) {
                value_probs_output_name_ = name;
            } else if (name_str.find("value") != std::string::npos ||
                       name_str.find("Value") != std::string::npos) {
                value_output_name_ = name;
            } else if (name_str.find("optimistic_policy") != std::string::npos) {
                optimistic_policy_output_name_ = name;
            } else if (name_str.find("policy") != std::string::npos ||
                       name_str.find("Policy") != std::string::npos) {
                policy_output_name_ = name;
            } else {
                // Fallback: detect by shape
                // Value output: scalar or (batch,) or (batch, 1)
                // Value probs output: (batch, 81)
                // WDL output: (batch, 3)
                // Policy output: (batch, 4672)
                if (dims.nbDims == 1 ||
                    (dims.nbDims == 2 && dims.d[1] == 1)) {
                    if (value_output_name_.empty()) value_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == VALUE_NUM_BINS) {
                    if (value_probs_output_name_.empty()) value_probs_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == 3) {
                    if (wdl_output_name_.empty()) wdl_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == POLICY_SIZE) {
                    // Collect policy-shaped outputs for later assignment
                    policy_outputs.push_back(name_str);
                }
            }
        }

        // Assign policy-shaped outputs: first is vanilla policy, second is optimistic
        for (const auto& pname : policy_outputs) {
            if (policy_output_name_.empty()) {
                policy_output_name_ = pname;
            } else if (optimistic_policy_output_name_.empty()) {
                optimistic_policy_output_name_ = pname;
            }
        }

        if (input_name_.empty() || value_output_name_.empty() || policy_output_name_.empty()) {
            throw std::runtime_error("Could not find all required I/O tensors");
        }
        if (value_probs_output_name_.empty()) {
            std::println(stderr, "[TRT WARNING] bestq_probs output not found - using zeros");
        }
        if (wdl_output_name_.empty()) {
            std::println(stderr, "[TRT WARNING] wdl_probs output not found - using defaults");
        }
        if (!optimistic_policy_output_name_.empty()) {
            std::println(stderr, "[TRT INFO] Optimistic policy head detected: {}", optimistic_policy_output_name_);
        }
    }

    void allocate_buffers() {
        // Create CUDA stream
        CATGPT_CUDA_CHECK(cudaStreamCreate(&stream_));

        // Allocate device buffers
        CATGPT_CUDA_CHECK(cudaMalloc(&d_input_, SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_, sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_probs_, VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_wdl_, 3 * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_policy_, POLICY_SIZE * sizeof(float)));
        if (!optimistic_policy_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMalloc(&d_optimistic_policy_, POLICY_SIZE * sizeof(float)));
        }

        // Allocate pinned host buffers for async transfers
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_input_, SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_, sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_probs_, VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_wdl_, 3 * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_policy_, POLICY_SIZE * sizeof(float)));
        if (!optimistic_policy_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMallocHost(&h_optimistic_policy_, POLICY_SIZE * sizeof(float)));
        }
    }

    void capture_cuda_graph() {
        // Set up tensor addresses
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(value_output_name_.c_str(), d_value_);
        if (!value_probs_output_name_.empty()) {
            context_->setTensorAddress(value_probs_output_name_.c_str(), d_value_probs_);
        }
        if (!wdl_output_name_.empty()) {
            context_->setTensorAddress(wdl_output_name_.c_str(), d_wdl_);
        }
        context_->setTensorAddress(policy_output_name_.c_str(), d_policy_);
        if (!optimistic_policy_output_name_.empty()) {
            context_->setTensorAddress(optimistic_policy_output_name_.c_str(), d_optimistic_policy_);
        }

        // Set input shape (batch size 1)
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = 1;  // batch size
        input_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), input_dims);

        // Begin graph capture
        CATGPT_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

        // H2D transfer
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(
            d_input_, h_input_, SEQ_LENGTH * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, stream_));

        // TensorRT inference
        if (!context_->enqueueV3(stream_)) {
            cudaGraph_t dummy;
            cudaStreamEndCapture(stream_, &dummy);
            throw std::runtime_error("TensorRT inference failed during graph capture");
        }

        // D2H transfers
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(
            h_value_, d_value_, sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        if (!value_probs_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(
                h_value_probs_, d_value_probs_, VALUE_NUM_BINS * sizeof(float),
                cudaMemcpyDeviceToHost, stream_));
        }
        if (!wdl_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(
                h_wdl_, d_wdl_, 3 * sizeof(float),
                cudaMemcpyDeviceToHost, stream_));
        }
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(
            h_policy_, d_policy_, POLICY_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost, stream_));
        if (!optimistic_policy_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(
                h_optimistic_policy_, d_optimistic_policy_, POLICY_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost, stream_));
        }

        // End capture and instantiate
        cudaGraph_t graph;
        CATGPT_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        CATGPT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0));
        cudaGraphDestroy(graph);
    }

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string value_output_name_;
    std::string value_probs_output_name_;
    std::string wdl_output_name_;
    std::string policy_output_name_;
    std::string optimistic_policy_output_name_;

    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    // Device buffers
    std::int32_t* d_input_ = nullptr;
    float* d_value_ = nullptr;
    float* d_value_probs_ = nullptr;
    float* d_wdl_ = nullptr;
    float* d_policy_ = nullptr;
    float* d_optimistic_policy_ = nullptr;

    // Pinned host buffers
    std::int32_t* h_input_ = nullptr;
    float* h_value_ = nullptr;
    float* h_value_probs_ = nullptr;
    float* h_wdl_ = nullptr;
    float* h_policy_ = nullptr;
    float* h_optimistic_policy_ = nullptr;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_TRT_EVALUATOR_HPP
