/**
 * Batched TensorRT Evaluator with coroutine integration.
 *
 * Runs on a dedicated GPU thread.  Search coroutines submit EvalRequests
 * to a thread-safe queue; the GPU thread drains the queue, batches
 * positions, runs TensorRT inference, distributes results, and resumes
 * the coroutines on the worker thread pool.
 *
 * Batching strategy:
 *   - Wait on the queue until at least one request arrives
 *   - Drain all available requests (up to max_batch_size)
 *   - Run TRT inference on the batch
 *   - Write results back to each request
 *   - Resume each coroutine via thread_pool->resume()
 */

#ifndef CATGPT_SELFPLAY_BATCH_EVALUATOR_HPP
#define CATGPT_SELFPLAY_BATCH_EVALUATOR_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <print>
#include <thread>
#include <vector>

#include <coro/thread_pool.hpp>

#include "../engine/policy.hpp"
#include "../engine/trt_evaluator.hpp"  // VALUE_NUM_BINS, TrtLogger, CATGPT_CUDA_CHECK
#include "eval_request.hpp"

namespace fs = std::filesystem;

namespace catgpt {

/**
 * Batched TensorRT evaluator running on a dedicated GPU thread.
 *
 * Search coroutines call submit() to enqueue an evaluation request,
 * then suspend.  The GPU thread processes batches and resumes the
 * coroutines on the thread pool.
 */
class BatchEvaluator {
public:
    static constexpr int SEQ_LENGTH = 64;
    static constexpr int MAX_SUPPORTED_BATCH = 256;

    /**
     * @param engine_path   Path to the serialized TensorRT engine.
     * @param thread_pool   Shared pointer to the worker thread pool
     *                      (coroutines are resumed here after GPU eval).
     * @param max_batch_size Maximum positions per batch.
     */
    BatchEvaluator(const fs::path& engine_path,
                   std::shared_ptr<coro::thread_pool> thread_pool,
                   int max_batch_size = 32)
        : thread_pool_(std::move(thread_pool))
        , max_batch_size_(max_batch_size)
        , shutdown_(false)
        , total_evals_(0)
    {
        load_engine(engine_path);
        setup_io();
        allocate_buffers();

        // Start the GPU thread
        gpu_thread_ = std::jthread([this](std::stop_token st) { gpu_loop(st); });
    }

    ~BatchEvaluator() {
        shutdown();

        // Free CUDA resources
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
    BatchEvaluator(const BatchEvaluator&) = delete;
    BatchEvaluator& operator=(const BatchEvaluator&) = delete;
    BatchEvaluator(BatchEvaluator&&) = delete;
    BatchEvaluator& operator=(BatchEvaluator&&) = delete;

    /**
     * Submit an evaluation request to the GPU queue.
     *
     * Called from await_suspend — the request pointer must remain valid
     * until the coroutine is resumed (it lives in the coroutine frame).
     */
    void submit(EvalRequest* request) {
        {
            std::lock_guard lock(queue_mutex_);
            queue_.push_back(request);
        }
        queue_cv_.notify_one();
    }

    /**
     * Shut down the GPU thread gracefully.
     * All pending requests are processed before exit.
     */
    void shutdown() {
        if (shutdown_.exchange(true)) return;  // Already shut down

        gpu_thread_.request_stop();
        queue_cv_.notify_all();
        if (gpu_thread_.joinable()) {
            gpu_thread_.join();
        }
    }

    /** Total positions evaluated since construction. */
    [[nodiscard]] std::int64_t total_evals() const noexcept {
        return total_evals_.load(std::memory_order_relaxed);
    }

    /** Total batches dispatched to the GPU since construction. */
    [[nodiscard]] std::int64_t total_batches() const noexcept {
        return total_batches_.load(std::memory_order_relaxed);
    }

private:
    // ─── GPU thread main loop ───────────────────────────────────────────

    void gpu_loop(std::stop_token stop) {
        std::vector<EvalRequest*> batch;
        batch.reserve(max_batch_size_);

        while (true) {
            batch.clear();

            // Wait for at least one request (or shutdown)
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [&] {
                    return !queue_.empty() || stop.stop_requested();
                });

                if (stop.stop_requested() && queue_.empty()) {
                    break;
                }

                // Drain up to max_batch_size requests
                int count = std::min(static_cast<int>(queue_.size()), max_batch_size_);
                for (int i = 0; i < count; ++i) {
                    batch.push_back(queue_.front());
                    queue_.pop_front();
                }
            }

            if (batch.empty()) continue;

            // Run batched inference
            process_batch(batch);
        }
    }

    // ─── Batched inference ──────────────────────────────────────────────

    void process_batch(std::vector<EvalRequest*>& batch) {
        int batch_size = static_cast<int>(batch.size());

        // Pack tokens into the pinned input buffer
        for (int b = 0; b < batch_size; ++b) {
            std::memcpy(h_input_ + b * SEQ_LENGTH,
                        batch[b]->tokens.data(),
                        SEQ_LENGTH * sizeof(std::int32_t));
        }
        // Pad remaining slots with zeros (TRT needs valid memory but values don't matter)
        if (batch_size < max_batch_size_) {
            std::memset(h_input_ + batch_size * SEQ_LENGTH, 0,
                        (max_batch_size_ - batch_size) * SEQ_LENGTH * sizeof(std::int32_t));
        }

        // Set dynamic batch dimension
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), input_dims);

        // H2D transfer
        std::size_t input_bytes = batch_size * SEQ_LENGTH * sizeof(std::int32_t);
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_, input_bytes,
                                          cudaMemcpyHostToDevice, stream_));

        // Run inference
        if (!context_->enqueueV3(stream_)) {
            std::println(stderr, "[BatchEvaluator] TensorRT inference failed for batch_size={}", batch_size);
            // Resume coroutines anyway (with garbage output) to avoid deadlock
        }

        // D2H transfer
        std::size_t value_bytes = batch_size * sizeof(float);
        std::size_t vprobs_bytes = batch_size * VALUE_NUM_BINS * sizeof(float);
        std::size_t wdl_bytes = batch_size * 3 * sizeof(float);
        std::size_t policy_bytes = batch_size * POLICY_SIZE * sizeof(float);

        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_value_, d_value_, value_bytes,
                                          cudaMemcpyDeviceToHost, stream_));
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_value_probs_, d_value_probs_, vprobs_bytes,
                                          cudaMemcpyDeviceToHost, stream_));
        if (!wdl_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_wdl_, d_wdl_, wdl_bytes,
                                              cudaMemcpyDeviceToHost, stream_));
        }
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_policy_, d_policy_, policy_bytes,
                                          cudaMemcpyDeviceToHost, stream_));
        if (!optimistic_policy_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMemcpyAsync(h_optimistic_policy_, d_optimistic_policy_, policy_bytes,
                                              cudaMemcpyDeviceToHost, stream_));
        }

        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Distribute results and resume coroutines
        for (int b = 0; b < batch_size; ++b) {
            auto* req = batch[b];
            req->result.value = h_value_[b];
            std::memcpy(req->result.value_probs.data(),
                        h_value_probs_ + b * VALUE_NUM_BINS,
                        VALUE_NUM_BINS * sizeof(float));
            if (!wdl_output_name_.empty()) {
                std::memcpy(req->result.wdl.data(),
                            h_wdl_ + b * 3,
                            3 * sizeof(float));
            } else {
                req->result.wdl = {0.0f, 1.0f, 0.0f};  // Default to draw
            }
            std::memcpy(req->result.policy.data(),
                        h_policy_ + b * POLICY_SIZE,
                        POLICY_SIZE * sizeof(float));
            if (!optimistic_policy_output_name_.empty()) {
                std::memcpy(req->result.optimistic_policy.data(),
                            h_optimistic_policy_ + b * POLICY_SIZE,
                            POLICY_SIZE * sizeof(float));
                req->result.has_optimistic_policy = true;
            }

            // Resume the coroutine on the worker thread pool
            thread_pool_->resume(req->continuation);
        }

        total_evals_.fetch_add(batch_size, std::memory_order_relaxed);
        total_batches_.fetch_add(1, std::memory_order_relaxed);
    }

    // ─── TensorRT setup ─────────────────────────────────────────────────

    void load_engine(const fs::path& engine_path) {
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

        std::println(stderr, "[BatchEvaluator] Loaded engine: {} ({:.1f} MB)",
                     engine_path.string(), static_cast<double>(size) / (1024.0 * 1024.0));

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_) throw std::runtime_error("Failed to deserialize CUDA engine");

        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("Failed to create execution context");
    }

    void setup_io() {
        int num_io = engine_->getNbIOTensors();

        // Collect all policy-shaped outputs for later assignment
        std::vector<std::string> policy_outputs;

        for (int i = 0; i < num_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            auto dims = engine_->getTensorShape(name);
            std::string name_str(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
                continue;
            }

            // Match by name first (specific names before generic)
            if (name_str.find("wdl") != std::string::npos ||
                name_str.find("WDL") != std::string::npos) {
                // Distinguish wdl_probs (batch, 3) from wdl_value (scalar)
                if (dims.nbDims == 2 && dims.d[1] == 3) {
                    wdl_output_name_ = name;
                } else {
                    value_output_name_ = name;
                }
            } else if (name_str.find("optimistic_policy") != std::string::npos) {
                optimistic_policy_output_name_ = name;
            } else if (name_str.find("policy") != std::string::npos ||
                       name_str.find("Policy") != std::string::npos) {
                policy_output_name_ = name;
            } else if (name_str.find("value_probs") != std::string::npos ||
                       name_str.find("bestq_probs") != std::string::npos) {
                value_probs_output_name_ = name;
            } else if (name_str.find("value") != std::string::npos ||
                       name_str.find("Value") != std::string::npos) {
                if (value_output_name_.empty()) {
                    value_output_name_ = name;
                }
            } else {
                // Fallback by shape
                if (dims.nbDims == 1 || (dims.nbDims == 2 && dims.d[1] == 1)) {
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

        if (input_name_.empty()) throw std::runtime_error("Could not find input tensor");
        if (value_output_name_.empty()) throw std::runtime_error("Could not find value output tensor");
        if (policy_output_name_.empty()) throw std::runtime_error("Could not find policy output tensor");

        std::println(stderr, "[BatchEvaluator] IO: input='{}', value='{}', bestq_probs='{}', wdl='{}', policy='{}'",
                     input_name_, value_output_name_, value_probs_output_name_, wdl_output_name_, policy_output_name_);
        if (!optimistic_policy_output_name_.empty()) {
            std::println(stderr, "[BatchEvaluator] Optimistic policy head detected: {}", optimistic_policy_output_name_);
        }
    }

    void allocate_buffers() {
        int B = max_batch_size_;

        CATGPT_CUDA_CHECK(cudaStreamCreate(&stream_));

        // Device buffers (sized for max batch)
        CATGPT_CUDA_CHECK(cudaMalloc(&d_input_, B * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_, B * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_probs_, B * VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_wdl_, B * 3 * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_policy_, B * POLICY_SIZE * sizeof(float)));
        if (!optimistic_policy_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMalloc(&d_optimistic_policy_, B * POLICY_SIZE * sizeof(float)));
        }

        // Pinned host buffers
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_input_, B * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_, B * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_probs_, B * VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_wdl_, B * 3 * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_policy_, B * POLICY_SIZE * sizeof(float)));
        if (!optimistic_policy_output_name_.empty()) {
            CATGPT_CUDA_CHECK(cudaMallocHost(&h_optimistic_policy_, B * POLICY_SIZE * sizeof(float)));
        }

        // Bind tensor addresses (will rebind input shape per batch)
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

        std::println(stderr, "[BatchEvaluator] Allocated buffers for max_batch_size={}", B);
    }

    // ─── Members ────────────────────────────────────────────────────────

    // Thread pool for resuming coroutines
    std::shared_ptr<coro::thread_pool> thread_pool_;

    // Configuration
    int max_batch_size_;

    // Request queue
    std::deque<EvalRequest*> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // GPU thread
    std::jthread gpu_thread_;
    std::atomic<bool> shutdown_;

    // Stats
    std::atomic<std::int64_t> total_evals_;
    std::atomic<std::int64_t> total_batches_{0};

    // TensorRT state
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

// ─── EvalAwaitable::await_suspend (needs BatchEvaluator to be complete) ────

inline void EvalAwaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    request_.continuation = h;
    evaluator_.submit(&request_);
}

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_BATCH_EVALUATOR_HPP
