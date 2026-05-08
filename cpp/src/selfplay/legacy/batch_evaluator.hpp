/**
 * Batched TensorRT Evaluator with coroutine integration (libcoro legacy).
 *
 * Runs on a dedicated GPU thread. Search coroutines submit EvalRequests
 * to a thread-safe queue; the GPU thread drains the queue, batches
 * positions, runs TensorRT inference, distributes results, and resumes
 * the coroutines on the worker thread pool.
 *
 * Bucketed batching:
 *   - The engine is built with one optimization profile per bucket size
 *     in `kBucketSizes` (see scripts/trt.sh). Each bucket has a profile
 *     pinned at min == opt == max, so kernels are tuned for that exact
 *     batch size.
 *   - Each bucket gets its own IExecutionContext bound to its profile,
 *     with input shape and tensor addresses preset at construction. The
 *     H2D copy, enqueueV3, and D2H copies are then captured into a
 *     per-bucket cudaGraphExec, so the hot path is just:
 *       host memcpy into pinned input → cudaGraphLaunch →
 *       cudaStreamSynchronize → host memcpy out of pinned outputs.
 *     No setInputShape, no setTensorAddress, no per-launch TRT setup,
 *     no padding.
 *   - The GPU thread drains exactly `pick_bucket(min(queue.size,
 *     max_batch_size))` requests per iteration. Leftovers stay queued.
 *
 * Lives in `catgpt::legacy` to coexist with the libfork canonical version
 * (`catgpt::BatchEvaluator`) at cpp/src/selfplay/batch_evaluator.hpp. The
 * subnamespace is `legacy` rather than `coro` so it does not shadow the
 * libcoro `::coro::` namespace inside `namespace catgpt { ... }`.
 */

#ifndef CATGPT_SELFPLAY_LEGACY_BATCH_EVALUATOR_HPP
#define CATGPT_SELFPLAY_LEGACY_BATCH_EVALUATOR_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <print>
#include <thread>
#include <unordered_map>
#include <vector>

#include <coro/thread_pool.hpp>

#include "../../engine/nn_constants.hpp"
#include "../../engine/policy.hpp"
#include "../../engine/trt_runtime.hpp"
#include "eval_request.hpp"

namespace fs = std::filesystem;

namespace catgpt::legacy {

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
     * Bucket sizes that the engine has tuned optimization profiles for.
     * Must stay in sync with scripts/trt.sh — load-time validation throws
     * if any bucket <= max_batch_size_ is missing in the engine.
     */
    static constexpr std::array<int, 12> kBucketSizes = {
        1, 2, 3, 4, 6, 8, 12, 18, 26, 36, 56, 112,
    };

    /**
     * @param engine_path   Path to the serialized TensorRT engine.
     * @param thread_pool   Shared pointer to the worker thread pool
     *                      (coroutines are resumed here after GPU eval).
     * @param max_batch_size Soft cap on bucket selection. Only buckets
     *                      <= max_batch_size are usable. Effective max
     *                      batch is the largest bucket <= max_batch_size.
     */
    BatchEvaluator(const fs::path& engine_path,
                   std::shared_ptr<::coro::thread_pool> thread_pool,
                   int max_batch_size = 32)
        : thread_pool_(std::move(thread_pool))
        , max_batch_size_(max_batch_size > 0 ? max_batch_size : 1)
        , shutdown_(false)
        , total_evals_(0)
    {
        compute_effective_buckets();
        load_engine(engine_path);
        setup_io();
        discover_profiles();
        allocate_buffers();
        setup_contexts();
        capture_graphs();

        // Start the GPU thread
        gpu_thread_ = std::jthread([this](std::stop_token st) { gpu_loop(st); });
    }

    ~BatchEvaluator() {
        shutdown();

        // Destroy captured graph executables first; each holds references
        // to TRT context state, the stream, and the d_*/h_* buffers.
        for (auto& [bucket, exec] : graph_execs_) {
            if (exec) cudaGraphExecDestroy(exec);
        }
        graph_execs_.clear();

        // Destroy execution contexts before freeing the stream/buffers
        // they were bound to (and before the engine that produced them).
        contexts_.clear();

        // Free CUDA resources
        if (stream_) cudaStreamDestroy(stream_);
        if (d_input_) cudaFree(d_input_);
        if (d_value_) cudaFree(d_value_);
        if (d_value_probs_) cudaFree(d_value_probs_);
        if (d_wdl_) cudaFree(d_wdl_);
        if (d_policy_) cudaFree(d_policy_);
        if (h_input_) cudaFreeHost(h_input_);
        if (h_value_) cudaFreeHost(h_value_);
        if (h_value_probs_) cudaFreeHost(h_value_probs_);
        if (h_wdl_) cudaFreeHost(h_wdl_);
        if (h_policy_) cudaFreeHost(h_policy_);
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
        batch.reserve(largest_effective_bucket());

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

                // Drain exactly `pick_bucket(queue.size())` requests.
                // Leftovers stay queued for the next iteration; this avoids
                // padding and keeps every batch on a tuned profile.
                int count = pick_bucket(static_cast<int>(queue_.size()));
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

        // batch_size is always a bucket size (drain logic guarantees this),
        // so a captured cudaGraphExec is always present for it.

        // Pack tokens into the pinned input buffer (the graph's H2D copy
        // reads exactly this region; tail slots are never touched).
        for (int b = 0; b < batch_size; ++b) {
            std::memcpy(h_input_ + b * SEQ_LENGTH,
                        batch[b]->tokens.data(),
                        SEQ_LENGTH * sizeof(std::int32_t));
        }

        // Launch the captured graph: H2D + enqueueV3 + D2H, all sized for
        // this bucket. One CUDA API call instead of ~7.
        CATGPT_CUDA_CHECK(cudaGraphLaunch(graph_execs_.at(batch_size), stream_));
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
    }

    void setup_io() {
        int num_io = engine_->getNbIOTensors();

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
                    if (policy_output_name_.empty()) policy_output_name_ = name_str;
                }
            }
        }

        if (input_name_.empty()) throw std::runtime_error("Could not find input tensor");
        if (value_output_name_.empty()) throw std::runtime_error("Could not find value output tensor");
        if (policy_output_name_.empty()) throw std::runtime_error("Could not find policy output tensor");

        std::println(stderr, "[BatchEvaluator] IO: input='{}', value='{}', bestq_probs='{}', wdl='{}', policy='{}'",
                     input_name_, value_output_name_, value_probs_output_name_, wdl_output_name_, policy_output_name_);
    }

    void allocate_buffers() {
        // Size to the largest bucket we might actually use (== effective max).
        const int B = largest_effective_bucket();

        CATGPT_CUDA_CHECK(cudaStreamCreate(&stream_));

        // Device buffers (sized for largest effective bucket; shared across contexts)
        CATGPT_CUDA_CHECK(cudaMalloc(&d_input_, B * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_, B * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_value_probs_, B * VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_wdl_, B * 3 * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMalloc(&d_policy_, B * POLICY_SIZE * sizeof(float)));

        // Pinned host buffers
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_input_, B * SEQ_LENGTH * sizeof(std::int32_t)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_, B * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_value_probs_, B * VALUE_NUM_BINS * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_wdl_, B * 3 * sizeof(float)));
        CATGPT_CUDA_CHECK(cudaMallocHost(&h_policy_, B * POLICY_SIZE * sizeof(float)));

        std::println(stderr, "[BatchEvaluator] Allocated buffers for max effective bucket={}", B);
    }

    /**
     * Build effective_buckets_: the subset of kBucketSizes that are <=
     * max_batch_size_. Bucket 1 is always present so selection always
     * returns a valid bucket.
     */
    void compute_effective_buckets() {
        effective_buckets_.clear();
        effective_buckets_.reserve(kBucketSizes.size());
        for (int b : kBucketSizes) {
            if (b <= max_batch_size_) effective_buckets_.push_back(b);
        }
        if (effective_buckets_.empty()) {
            // max_batch_size_ < 1 would have been clamped to 1 in the
            // constructor; treat any other oddity defensively.
            effective_buckets_.push_back(kBucketSizes.front());
        }
    }

    /**
     * Largest bucket we'll ever drain. effective_buckets_ is sorted
     * ascending (kBucketSizes is), so the back element is the max.
     */
    int largest_effective_bucket() const {
        return effective_buckets_.back();
    }

    /**
     * Largest bucket b in effective_buckets_ with b <= n.
     * Always >= 1 since effective_buckets_.front() == 1 (or the first
     * bucket >= 1, but that is 1). Caller must pass n >= 1.
     */
    int pick_bucket(int n) const {
        int chosen = effective_buckets_.front();
        for (int b : effective_buckets_) {
            if (b > n) break;
            chosen = b;
        }
        return chosen;
    }

    /**
     * Enumerate optimization profiles in the engine and build
     * bucket_to_profile_. Throws if any expected bucket (in
     * effective_buckets_) is missing.
     */
    void discover_profiles() {
        const int num_profiles = engine_->getNbOptimizationProfiles();
        if (num_profiles < 1) {
            throw std::runtime_error("Engine has no optimization profiles");
        }

        bucket_to_profile_.clear();
        for (int i = 0; i < num_profiles; ++i) {
            auto opt_dims = engine_->getProfileShape(
                input_name_.c_str(), i, nvinfer1::OptProfileSelector::kOPT);
            if (opt_dims.nbDims < 1) continue;
            const int batch = static_cast<int>(opt_dims.d[0]);
            bucket_to_profile_[batch] = i;
        }

        // Validate: every effective bucket has a matching profile.
        for (int b : effective_buckets_) {
            if (!bucket_to_profile_.contains(b)) {
                std::string expected;
                for (int e : effective_buckets_) {
                    if (!expected.empty()) expected += ", ";
                    expected += std::to_string(e);
                }
                std::string discovered;
                for (const auto& [k, v] : bucket_to_profile_) {
                    if (!discovered.empty()) discovered += ", ";
                    discovered += std::to_string(k);
                }
                throw std::runtime_error(std::format(
                    "Engine missing optimization profile for bucket {}. "
                    "Expected (effective): [{}]. Discovered (in engine): [{}]. "
                    "Rebuild the engine with scripts/trt.sh.",
                    b, expected, discovered));
            }
        }

        std::println(stderr, "[BatchEvaluator] Discovered {} optimization profile(s); "
                             "all {} effective bucket(s) covered",
                     num_profiles, effective_buckets_.size());
    }

    /**
     * Create one IExecutionContext per effective bucket, bind it to the
     * bucket's profile, set input shape (min == opt == max), and bind
     * tensor addresses to the shared d_* device buffers.
     *
     * After this, the hot path never touches setInputShape /
     * setTensorAddress / setOptimizationProfileAsync.
     */
    void setup_contexts() {
        contexts_.clear();
        contexts_.reserve(effective_buckets_.size());

        for (int b : effective_buckets_) {
            const int profile_idx = bucket_to_profile_.at(b);

            std::unique_ptr<nvinfer1::IExecutionContext> ctx(
                engine_->createExecutionContext());
            if (!ctx) {
                throw std::runtime_error(std::format(
                    "Failed to create IExecutionContext for bucket {}", b));
            }

            if (!ctx->setOptimizationProfileAsync(profile_idx, stream_)) {
                throw std::runtime_error(std::format(
                    "Failed to bind optimization profile {} for bucket {}",
                    profile_idx, b));
            }
            CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

            // min == opt == max for these profiles, so this is one-time.
            nvinfer1::Dims input_dims;
            input_dims.nbDims = 2;
            input_dims.d[0] = b;
            input_dims.d[1] = SEQ_LENGTH;
            ctx->setInputShape(input_name_.c_str(), input_dims);

            // Bind addresses to the shared device buffers.
            ctx->setTensorAddress(input_name_.c_str(), d_input_);
            ctx->setTensorAddress(value_output_name_.c_str(), d_value_);
            if (!value_probs_output_name_.empty()) {
                ctx->setTensorAddress(value_probs_output_name_.c_str(), d_value_probs_);
            }
            if (!wdl_output_name_.empty()) {
                ctx->setTensorAddress(wdl_output_name_.c_str(), d_wdl_);
            }
            ctx->setTensorAddress(policy_output_name_.c_str(), d_policy_);

            contexts_.emplace(b, std::move(ctx));
        }

        std::println(stderr,
                     "[BatchEvaluator] Set up {} bucket context(s) (max_batch_size={}, "
                     "max_effective_bucket={})",
                     contexts_.size(), max_batch_size_, largest_effective_bucket());
    }

    /**
     * Capture one cudaGraphExec per effective bucket. The graph wraps:
     *   - H2D: bucket * SEQ_LENGTH int32 tokens
     *   - enqueueV3 on the bucket's IExecutionContext
     *   - D2H: value, value_probs, [wdl], policy
     *
     * After this, process_batch only does host memcpys + cudaGraphLaunch
     * + cudaStreamSynchronize. No per-launch TRT setup, no per-launch
     * cudaMemcpyAsync host calls.
     */
    void capture_graphs() {
        graph_execs_.clear();
        graph_execs_.reserve(effective_buckets_.size());
        for (int b : effective_buckets_) {
            graph_execs_.emplace(b, capture_graph(b));
        }
        std::println(stderr,
                     "[BatchEvaluator] Captured {} CUDA graph(s) for buckets [{}]",
                     graph_execs_.size(), join_buckets(effective_buckets_));
    }

    /**
     * Capture a single bucket's H2D + enqueueV3 + D2H into a graph and
     * return the instantiated executable. Performs one TRT warmup
     * inference outside capture so internal state is stable, per the
     * TRT-with-CUDA-graphs guidance.
     */
    cudaGraphExec_t capture_graph(int bucket) {
        auto& ctx = *contexts_.at(bucket);

        // Warmup: run the context once so any one-time TRT internal
        // allocations / state setup happen before we start capturing.
        if (!ctx.enqueueV3(stream_)) {
            throw std::runtime_error(std::format(
                "TRT warmup enqueueV3 failed for bucket {}", bucket));
        }
        CATGPT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        const std::size_t input_bytes  = bucket * SEQ_LENGTH * sizeof(std::int32_t);
        const std::size_t value_bytes  = bucket * sizeof(float);
        const std::size_t vprobs_bytes = bucket * VALUE_NUM_BINS * sizeof(float);
        const std::size_t wdl_bytes    = bucket * 3 * sizeof(float);
        const std::size_t policy_bytes = bucket * POLICY_SIZE * sizeof(float);

        CATGPT_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));

        // H2D
        CATGPT_CUDA_CHECK(cudaMemcpyAsync(d_input_, h_input_, input_bytes,
                                          cudaMemcpyHostToDevice, stream_));

        // Inference
        if (!ctx.enqueueV3(stream_)) {
            cudaGraph_t dropped;
            cudaStreamEndCapture(stream_, &dropped);
            if (dropped) cudaGraphDestroy(dropped);
            throw std::runtime_error(std::format(
                "TRT enqueueV3 failed during graph capture for bucket {}", bucket));
        }

        // D2H
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

        cudaGraph_t graph = nullptr;
        CATGPT_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));

        cudaGraphExec_t exec = nullptr;
        CATGPT_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        // The executable retains everything it needs; the source graph
        // can be freed.
        cudaGraphDestroy(graph);

        return exec;
    }

    static std::string join_buckets(const std::vector<int>& bs) {
        std::string s;
        for (int b : bs) {
            if (!s.empty()) s += ", ";
            s += std::to_string(b);
        }
        return s;
    }

    // ─── Members ────────────────────────────────────────────────────────

    // Thread pool for resuming coroutines
    std::shared_ptr<::coro::thread_pool> thread_pool_;

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

    // Buckets <= max_batch_size_, ascending. Source of truth for
    // pick_bucket and the keys of contexts_.
    std::vector<int> effective_buckets_;

    // bucket_size -> engine optimization profile index, populated by
    // discover_profiles() at construction.
    std::unordered_map<int, int> bucket_to_profile_;

    // bucket_size -> IExecutionContext. Each context is bound to its
    // bucket's profile with input shape and tensor addresses preset.
    std::unordered_map<int, std::unique_ptr<nvinfer1::IExecutionContext>> contexts_;

    // bucket_size -> captured cudaGraphExec wrapping H2D + enqueueV3 +
    // D2H for that bucket. Owned (destroyed in ~BatchEvaluator before
    // contexts_ and CUDA buffers).
    std::unordered_map<int, cudaGraphExec_t> graph_execs_;

    std::string input_name_;
    std::string value_output_name_;
    std::string value_probs_output_name_;
    std::string wdl_output_name_;
    std::string policy_output_name_;

    cudaStream_t stream_ = nullptr;

    // Device buffers
    std::int32_t* d_input_ = nullptr;
    float* d_value_ = nullptr;
    float* d_value_probs_ = nullptr;
    float* d_wdl_ = nullptr;
    float* d_policy_ = nullptr;

    // Pinned host buffers
    std::int32_t* h_input_ = nullptr;
    float* h_value_ = nullptr;
    float* h_value_probs_ = nullptr;
    float* h_wdl_ = nullptr;
    float* h_policy_ = nullptr;
};

// ─── EvalAwaitable::await_suspend (needs BatchEvaluator to be complete) ────

inline void EvalAwaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    request_.continuation = h;
    evaluator_.submit(&request_);
}

}  // namespace catgpt::legacy

#endif  // CATGPT_SELFPLAY_LEGACY_BATCH_EVALUATOR_HPP
