/**
 * TensorRT Benchmark for CatGPT Chess Engine
 *
 * Loads a TensorRT engine and benchmarks inference at various batch sizes.
 * The model expects two inputs and three outputs:
 *   - Input  in_0 (tokens):                       int32   (batch, 64)
 *   - Input  in_1 (legal_indices):                int32   (batch, MAX_LEGAL_MOVES)
 *   - Output wdl_logit:                           float32 (batch, 3)
 *   - Output bestq_probs:                         float32 (batch, 81)
 *   - Output optimistic_policy_legal_logit:       float32 (batch, MAX_LEGAL_MOVES)
 *
 * `legal_indices` carries flat indices into the (64*73=4672) policy tensor;
 * the model gathers along that axis on the GPU so only the legal-move logits
 * are D2H'd. For benchmarking we fill the indices with random in-range int32
 * values — content doesn't matter, only shape/dtype, since we measure
 * throughput rather than correctness.
 */

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "catgpt_version.hpp"
#include "engine/network_file.hpp"
#include "engine/nn_constants.hpp"

namespace fs = std::filesystem;

// TensorRT logger implementation
class Logger : public nvinfer1::ILogger {
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

// CUDA error checking macro
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t status = (call);                                                  \
        if (status != cudaSuccess) {                                                  \
            std::println(stderr, "CUDA error at {}:{}: {}", __FILE__, __LINE__,       \
                         cudaGetErrorString(status));                                 \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

// RAII wrapper for CUDA device memory
template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    // Move only
    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }

    // Async copy from pinned host memory
    void copy_from_host_async(const T* host_data, size_t count, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    // Async copy to pinned host memory
    void copy_to_host_async(T* host_data, size_t count, cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// RAII wrapper for pinned (page-locked) host memory
// Pinned memory enables faster and async DMA transfers to/from GPU
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMallocHost(&ptr_, count_ * sizeof(T)));
        }
    }

    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    // Move only
    PinnedBuffer(PinnedBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t size_bytes() const { return count_ * sizeof(T); }

    // Copy from regular host memory
    void copy_from(const T* src, size_t count) {
        std::memcpy(ptr_, src, count * sizeof(T));
    }

    // Access as span for convenience
    std::span<T> span() { return {ptr_, count_}; }
    std::span<const T> span() const { return {ptr_, count_}; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// RAII wrapper for CUDA stream
class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    // Move only
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }
    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    // Begin graph capture on this stream
    void begin_capture() {
        CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    }

    // End graph capture and return the captured graph
    cudaGraph_t end_capture() {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        return graph;
    }

private:
    cudaStream_t stream_ = nullptr;
};

// RAII wrapper for CUDA Graph and its executable instance
class CudaGraphExec {
public:
    CudaGraphExec() = default;

    // Create from a captured graph (takes ownership and destroys the source graph)
    explicit CudaGraphExec(cudaGraph_t graph) {
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0));
        cudaGraphDestroy(graph);  // No longer needed after instantiation
    }

    ~CudaGraphExec() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    }

    // Move only
    CudaGraphExec(CudaGraphExec&& other) noexcept : graph_exec_(other.graph_exec_) {
        other.graph_exec_ = nullptr;
    }
    CudaGraphExec& operator=(CudaGraphExec&& other) noexcept {
        if (this != &other) {
            if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = other.graph_exec_;
            other.graph_exec_ = nullptr;
        }
        return *this;
    }
    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;

    // Launch the graph on a stream
    void launch(cudaStream_t stream) const {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
    }

    bool valid() const { return graph_exec_ != nullptr; }

private:
    cudaGraphExec_t graph_exec_ = nullptr;
};

// RAII wrapper for CUDA event
class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }

    // Move only
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream) const {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    // Make another stream wait for this event
    void wait_on(cudaStream_t stream) const {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
    }

private:
    cudaEvent_t event_ = nullptr;
};

// Number of bins in the HL-Gauss value distribution
static constexpr int32_t VALUE_NUM_BINS = 81;
static constexpr int32_t WDL_NUM_CLASSES = 3;

// MAX_LEGAL_MOVES from cpp/src/engine/nn_constants.hpp; pulled in as int32_t
// for the buffer-sizing arithmetic below.
static constexpr int32_t MAX_LEGAL_MOVES =
    static_cast<int32_t>(catgpt::MAX_LEGAL_MOVES);

inline float wdl_logits_to_q(const float* logits) noexcept
{
    const float m = std::max({logits[0], logits[1], logits[2]});
    const float ew = std::exp(logits[0] - m);
    const float ed = std::exp(logits[1] - m);
    const float el = std::exp(logits[2] - m);
    const float inv_z = 1.0f / (ew + ed + el);
    return (ew - el) * inv_z;
}

// Inference result containing all model outputs
struct InferenceResult {
    std::vector<float> wdl_logits;                 // (batch * 3) - raw WDL logits
    std::vector<float> value_probs;                // (batch * VALUE_NUM_BINS) - bestq HL-Gauss distribution
    std::vector<float> policy_logits;              // (batch * MAX_LEGAL_MOVES) - gathered legal-move logits
};

// TensorRT Engine wrapper with CUDA Graph caching and double buffering
class TrtEngine {
public:
    static constexpr int32_t SEQ_LENGTH = 64;
    static constexpr int32_t VOCAB_SIZE = 26;  // Must match catgpt::VOCAB_SIZE
    // Width of the model's flat (un-gathered) optimistic-policy tensor; only
    // used as the [0, POLICY_SIZE) range for the random legal_indices the
    // benchmark feeds into the GPU gather.
    static constexpr int32_t POLICY_SIZE = 4672;  // 64 * 73 (from_sq * to_sq)
    static constexpr int NUM_BUFFERS = 2;  // Double buffering

    // Batch sizes to cache CUDA graphs for
    static constexpr std::array<int32_t, 12> CACHED_BATCH_SIZES = {1, 2, 3, 4, 6, 8, 12, 18, 26, 36, 56, 112};

    // One bucket = one per-bucket sub-engine (single profile, min == opt ==
    // max == bucket_size).
    struct BucketInfo {
        int32_t bucket_size;
    };

    TrtEngine(const fs::path& network_path, Logger& logger) : logger_(logger) {
        load_engines(network_path);
        setup_io();
        discover_buckets();
        preallocate_cached_buffers();
        setup_double_buffering();
    }

    ~TrtEngine() = default;

    int32_t max_batch_size() const { return max_batch_size_; }

    // Bucket sizes available in the loaded .network, ascending.
    const std::vector<BucketInfo>& buckets() const { return buckets_; }

    // .network files only support fixed bucket sizes. Returns true if
    // batch_size is one of the bucket sizes in this network.
    bool has_bucket(int32_t batch_size) const {
        for (const auto& p : buckets_) {
            if (p.bucket_size == batch_size) return true;
        }
        return false;
    }

    // Register a batch size for CUDA graph caching and pipelined inference.
    // batch_size MUST be one of the bucket sizes baked into the .network.
    void register_batch_size(int32_t batch_size) {
        if (!has_bucket(batch_size)) {
            std::string available;
            for (const auto& p : buckets_) {
                if (!available.empty()) available += ", ";
                available += std::to_string(p.bucket_size);
            }
            throw std::runtime_error(std::format(
                "Batch size {} is not a bucket in this .network. Available: [{}]",
                batch_size, available));
        }
        if (!cached_resources_.contains(batch_size)) {
            auto [it, _] = cached_resources_.emplace(batch_size, CachedBatchResources(batch_size));
            create_cached_context(it->second, batch_size);
        }
        if (!double_buffer_resources_.contains(batch_size)) {
            auto [it, _] = double_buffer_resources_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(batch_size),
                std::forward_as_tuple(batch_size));
            create_slot_contexts(it->second, batch_size);
        }
    }

    // Run inference on a batch of chess positions
    // Input: vector of int32 tokens, size = batch_size * SEQ_LENGTH
    // Output: InferenceResult with values and policy logits
    InferenceResult infer(std::span<const int32_t> input_tokens, int32_t batch_size) {
        if (static_cast<size_t>(batch_size * SEQ_LENGTH) != input_tokens.size()) {
            throw std::runtime_error("Input size mismatch");
        }
        if (!cached_resources_.contains(batch_size)) {
            register_batch_size(batch_size);  // Lazily allocate context+buffers for new batch size
        }
        return infer_with_graph(input_tokens, batch_size, cached_resources_.at(batch_size));
    }

private:
    // Per-bucket cached resources.
    // Each bucket owns its own IExecutionContext, created from that bucket's
    // dedicated sub-engine (single profile pinned at min == opt == max ==
    // bucket). The captured CUDA graph thus references kernels that were
    // tuned specifically for that one shape.
    struct CachedBatchResources {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        CudaBuffer<int32_t> input_buffer;
        CudaBuffer<int32_t> legal_indices_buffer;
        CudaBuffer<float> wdl_buffer;
        CudaBuffer<float> value_probs_buffer;
        CudaBuffer<float> policy_buffer;
        PinnedBuffer<int32_t> pinned_input;
        PinnedBuffer<int32_t> pinned_legal_indices;
        PinnedBuffer<float> pinned_wdl;
        PinnedBuffer<float> pinned_value_probs;
        PinnedBuffer<float> pinned_policy;
        CudaGraphExec graph_exec;

        CachedBatchResources(int32_t batch_size)
            : input_buffer(batch_size * SEQ_LENGTH),
              legal_indices_buffer(batch_size * MAX_LEGAL_MOVES),
              wdl_buffer(batch_size * WDL_NUM_CLASSES),
              value_probs_buffer(batch_size * VALUE_NUM_BINS),
              policy_buffer(batch_size * MAX_LEGAL_MOVES),
              pinned_input(batch_size * SEQ_LENGTH),
              pinned_legal_indices(batch_size * MAX_LEGAL_MOVES),
              pinned_wdl(batch_size * WDL_NUM_CLASSES),
              pinned_value_probs(batch_size * VALUE_NUM_BINS),
              pinned_policy(batch_size * MAX_LEGAL_MOVES) {}
    };

    // Inference using cached CUDA graph
    InferenceResult infer_with_graph(std::span<const int32_t> input_tokens,
                                     int32_t batch_size,
                                     CachedBatchResources& res) {
        // Copy input to pinned buffer (CPU-side, not part of graph). The
        // pinned_legal_indices buffer was filled once at registration with
        // random in-range int32; the captured graph H2Ds the same data every
        // launch (correctness doesn't matter for benchmarking, only shape
        // and that indices are in [0, POLICY_SIZE)).
        res.pinned_input.copy_from(input_tokens.data(), input_tokens.size());

        // Capture graph on first use
        if (!res.graph_exec.valid()) {
            capture_graph(batch_size, res);
        }

        // Launch the cached graph
        res.graph_exec.launch(stream_.get());
        stream_.synchronize();

        // Copy from pinned output buffers (CPU-side, not part of graph)
        InferenceResult result;
        result.wdl_logits.resize(batch_size * WDL_NUM_CLASSES);
        result.value_probs.resize(batch_size * VALUE_NUM_BINS);
        result.policy_logits.resize(batch_size * MAX_LEGAL_MOVES);
        std::memcpy(result.wdl_logits.data(), res.pinned_wdl.get(),
                    batch_size * WDL_NUM_CLASSES * sizeof(float));
        std::memcpy(result.value_probs.data(), res.pinned_value_probs.get(),
                    batch_size * VALUE_NUM_BINS * sizeof(float));
        std::memcpy(result.policy_logits.data(), res.pinned_policy.get(),
                    batch_size * MAX_LEGAL_MOVES * sizeof(float));

        return result;
    }

    // Capture CUDA graph for a batch size, using this batch size's profile-bound context.
    void capture_graph(int32_t batch_size, CachedBatchResources& res) {
        auto& ctx = *res.context;

        // Set up tensor addresses and shapes before capture (all inputs and
        // outputs must be bound).
        ctx.setTensorAddress(input_name_.c_str(), res.input_buffer.get());
        ctx.setTensorAddress(legal_indices_input_name_.c_str(),
                             res.legal_indices_buffer.get());
        ctx.setTensorAddress(wdl_output_name_.c_str(), res.wdl_buffer.get());
        if (!value_probs_output_name_.empty()) {
            ctx.setTensorAddress(value_probs_output_name_.c_str(), res.value_probs_buffer.get());
        }
        ctx.setTensorAddress(policy_output_name_.c_str(), res.policy_buffer.get());

        nvinfer1::Dims tokens_dims;
        tokens_dims.nbDims = 2;
        tokens_dims.d[0] = batch_size;
        tokens_dims.d[1] = SEQ_LENGTH;
        ctx.setInputShape(input_name_.c_str(), tokens_dims);

        nvinfer1::Dims legal_dims;
        legal_dims.nbDims = 2;
        legal_dims.d[0] = batch_size;
        legal_dims.d[1] = MAX_LEGAL_MOVES;
        ctx.setInputShape(legal_indices_input_name_.c_str(), legal_dims);

        const size_t input_count       = batch_size * SEQ_LENGTH;
        const size_t legal_in_count    = batch_size * MAX_LEGAL_MOVES;
        const size_t value_probs_count = batch_size * VALUE_NUM_BINS;
        const size_t policy_count      = batch_size * MAX_LEGAL_MOVES;

        // Begin graph capture
        stream_.begin_capture();

        // H2D transfers (async, captured): tokens + legal_indices
        res.input_buffer.copy_from_host_async(res.pinned_input.get(), input_count, stream_.get());
        res.legal_indices_buffer.copy_from_host_async(
            res.pinned_legal_indices.get(), legal_in_count, stream_.get());

        // TensorRT inference (captured)
        if (!ctx.enqueueV3(stream_.get())) {
            cudaGraph_t dummy;
            cudaStreamEndCapture(stream_.get(), &dummy);  // Clean up capture state
            throw std::runtime_error("TensorRT inference failed during graph capture");
        }

        // D2H transfer (async, captured)
        res.wdl_buffer.copy_to_host_async(res.pinned_wdl.get(),
                                          batch_size * WDL_NUM_CLASSES, stream_.get());
        if (!value_probs_output_name_.empty()) {
            res.value_probs_buffer.copy_to_host_async(res.pinned_value_probs.get(), value_probs_count, stream_.get());
        }
        res.policy_buffer.copy_to_host_async(res.pinned_policy.get(), policy_count, stream_.get());

        // End capture and instantiate
        cudaGraph_t graph = stream_.end_capture();
        res.graph_exec = CudaGraphExec(graph);

        std::println("  Captured CUDA graph for bucket {}", batch_size);
    }

    // Fill a pinned int32 buffer with deterministic in-range gather indices.
    // Content doesn't matter for benchmarking — only that every value is in
    // [0, POLICY_SIZE), so the GPU gather doesn't trip on out-of-range loads.
    static void fill_random_legal_indices(std::int32_t* buf, std::size_t count) noexcept {
        std::mt19937 rng(0xCA7CA7);
        std::uniform_int_distribution<std::int32_t> dist(0, POLICY_SIZE - 1);
        for (std::size_t i = 0; i < count; ++i) buf[i] = dist(rng);
    }

public:
    // Benchmark inference at a specific batch size
    struct BenchmarkResult {
        int32_t batch_size;
        double avg_latency_ms;
        double throughput_samples_per_sec;
        double min_latency_ms;
        double max_latency_ms;
    };

    BenchmarkResult benchmark(int32_t batch_size, int warmup_iters = 10, int bench_iters = 100) {
        // Generate random input
        std::vector<int32_t> input(batch_size * SEQ_LENGTH);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, VOCAB_SIZE - 1);
        std::ranges::generate(input, [&]() { return dist(rng); });

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            infer(input, batch_size);
        }

        // Benchmark
        std::vector<double> latencies;
        latencies.reserve(bench_iters);

        for (int i = 0; i < bench_iters; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            infer(input, batch_size);
            auto end = std::chrono::high_resolution_clock::now();

            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(ms);
        }

        // Compute statistics
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double avg = sum / latencies.size();
        double min_lat = *std::ranges::min_element(latencies);
        double max_lat = *std::ranges::max_element(latencies);

        return BenchmarkResult{
            .batch_size = batch_size,
            .avg_latency_ms = avg,
            .throughput_samples_per_sec = (batch_size * 1000.0) / avg,
            .min_latency_ms = min_lat,
            .max_latency_ms = max_lat,
        };
    }

    const std::string& input_name() const { return input_name_; }
    const std::string& wdl_output_name() const { return wdl_output_name_; }
    const std::string& value_probs_output_name() const { return value_probs_output_name_; }
    const std::string& policy_output_name() const { return policy_output_name_; }

    // Pipelined inference result
    struct PipelinedResult {
        std::vector<float> wdl_logits;
        std::vector<float> value_probs;
        std::vector<float> policies;
    };

    // Process multiple batches with double buffering for maximum throughput
    // Returns all outputs concatenated
    PipelinedResult infer_pipelined(
        const std::vector<std::span<const int32_t>>& batches,
        int32_t batch_size) {
        if (batches.empty()) return {};

        auto it = double_buffer_resources_.find(batch_size);
        if (it == double_buffer_resources_.end()) {
            throw std::runtime_error("Batch size not supported for pipelined inference");
        }

        auto& db = it->second;

        PipelinedResult result;
        result.wdl_logits.reserve(batches.size() * batch_size * WDL_NUM_CLASSES);
        result.value_probs.reserve(batches.size() * batch_size * VALUE_NUM_BINS);
        result.policies.reserve(batches.size() * batch_size * MAX_LEGAL_MOVES);

        const size_t input_count = batch_size * SEQ_LENGTH;
        const size_t value_probs_count = batch_size * VALUE_NUM_BINS;
        const size_t policy_count = batch_size * MAX_LEGAL_MOVES;

        // Lazily capture per-slot CUDA graphs on first call for this batch size.
        if (!db.buffers[0].graph_exec.valid()) {
            capture_double_buffer_graph(batch_size, db);
        }

        // Process batches with double buffering
        for (size_t i = 0; i < batches.size(); ++i) {
            int slot = i % NUM_BUFFERS;
            auto& buf = db.buffers[slot];

            // Wait for this slot's previous work to complete before reusing buffers
            if (i >= NUM_BUFFERS) {
                buf.done_event.synchronize();
                // Collect output from the batch that was 2 iterations ago
                size_t out_idx = i - NUM_BUFFERS;
                int out_slot = out_idx % NUM_BUFFERS;
                auto& out_buf = db.buffers[out_slot];
                result.wdl_logits.insert(result.wdl_logits.end(),
                                         out_buf.pinned_wdl.get(),
                                         out_buf.pinned_wdl.get() + batch_size * WDL_NUM_CLASSES);
                result.value_probs.insert(result.value_probs.end(),
                                          out_buf.pinned_value_probs.get(),
                                          out_buf.pinned_value_probs.get() + value_probs_count);
                result.policies.insert(result.policies.end(),
                                       out_buf.pinned_policy.get(),
                                       out_buf.pinned_policy.get() + policy_count);
            }

            // Copy input to pinned buffer (CPU-side, not part of graph)
            buf.pinned_input.copy_from(batches[i].data(), input_count);

            // Launch the captured pipeline graph: H2D -> compute -> D2H
            buf.graph_exec.launch(buf.stream.get());

            // Record completion event
            buf.done_event.record(buf.stream.get());
        }

        // Collect remaining outputs
        for (size_t i = std::max(size_t(0), batches.size() > NUM_BUFFERS ? batches.size() - NUM_BUFFERS : 0);
             i < batches.size(); ++i) {
            int slot = i % NUM_BUFFERS;
            auto& buf = db.buffers[slot];
            buf.done_event.synchronize();
            result.wdl_logits.insert(result.wdl_logits.end(),
                                     buf.pinned_wdl.get(),
                                     buf.pinned_wdl.get() + batch_size * WDL_NUM_CLASSES);
            result.value_probs.insert(result.value_probs.end(),
                                      buf.pinned_value_probs.get(),
                                      buf.pinned_value_probs.get() + value_probs_count);
            result.policies.insert(result.policies.end(),
                                   buf.pinned_policy.get(),
                                   buf.pinned_policy.get() + policy_count);
        }

        return result;
    }

    // Benchmark pipelined inference (sustained throughput)
    BenchmarkResult benchmark_pipelined(int32_t batch_size, int num_batches = 100,
                                         int num_warmup = 20) {
        // Generate random batches
        std::vector<std::vector<int32_t>> batch_data(num_batches + num_warmup);
        std::vector<std::span<const int32_t>> batches;
        batches.reserve(batch_data.size());

        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, VOCAB_SIZE - 1);

        for (auto& data : batch_data) {
            data.resize(batch_size * SEQ_LENGTH);
            std::ranges::generate(data, [&]() { return dist(rng); });
            batches.emplace_back(data);
        }

        // Warmup
        std::vector<std::span<const int32_t>> warmup_span(
            batches.begin(), batches.begin() + num_warmup);
        infer_pipelined(warmup_span, batch_size);

        // Benchmark
        std::vector<std::span<const int32_t>> bench_batches(
            batches.begin() + num_warmup, batches.end());

        auto start = std::chrono::high_resolution_clock::now();
        auto outputs = infer_pipelined(bench_batches, batch_size);
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / num_batches;
        int64_t total_samples = static_cast<int64_t>(num_batches) * batch_size;

        return BenchmarkResult{
            .batch_size = batch_size,
            .avg_latency_ms = avg_ms,
            .throughput_samples_per_sec = (total_samples * 1000.0) / total_ms,
            .min_latency_ms = avg_ms,  // Can't measure per-batch in pipelined mode
            .max_latency_ms = avg_ms,
        };
    }

private:
    // Double buffer resources per batch size.
    // Each slot owns its own IExecutionContext and CUDA graph so the two streams
    // can run truly concurrently (separate activation memory, no host-side context
    // state contention between launches).
    struct DoubleBufferSlot {
        CudaStream stream;
        CudaEvent done_event;
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        CudaGraphExec graph_exec;
        CudaBuffer<int32_t> input_buffer;
        CudaBuffer<int32_t> legal_indices_buffer;
        CudaBuffer<float> wdl_buffer;
        CudaBuffer<float> value_probs_buffer;
        CudaBuffer<float> policy_buffer;
        PinnedBuffer<int32_t> pinned_input;
        PinnedBuffer<int32_t> pinned_legal_indices;
        PinnedBuffer<float> pinned_wdl;
        PinnedBuffer<float> pinned_value_probs;
        PinnedBuffer<float> pinned_policy;

        DoubleBufferSlot(int32_t batch_size)
            : input_buffer(batch_size * SEQ_LENGTH),
              legal_indices_buffer(batch_size * MAX_LEGAL_MOVES),
              wdl_buffer(batch_size * WDL_NUM_CLASSES),
              value_probs_buffer(batch_size * VALUE_NUM_BINS),
              policy_buffer(batch_size * MAX_LEGAL_MOVES),
              pinned_input(batch_size * SEQ_LENGTH),
              pinned_legal_indices(batch_size * MAX_LEGAL_MOVES),
              pinned_wdl(batch_size * WDL_NUM_CLASSES),
              pinned_value_probs(batch_size * VALUE_NUM_BINS),
              pinned_policy(batch_size * MAX_LEGAL_MOVES) {}
    };

    struct DoubleBufferResources {
        std::array<DoubleBufferSlot, NUM_BUFFERS> buffers;

        DoubleBufferResources(int32_t batch_size)
            : buffers{DoubleBufferSlot(batch_size), DoubleBufferSlot(batch_size)} {}
    };

    void setup_double_buffering() {
        size_t allocated = 0;
        for (int32_t batch_size : CACHED_BATCH_SIZES) {
            if (!has_bucket(batch_size)) continue;
            auto [it, _] = double_buffer_resources_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(batch_size),
                std::forward_as_tuple(batch_size));
            create_slot_contexts(it->second, batch_size);
            ++allocated;
        }
        std::println("  Set up double buffering for {} batch sizes", allocated);
    }

    // Create the cached-batch context from this bucket's sub-engine and bind
    // it to profile 0 (the only profile in a per-bucket sub-engine). Also
    // seeds the pinned legal_indices buffer once (random in-range int32) so
    // the captured graph's H2D always reads valid gather indices.
    void create_cached_context(CachedBatchResources& res, int32_t bucket_size) {
        if (res.context) return;
        res.context.reset(engines_.at(bucket_size)->createExecutionContext());
        if (!res.context) {
            throw std::runtime_error(std::format(
                "Failed to create cached IExecutionContext for bucket {}", bucket_size));
        }
        bind_profile_zero(*res.context, stream_);
        fill_random_legal_indices(res.pinned_legal_indices.get(),
                                  static_cast<std::size_t>(bucket_size) * MAX_LEGAL_MOVES);
    }

    // Create a dedicated IExecutionContext for each slot in `db` from this
    // bucket's sub-engine. Separate contexts give each stream its own
    // activation memory, enabling true concurrent execution of the two
    // pipelined inferences. Each slot's legal_indices pinned buffer is seeded
    // with random in-range int32 (deterministic) at creation.
    void create_slot_contexts(DoubleBufferResources& db, int32_t bucket_size) {
        for (auto& slot : db.buffers) {
            if (slot.context) continue;  // Already created
            slot.context.reset(engines_.at(bucket_size)->createExecutionContext());
            if (!slot.context) {
                throw std::runtime_error(std::format(
                    "Failed to create slot IExecutionContext for bucket {}", bucket_size));
            }
            bind_profile_zero(*slot.context, slot.stream);
            fill_random_legal_indices(slot.pinned_legal_indices.get(),
                                      static_cast<std::size_t>(bucket_size) * MAX_LEGAL_MOVES);
        }
    }

    // Each per-bucket sub-engine has exactly one profile (index 0). Bind it
    // and synchronize so it's in effect before any subsequent
    // setTensorAddress / capture.
    static void bind_profile_zero(nvinfer1::IExecutionContext& ctx, CudaStream& stream) {
        if (!ctx.setOptimizationProfileAsync(0, stream.get())) {
            throw std::runtime_error("Failed to set optimization profile 0 on context");
        }
        stream.synchronize();
    }

    // Capture a per-slot CUDA graph that wraps H2D + enqueueV3 + D2H.
    // Each slot uses its own context + buffers so the two graphs share no
    // mutable state and can be launched concurrently on separate streams.
    void capture_double_buffer_graph(int32_t batch_size, DoubleBufferResources& db) {
        const size_t input_count       = batch_size * SEQ_LENGTH;
        const size_t legal_in_count    = batch_size * MAX_LEGAL_MOVES;
        const size_t value_probs_count = batch_size * VALUE_NUM_BINS;
        const size_t policy_count      = batch_size * MAX_LEGAL_MOVES;

        nvinfer1::Dims tokens_dims;
        tokens_dims.nbDims = 2;
        tokens_dims.d[0] = batch_size;
        tokens_dims.d[1] = SEQ_LENGTH;

        nvinfer1::Dims legal_dims;
        legal_dims.nbDims = 2;
        legal_dims.d[0] = batch_size;
        legal_dims.d[1] = MAX_LEGAL_MOVES;

        for (auto& slot : db.buffers) {
            // Bind tensor addresses & shape on this slot's context (host-side state).
            // These persist on the context so the captured kernels reference the
            // correct device pointers.
            slot.context->setTensorAddress(input_name_.c_str(), slot.input_buffer.get());
            slot.context->setTensorAddress(legal_indices_input_name_.c_str(),
                                           slot.legal_indices_buffer.get());
            slot.context->setTensorAddress(wdl_output_name_.c_str(), slot.wdl_buffer.get());
            if (!value_probs_output_name_.empty()) {
                slot.context->setTensorAddress(value_probs_output_name_.c_str(),
                                               slot.value_probs_buffer.get());
            }
            slot.context->setTensorAddress(policy_output_name_.c_str(), slot.policy_buffer.get());
            slot.context->setInputShape(input_name_.c_str(), tokens_dims);
            slot.context->setInputShape(legal_indices_input_name_.c_str(), legal_dims);

            // Capture: H2D -> compute -> D2H, all on this slot's stream.
            slot.stream.begin_capture();

            slot.input_buffer.copy_from_host_async(slot.pinned_input.get(), input_count, slot.stream.get());
            slot.legal_indices_buffer.copy_from_host_async(
                slot.pinned_legal_indices.get(), legal_in_count, slot.stream.get());

            if (!slot.context->enqueueV3(slot.stream.get())) {
                cudaGraph_t dummy;
                cudaStreamEndCapture(slot.stream.get(), &dummy);  // Clean up capture state
                throw std::runtime_error("TensorRT inference failed during pipelined graph capture");
            }

            slot.wdl_buffer.copy_to_host_async(slot.pinned_wdl.get(),
                                               batch_size * WDL_NUM_CLASSES, slot.stream.get());
            if (!value_probs_output_name_.empty()) {
                slot.value_probs_buffer.copy_to_host_async(slot.pinned_value_probs.get(),
                                                           value_probs_count, slot.stream.get());
            }
            slot.policy_buffer.copy_to_host_async(slot.pinned_policy.get(), policy_count, slot.stream.get());

            cudaGraph_t graph = slot.stream.end_capture();
            slot.graph_exec = CudaGraphExec(graph);
        }

        std::println("  Captured pipelined CUDA graphs for batch size {} ({} slots, separate contexts)",
                     batch_size, NUM_BUFFERS);
    }

    // Pre-allocate resources for every bucket present in the .network.
    void preallocate_cached_buffers() {
        size_t allocated = 0;
        for (int32_t batch_size : CACHED_BATCH_SIZES) {
            if (!has_bucket(batch_size)) continue;
            auto [it, _] = cached_resources_.emplace(batch_size, CachedBatchResources(batch_size));
            create_cached_context(it->second, batch_size);
            ++allocated;
        }
        std::println("  Pre-allocated buffers for {} cached buckets (<= max {})",
                     allocated, max_batch_size_);
    }

    // Load a packed .network file: deserialize one ICudaEngine per bucket.
    void load_engines(const fs::path& network_path) {
        catgpt::NetworkFile file(network_path);
        std::println("Loaded network file: {} ({:.1f} MB, {} sub-engine(s))",
                     network_path.string(),
                     file.file_size() / (1024.0 * 1024.0),
                     file.sub_engines().size());

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engines_.clear();
        for (const auto& sub : file.sub_engines()) {
            std::unique_ptr<nvinfer1::ICudaEngine> engine(
                runtime_->deserializeCudaEngine(sub.blob.data(), sub.blob.size()));
            if (!engine) {
                throw std::runtime_error(std::format(
                    "Failed to deserialize sub-engine for bucket {}", sub.bucket_size));
            }
            std::println("  Bucket {:>3}: {:.1f} MB",
                         sub.bucket_size,
                         sub.blob.size() / (1024.0 * 1024.0));
            engines_.emplace(sub.bucket_size, std::move(engine));
        }
    }

    void setup_io() {
        // IO tensor names are identical across all sub-engines (same source
        // ONNX). Pick the smallest-bucket engine to inspect.
        nvinfer1::ICudaEngine& engine = *engines_.begin()->second;
        int num_io = engine.getNbIOTensors();
        std::println("Engine has {} I/O tensors:", num_io);

        for (int i = 0; i < num_io; ++i) {
            const char* name = engine.getIOTensorName(i);
            auto mode = engine.getTensorIOMode(name);
            auto dims = engine.getTensorShape(name);
            auto dtype = engine.getTensorDataType(name);

            std::string dims_str = "(";
            for (int d = 0; d < dims.nbDims; ++d) {
                if (d > 0) dims_str += ", ";
                dims_str += (dims.d[d] == -1) ? "?" : std::to_string(dims.d[d]);
            }
            dims_str += ")";

            const char* dtype_str = dtype == nvinfer1::DataType::kFLOAT  ? "float32"
                                    : dtype == nvinfer1::DataType::kINT32 ? "int32"
                                    : dtype == nvinfer1::DataType::kHALF  ? "float16"
                                                                          : "other";

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                std::println("  Input '{}': {} {}", name, dims_str, dtype_str);
                // Two inputs: tokens (batch, SEQ_LENGTH) and legal_indices
                // (batch, MAX_LEGAL_MOVES). jax2onnx names them positionally
                // (e.g. "in_0", "in_1") so we disambiguate by trailing dim.
                if (dims.nbDims == 2 && dims.d[1] == SEQ_LENGTH) {
                    input_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == MAX_LEGAL_MOVES) {
                    legal_indices_input_name_ = name;
                } else {
                    throw std::runtime_error(std::format(
                        "Unexpected input tensor '{}' with shape {} {}; "
                        "expected (batch, {}) tokens or (batch, {}) legal_indices.",
                        name, dims_str, dtype_str, SEQ_LENGTH, MAX_LEGAL_MOVES));
                }
                continue;
            }

            std::string name_str(name);
            std::println("  Output '{}': {} {}", name, dims_str, dtype_str);

            // Try to match by name first (order matters: check specific names
            // before generic). The gather-aware export emits
            // `optimistic_policy_legal_logit` so a "policy" substring catches
            // it. The bestq distribution and WDL logits are name-detected
            // first; ambiguous fallbacks land on shape-based matching.
            if (name_str.find("bestq_probs") != std::string::npos ||
                name_str.find("value_probs") != std::string::npos) {
                value_probs_output_name_ = name;
            } else if (name_str.find("wdl_logit") != std::string::npos ||
                       name_str.find("wdl") != std::string::npos) {
                wdl_output_name_ = name;
            } else if (name_str.find("policy") != std::string::npos ||
                       name_str.find("Policy") != std::string::npos) {
                policy_output_name_ = name;
            } else {
                // Fallback: detect by shape
                if (dims.nbDims == 2 && dims.d[1] == WDL_NUM_CLASSES) {
                    if (wdl_output_name_.empty()) wdl_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == VALUE_NUM_BINS) {
                    if (value_probs_output_name_.empty()) value_probs_output_name_ = name;
                } else if (dims.nbDims == 2 && dims.d[1] == MAX_LEGAL_MOVES) {
                    if (policy_output_name_.empty()) policy_output_name_ = name;
                }
            }
        }

        if (input_name_.empty()) {
            throw std::runtime_error(
                "Could not find tokens input tensor (batch, 64) int32");
        }
        if (legal_indices_input_name_.empty()) {
            throw std::runtime_error(std::format(
                "Could not find legal_indices input tensor (batch, {}) int32. "
                "Re-export ONNX with the gather-aware export-onnx.sh.",
                MAX_LEGAL_MOVES));
        }
        if (wdl_output_name_.empty()) {
            throw std::runtime_error("Could not find WDL output tensor (wdl_logit)");
        }
        if (policy_output_name_.empty()) {
            throw std::runtime_error(
                "Could not find policy output tensor (optimistic_policy_legal_logit)");
        }

        std::println("  -> Tokens input:        '{}'", input_name_);
        std::println("  -> Legal-indices input: '{}'", legal_indices_input_name_);
        std::println("  -> WDL output:          '{}'", wdl_output_name_);
        std::println("  -> Value probs output:  '{}'",
                     value_probs_output_name_.empty() ? "(not found)" : value_probs_output_name_);
        std::println("  -> Policy output:       '{}'", policy_output_name_);
    }

    // Build the buckets_ list from loaded sub-engines and validate that each
    // engine has exactly one profile pinned at min == opt == max == bucket.
    void discover_buckets() {
        max_batch_size_ = 0;
        buckets_.clear();
        buckets_.reserve(engines_.size());

        // Collect bucket sizes in ascending order.
        std::vector<int32_t> ordered;
        ordered.reserve(engines_.size());
        for (const auto& [b, _] : engines_) ordered.push_back(b);
        std::ranges::sort(ordered);

        std::println("  .network has {} bucket(s):", ordered.size());
        for (int32_t b : ordered) {
            auto& engine = *engines_.at(b);
            const int n = engine.getNbOptimizationProfiles();
            if (n != 1) {
                throw std::runtime_error(std::format(
                    "Sub-engine for bucket {} has {} optimization profile(s); "
                    "expected exactly 1 (rebuild with scripts/trt.sh).", b, n));
            }
            // Validate batch (d[0]) on both inputs; trailing dims are baked in
            // and validated when setInputShape is called against opt at capture.
            for (const char* in_name : {input_name_.c_str(),
                                        legal_indices_input_name_.c_str()}) {
                auto min_dims = engine.getProfileShape(
                    in_name, 0, nvinfer1::OptProfileSelector::kMIN);
                auto opt_dims = engine.getProfileShape(
                    in_name, 0, nvinfer1::OptProfileSelector::kOPT);
                auto max_dims = engine.getProfileShape(
                    in_name, 0, nvinfer1::OptProfileSelector::kMAX);
                if (opt_dims.d[0] != b || min_dims.d[0] != b || max_dims.d[0] != b) {
                    throw std::runtime_error(std::format(
                        "Sub-engine for bucket {} input '{}' has profile (min={}, "
                        "opt={}, max={}); expected all == {}.",
                        b, in_name, min_dims.d[0], opt_dims.d[0], max_dims.d[0], b));
                }
            }
            // Print using the tokens-input opt for the human-readable summary.
            auto opt_tokens = engine.getProfileShape(
                input_name_.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
            buckets_.push_back(BucketInfo{b});
            max_batch_size_ = std::max(max_batch_size_, b);
            std::println("    Bucket {:>3} (opt batch={})", b, opt_tokens.d[0]);
        }
        std::println("  Overall max batch size: {}", max_batch_size_);
    }

    Logger& logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;

    // bucket_size -> ICudaEngine. One sub-engine per bucket, deserialized
    // from a packed .network file.
    std::unordered_map<int32_t, std::unique_ptr<nvinfer1::ICudaEngine>> engines_;

    // Input tensor names (auto-detected by shape in setup_io). The model has
    // two inputs: tokens (batch, SEQ_LENGTH) and legal_indices (batch,
    // MAX_LEGAL_MOVES); jax2onnx names them positionally so we disambiguate
    // by trailing dim.
    std::string input_name_;
    std::string legal_indices_input_name_;
    std::string wdl_output_name_;
    std::string value_probs_output_name_;
    std::string policy_output_name_;

    std::vector<BucketInfo> buckets_;  // Buckets present in the .network, ascending
    int32_t max_batch_size_ = 1;  // Largest bucket size

    CudaStream stream_;  // Stream used for single-mode capture/launch (graphs are stream-agnostic)

    // Cached resources per batch size (with CUDA graphs and per-batch IExecutionContext)
    std::unordered_map<int32_t, CachedBatchResources> cached_resources_;

    // Double buffer resources per batch size (for pipelined inference)
    std::unordered_map<int32_t, DoubleBufferResources> double_buffer_resources_;
};

void print_header() {
    std::println("╔════════════════════════════════════════════════════════════════╗");
    std::println("║         CatGPT Chess Engine - TensorRT Benchmark               ║");
    std::println("╚════════════════════════════════════════════════════════════════╝");
    std::println("Version: {}  commit: {}{}  branch: {}",
                 catgpt::version::GIT_DESCRIBE,
                 catgpt::version::GIT_HASH_SHORT,
                 catgpt::version::GIT_DIRTY ? " (dirty)" : "",
                 catgpt::version::GIT_BRANCH);
}

void print_cuda_info() {
    int device;
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::println("\n┌─ CUDA Device ─────────────────────────────────────────────────┐");
    std::println("│ Device {}: {}", device, props.name);
    std::println("│ Compute: {}.{}, Memory: {:.1f} GB, SMs: {}",
                 props.major, props.minor,
                 props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                 props.multiProcessorCount);
    std::println("└───────────────────────────────────────────────────────────────┘");
}

void print_usage(const char* prog) {
    std::println("Usage: {} [NETWORK_PATH] [OPTIONS]", prog);
    std::println("");
    std::println("Arguments:");
    std::println("  NETWORK_PATH        Path to a packed .network file (multi-engine bundle).");
    std::println("                      (default: /home/shadeform/CatGPT/S4.network)");
    std::println("");
    std::println("Options:");
    std::println("  -b, --batch N       Benchmark only batch size N (single + pipelined).");
    std::println("                      Must be one of the bucket sizes baked into the");
    std::println("                      .network file. Skips the default sweep.");
    std::println("  -h, --help          Show this help message.");
}

int main(int argc, char* argv[]) {
    print_header();
    print_cuda_info();

    fs::path engine_path = "/home/shadeform/CatGPT/S4.network";
    int32_t target_batch_size = 0;  // 0 = default sweep

    // Simple positional + flag parsing
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "-b" || arg == "--batch") {
            if (i + 1 >= argc) {
                std::println(stderr, "Error: {} requires an argument", arg);
                return 1;
            }
            target_batch_size = std::stoi(argv[++i]);
            if (target_batch_size <= 0) {
                std::println(stderr, "Error: batch size must be positive");
                return 1;
            }
            continue;
        }
        if (arg.starts_with("-")) {
            std::println(stderr, "Error: unknown option '{}'", arg);
            print_usage(argv[0]);
            return 1;
        }
        // Positional: engine path
        engine_path = arg;
    }

    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: .network file not found: {}", engine_path.string());
        return 1;
    }

    try {
        Logger logger;
        std::println("\n┌─ Loading .network ───────────────────────────────────────────┐");
        TrtEngine engine(engine_path, logger);
        std::println("└───────────────────────────────────────────────────────────────┘");

        // Quick inference test
        std::println("\n┌─ Inference Test ──────────────────────────────────────────────┐");
        {
            std::vector<int32_t> test_input(TrtEngine::SEQ_LENGTH, 1);  // Simple test input
            auto result = engine.infer(test_input, 1);
            std::println("│ Single inference test:");
            std::println("│   Value (Q from WDL): {:.6f}",
                         wdl_logits_to_q(result.wdl_logits.data()));

            // Show value distribution summary
            if (!result.value_probs.empty()) {
                float max_prob = *std::max_element(result.value_probs.begin(),
                                                   result.value_probs.begin() + VALUE_NUM_BINS);
                int mode_bin = static_cast<int>(std::distance(
                    result.value_probs.begin(),
                    std::max_element(result.value_probs.begin(),
                                     result.value_probs.begin() + VALUE_NUM_BINS)));
                float mode_value = static_cast<float>(mode_bin) / (VALUE_NUM_BINS - 1);
                std::println("│   BestQ dist: mode bin={} (v={:.3f}, p={:.4f})",
                             mode_bin, mode_value, max_prob);
            }

            // Show top-5 gathered policy logits. The benchmark feeds random
            // in-range gather indices, so positions in `policy_logits` don't
            // correspond to real (from, to) squares — only the values'
            // distribution is meaningful here.
            auto& policy = result.policy_logits;
            std::vector<std::pair<int, float>> indexed_logits;
            indexed_logits.reserve(policy.size());
            for (size_t i = 0; i < policy.size(); ++i) {
                indexed_logits.emplace_back(static_cast<int>(i), policy[i]);
            }
            std::partial_sort(indexed_logits.begin(), indexed_logits.begin() + 5, indexed_logits.end(),
                              [](const auto& a, const auto& b) { return a.second > b.second; });

            std::println("│   Top-5 gathered policy logits (random indices, values only):");
            for (int i = 0; i < 5; ++i) {
                std::println("│     [{}] slot={}: {:.4f}",
                             i + 1, indexed_logits[i].first, indexed_logits[i].second);
            }
        }
        std::println("└───────────────────────────────────────────────────────────────┘");

        // Choose batch sizes: either a single targeted bucket, or a default
        // sweep of every bucket present in the .network. register_batch_size
        // throws with the available-buckets list if the user picks something
        // that isn't a bucket.
        std::vector<int32_t> batch_sizes;
        if (target_batch_size > 0) {
            engine.register_batch_size(target_batch_size);
            batch_sizes.push_back(target_batch_size);
        } else {
            for (int32_t bs : TrtEngine::CACHED_BATCH_SIZES) {
                if (engine.has_bucket(bs)) batch_sizes.push_back(bs);
            }
        }

        std::println("\n┌─ Benchmark Results ───────────────────────────────────────────┐");
        std::println("│ {:>6} │ {:>10} │ {:>12} │ {:>10} │ {:>10} │",
                     "Batch", "Avg (ms)", "Throughput", "Min (ms)", "Max (ms)");
        std::println("│────────┼────────────┼──────────────┼────────────┼────────────│");

        for (int32_t batch_size : batch_sizes) {
            auto result = engine.benchmark(batch_size, /*warmup=*/10, /*iters=*/100);
            std::println("│ {:>6} │ {:>10.3f} │ {:>10.0f}/s │ {:>10.3f} │ {:>10.3f} │",
                         result.batch_size,
                         result.avg_latency_ms,
                         result.throughput_samples_per_sec,
                         result.min_latency_ms,
                         result.max_latency_ms);
        }
        std::println("└───────────────────────────────────────────────────────────────┘");

        // Find optimal batch size (highest throughput)
        double best_throughput = 0;
        int32_t best_batch = 1;
        for (int32_t batch_size : batch_sizes) {
            auto result = engine.benchmark(batch_size, 5, 20);
            if (result.throughput_samples_per_sec > best_throughput) {
                best_throughput = result.throughput_samples_per_sec;
                best_batch = batch_size;
            }
        }

        std::println("\n★ Optimal batch size (single): {} ({:.0f} samples/sec)", best_batch, best_throughput);

        // Pipelined benchmark (double buffering for sustained throughput)
        std::println("\n┌─ Pipelined Benchmark (Double Buffering) ─────────────────────┐");
        std::println("│ {:>6} │ {:>10} │ {:>14} │ {:>12} │",
                     "Batch", "Avg (ms)", "Throughput", "Speedup");
        std::println("│────────┼────────────┼────────────────┼──────────────│");

        double best_pipelined = 0;
        int32_t best_pipelined_batch = 1;
        for (int32_t batch_size : batch_sizes) {
            auto single_result = engine.benchmark(batch_size, 5, 20);
            auto pipelined_result = engine.benchmark_pipelined(batch_size, /*num_batches=*/100,
                                                                /*warmup=*/20);
            double speedup = pipelined_result.throughput_samples_per_sec /
                             single_result.throughput_samples_per_sec;
            std::println("│ {:>6} │ {:>10.3f} │ {:>12.0f}/s │ {:>10.2f}x │",
                         pipelined_result.batch_size,
                         pipelined_result.avg_latency_ms,
                         pipelined_result.throughput_samples_per_sec,
                         speedup);

            if (pipelined_result.throughput_samples_per_sec > best_pipelined) {
                best_pipelined = pipelined_result.throughput_samples_per_sec;
                best_pipelined_batch = batch_size;
            }
        }
        std::println("└───────────────────────────────────────────────────────────────┘");

        std::println("\n★ Optimal batch size (pipelined): {} ({:.0f} samples/sec)",
                     best_pipelined_batch, best_pipelined);

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
