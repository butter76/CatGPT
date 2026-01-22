/**
 * TensorRT Benchmark for CatGPT Chess Engine
 *
 * Loads a TensorRT engine and benchmarks inference at various batch sizes.
 * The model expects:
 *   - Input: int32 tensor of shape (batch, 64) - chess position tokens
 *   - Output "value": float32 tensor of shape (batch,) - win probability
 *   - Output "policy_logit": float32 tensor of shape (batch, 4672) - policy logits
 */

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

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

// Inference result containing both value and policy outputs
struct InferenceResult {
    std::vector<float> values;          // (batch,) - win probabilities
    std::vector<float> policy_logits;   // (batch * POLICY_SIZE) - policy logits
};

// TensorRT Engine wrapper with CUDA Graph caching and double buffering
class TrtEngine {
public:
    static constexpr int32_t SEQ_LENGTH = 64;
    static constexpr int32_t VOCAB_SIZE = 28;
    static constexpr int32_t POLICY_SIZE = 4672;  // 64 * 73 (from_sq * to_sq)
    static constexpr int NUM_BUFFERS = 2;  // Double buffering

    // Batch sizes to cache CUDA graphs for
    static constexpr std::array<int32_t, 10> CACHED_BATCH_SIZES = {1, 2, 3, 4, 8, 16, 32, 64, 128, 256};

    TrtEngine(const fs::path& engine_path, Logger& logger) : logger_(logger) {
        load_engine(engine_path);
        setup_io();
        preallocate_cached_buffers();
        setup_double_buffering();
    }

    ~TrtEngine() = default;

    // Run inference on a batch of chess positions
    // Input: vector of int32 tokens, size = batch_size * SEQ_LENGTH
    // Output: InferenceResult with values and policy logits
    InferenceResult infer(std::span<const int32_t> input_tokens, int32_t batch_size) {
        if (static_cast<size_t>(batch_size * SEQ_LENGTH) != input_tokens.size()) {
            throw std::runtime_error("Input size mismatch");
        }

        // Check if this batch size has cached resources
        auto it = cached_resources_.find(batch_size);
        if (it != cached_resources_.end()) {
            return infer_with_graph(input_tokens, batch_size, it->second);
        }

        // Fallback to non-graph path for uncached batch sizes
        return infer_no_graph(input_tokens, batch_size);
    }

private:
    // Per-batch-size cached resources
    struct CachedBatchResources {
        CudaBuffer<int32_t> input_buffer;
        CudaBuffer<float> value_buffer;
        CudaBuffer<float> policy_buffer;
        PinnedBuffer<int32_t> pinned_input;
        PinnedBuffer<float> pinned_value;
        PinnedBuffer<float> pinned_policy;
        CudaGraphExec graph_exec;

        CachedBatchResources(int32_t batch_size)
            : input_buffer(batch_size * SEQ_LENGTH),
              value_buffer(batch_size),
              policy_buffer(batch_size * POLICY_SIZE),
              pinned_input(batch_size * SEQ_LENGTH),
              pinned_value(batch_size),
              pinned_policy(batch_size * POLICY_SIZE) {}
    };

    // Inference using cached CUDA graph
    InferenceResult infer_with_graph(std::span<const int32_t> input_tokens,
                                     int32_t batch_size,
                                     CachedBatchResources& res) {
        // Copy input to pinned buffer (CPU-side, not part of graph)
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
        result.values.resize(batch_size);
        result.policy_logits.resize(batch_size * POLICY_SIZE);
        std::memcpy(result.values.data(), res.pinned_value.get(), batch_size * sizeof(float));
        std::memcpy(result.policy_logits.data(), res.pinned_policy.get(),
                    batch_size * POLICY_SIZE * sizeof(float));

        return result;
    }

    // Capture CUDA graph for a batch size
    void capture_graph(int32_t batch_size, CachedBatchResources& res) {
        // Set up tensor addresses and shapes before capture
        context_->setTensorAddress(input_name_.c_str(), res.input_buffer.get());
        context_->setTensorAddress(value_output_name_.c_str(), res.value_buffer.get());
        context_->setTensorAddress(policy_output_name_.c_str(), res.policy_buffer.get());

        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), input_dims);

        size_t input_count = batch_size * SEQ_LENGTH;
        size_t policy_count = batch_size * POLICY_SIZE;

        // Begin graph capture
        stream_.begin_capture();

        // H2D transfer (async, captured)
        res.input_buffer.copy_from_host_async(res.pinned_input.get(), input_count, stream_.get());

        // TensorRT inference (captured)
        if (!context_->enqueueV3(stream_.get())) {
            cudaGraph_t dummy;
            cudaStreamEndCapture(stream_.get(), &dummy);  // Clean up capture state
            throw std::runtime_error("TensorRT inference failed during graph capture");
        }

        // D2H transfer (async, captured)
        res.value_buffer.copy_to_host_async(res.pinned_value.get(), batch_size, stream_.get());
        res.policy_buffer.copy_to_host_async(res.pinned_policy.get(), policy_count, stream_.get());

        // End capture and instantiate
        cudaGraph_t graph = stream_.end_capture();
        res.graph_exec = CudaGraphExec(graph);

        std::println("  Captured CUDA graph for batch size {}", batch_size);
    }

    // Fallback inference without graph (for uncached batch sizes)
    InferenceResult infer_no_graph(std::span<const int32_t> input_tokens, int32_t batch_size) {
        // Resize GPU and pinned buffers if needed
        ensure_fallback_buffers(batch_size);

        // Copy input to pinned buffer, then async H2D transfer
        fallback_pinned_input_.copy_from(input_tokens.data(), input_tokens.size());
        fallback_input_buffer_.copy_from_host_async(
            fallback_pinned_input_.get(), input_tokens.size(), stream_.get());

        // Set tensor addresses
        context_->setTensorAddress(input_name_.c_str(), fallback_input_buffer_.get());
        context_->setTensorAddress(value_output_name_.c_str(), fallback_value_buffer_.get());
        context_->setTensorAddress(policy_output_name_.c_str(), fallback_policy_buffer_.get());

        // Set input shape (dynamic batch)
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), input_dims);

        // Execute on dedicated stream
        if (!context_->enqueueV3(stream_.get())) {
            throw std::runtime_error("TensorRT inference failed");
        }

        // Async D2H transfer, then synchronize
        size_t policy_count = batch_size * POLICY_SIZE;
        fallback_value_buffer_.copy_to_host_async(
            fallback_pinned_value_.get(), batch_size, stream_.get());
        fallback_policy_buffer_.copy_to_host_async(
            fallback_pinned_policy_.get(), policy_count, stream_.get());
        stream_.synchronize();

        // Copy from pinned buffers to output
        InferenceResult result;
        result.values.resize(batch_size);
        result.policy_logits.resize(policy_count);
        std::memcpy(result.values.data(), fallback_pinned_value_.get(), batch_size * sizeof(float));
        std::memcpy(result.policy_logits.data(), fallback_pinned_policy_.get(),
                    policy_count * sizeof(float));

        return result;
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
    const std::string& value_output_name() const { return value_output_name_; }
    const std::string& policy_output_name() const { return policy_output_name_; }

    // Process multiple batches with double buffering for maximum throughput
    // Returns all outputs concatenated
    std::pair<std::vector<float>, std::vector<float>> infer_pipelined(
        const std::vector<std::span<const int32_t>>& batches,
        int32_t batch_size) {
        if (batches.empty()) return {};

        auto it = double_buffer_resources_.find(batch_size);
        if (it == double_buffer_resources_.end()) {
            throw std::runtime_error("Batch size not supported for pipelined inference");
        }

        auto& db = it->second;
        std::vector<float> all_values;
        std::vector<float> all_policies;
        all_values.reserve(batches.size() * batch_size);
        all_policies.reserve(batches.size() * batch_size * POLICY_SIZE);

        const size_t input_count = batch_size * SEQ_LENGTH;
        const size_t policy_count = batch_size * POLICY_SIZE;

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
                all_values.insert(all_values.end(),
                                  out_buf.pinned_value.get(),
                                  out_buf.pinned_value.get() + batch_size);
                all_policies.insert(all_policies.end(),
                                    out_buf.pinned_policy.get(),
                                    out_buf.pinned_policy.get() + policy_count);
            }

            // Copy input to pinned buffer
            buf.pinned_input.copy_from(batches[i].data(), input_count);

            // Launch the pipeline: H2D -> compute -> D2H
            if (db.graph_exec.valid()) {
                // Use CUDA graph for this slot
                db.graph_exec.launch(buf.stream.get());
            } else {
                // Capture graph on first use (using slot 0's buffers)
                if (slot == 0 && !db.graph_captured) {
                    capture_double_buffer_graph(batch_size, db);
                }
                // For now, fall back to non-graph execution
                execute_pipeline_step(batch_size, buf);
            }

            // Record completion event
            buf.done_event.record(buf.stream.get());
        }

        // Collect remaining outputs
        for (size_t i = std::max(size_t(0), batches.size() > NUM_BUFFERS ? batches.size() - NUM_BUFFERS : 0);
             i < batches.size(); ++i) {
            int slot = i % NUM_BUFFERS;
            auto& buf = db.buffers[slot];
            buf.done_event.synchronize();
            all_values.insert(all_values.end(),
                              buf.pinned_value.get(),
                              buf.pinned_value.get() + batch_size);
            all_policies.insert(all_policies.end(),
                                buf.pinned_policy.get(),
                                buf.pinned_policy.get() + policy_count);
        }

        return {std::move(all_values), std::move(all_policies)};
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
    // Double buffer resources per batch size
    struct DoubleBufferSlot {
        CudaStream stream;
        CudaEvent done_event;
        CudaBuffer<int32_t> input_buffer;
        CudaBuffer<float> value_buffer;
        CudaBuffer<float> policy_buffer;
        PinnedBuffer<int32_t> pinned_input;
        PinnedBuffer<float> pinned_value;
        PinnedBuffer<float> pinned_policy;

        DoubleBufferSlot(int32_t batch_size)
            : input_buffer(batch_size * SEQ_LENGTH),
              value_buffer(batch_size),
              policy_buffer(batch_size * POLICY_SIZE),
              pinned_input(batch_size * SEQ_LENGTH),
              pinned_value(batch_size),
              pinned_policy(batch_size * POLICY_SIZE) {}
    };

    struct DoubleBufferResources {
        std::array<DoubleBufferSlot, NUM_BUFFERS> buffers;
        CudaGraphExec graph_exec;
        bool graph_captured = false;

        DoubleBufferResources(int32_t batch_size)
            : buffers{DoubleBufferSlot(batch_size), DoubleBufferSlot(batch_size)} {}
    };

    void setup_double_buffering() {
        std::println("  Setting up double buffering for {} batch sizes...",
                     CACHED_BATCH_SIZES.size());
        for (int32_t batch_size : CACHED_BATCH_SIZES) {
            double_buffer_resources_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(batch_size),
                std::forward_as_tuple(batch_size));
        }
    }

    void execute_pipeline_step(int32_t batch_size, DoubleBufferSlot& buf) {
        size_t input_count = batch_size * SEQ_LENGTH;
        size_t policy_count = batch_size * POLICY_SIZE;

        // H2D transfer
        buf.input_buffer.copy_from_host_async(buf.pinned_input.get(), input_count, buf.stream.get());

        // Set tensor addresses
        context_->setTensorAddress(input_name_.c_str(), buf.input_buffer.get());
        context_->setTensorAddress(value_output_name_.c_str(), buf.value_buffer.get());
        context_->setTensorAddress(policy_output_name_.c_str(), buf.policy_buffer.get());

        // Set input shape
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), input_dims);

        // Execute
        context_->enqueueV3(buf.stream.get());

        // D2H transfer
        buf.value_buffer.copy_to_host_async(buf.pinned_value.get(), batch_size, buf.stream.get());
        buf.policy_buffer.copy_to_host_async(buf.pinned_policy.get(), policy_count, buf.stream.get());
    }

    void capture_double_buffer_graph(int32_t batch_size, DoubleBufferResources& db) {
        // For simplicity, we don't capture graphs for double buffering
        // The overlap comes from having multiple streams
        db.graph_captured = true;  // Mark as "attempted"
        std::println("  Double buffering enabled for batch size {} (no graph capture)", batch_size);
    }

    // Pre-allocate resources for all cached batch sizes
    void preallocate_cached_buffers() {
        std::println("  Pre-allocating buffers for {} cached batch sizes...",
                     CACHED_BATCH_SIZES.size());
        for (int32_t batch_size : CACHED_BATCH_SIZES) {
            cached_resources_.emplace(batch_size, CachedBatchResources(batch_size));
        }
    }

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

        std::println("Loaded engine file: {} ({:.1f} MB)", engine_path.string(),
                     size / (1024.0 * 1024.0));

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
        std::println("Engine has {} I/O tensors:", num_io);

        for (int i = 0; i < num_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode = engine_->getTensorIOMode(name);
            auto dims = engine_->getTensorShape(name);
            auto dtype = engine_->getTensorDataType(name);

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
                input_name_ = name;
                std::println("  Input '{}': {} {}", name, dims_str, dtype_str);
            } else {
                // Determine output type by name or order
                std::string name_str(name);
                std::println("  Output '{}': {} {}", name, dims_str, dtype_str);

                // Match output names - check for "value" or "policy" in the name
                if (name_str.find("value") != std::string::npos ||
                    name_str.find("Value") != std::string::npos) {
                    value_output_name_ = name;
                } else if (name_str.find("policy") != std::string::npos ||
                           name_str.find("Policy") != std::string::npos) {
                    policy_output_name_ = name;
                } else {
                    // Fallback: assign by shape (value has 1D, policy has 2D output)
                    if (dims.nbDims == 1 || (dims.nbDims == 2 && dims.d[1] == 1)) {
                        if (value_output_name_.empty()) {
                            value_output_name_ = name;
                        }
                    } else {
                        if (policy_output_name_.empty()) {
                            policy_output_name_ = name;
                        }
                    }
                }
            }
        }

        if (input_name_.empty()) {
            throw std::runtime_error("Could not find input tensor");
        }
        if (value_output_name_.empty()) {
            throw std::runtime_error("Could not find value output tensor");
        }
        if (policy_output_name_.empty()) {
            throw std::runtime_error("Could not find policy output tensor");
        }

        std::println("  -> Input: '{}'", input_name_);
        std::println("  -> Value output: '{}'", value_output_name_);
        std::println("  -> Policy output: '{}'", policy_output_name_);
    }

    void ensure_fallback_buffers(int32_t batch_size) {
        size_t input_count = batch_size * SEQ_LENGTH;
        size_t value_count = batch_size;
        size_t policy_count = batch_size * POLICY_SIZE;

        // GPU buffers
        if (fallback_input_buffer_.count() < input_count) {
            fallback_input_buffer_ = CudaBuffer<int32_t>(input_count);
        }
        if (fallback_value_buffer_.count() < value_count) {
            fallback_value_buffer_ = CudaBuffer<float>(value_count);
        }
        if (fallback_policy_buffer_.count() < policy_count) {
            fallback_policy_buffer_ = CudaBuffer<float>(policy_count);
        }

        // Pinned host buffers (for async transfers)
        if (fallback_pinned_input_.count() < input_count) {
            fallback_pinned_input_ = PinnedBuffer<int32_t>(input_count);
        }
        if (fallback_pinned_value_.count() < value_count) {
            fallback_pinned_value_ = PinnedBuffer<float>(value_count);
        }
        if (fallback_pinned_policy_.count() < policy_count) {
            fallback_pinned_policy_ = PinnedBuffer<float>(policy_count);
        }
    }

    Logger& logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string value_output_name_;
    std::string policy_output_name_;

    CudaStream stream_;  // Dedicated CUDA stream for inference

    // Cached resources per batch size (with CUDA graphs)
    std::unordered_map<int32_t, CachedBatchResources> cached_resources_;

    // Double buffer resources per batch size (for pipelined inference)
    std::unordered_map<int32_t, DoubleBufferResources> double_buffer_resources_;

    // Fallback buffers for uncached batch sizes (no graph)
    CudaBuffer<int32_t> fallback_input_buffer_;
    CudaBuffer<float> fallback_value_buffer_;
    CudaBuffer<float> fallback_policy_buffer_;
    PinnedBuffer<int32_t> fallback_pinned_input_;
    PinnedBuffer<float> fallback_pinned_value_;
    PinnedBuffer<float> fallback_pinned_policy_;
};

void print_header() {
    std::println("╔════════════════════════════════════════════════════════════════╗");
    std::println("║         CatGPT Chess Engine - TensorRT Benchmark               ║");
    std::println("╚════════════════════════════════════════════════════════════════╝");
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

int main(int argc, char* argv[]) {
    print_header();
    print_cuda_info();

    // Default engine path
    fs::path engine_path = "/home/shadeform/CatGPT/catgpt.trt";
    if (argc > 1) {
        engine_path = argv[1];
    }

    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: Engine file not found: {}", engine_path.string());
        return 1;
    }

    try {
        Logger logger;
        std::println("\n┌─ Loading TensorRT Engine ────────────────────────────────────┐");
        TrtEngine engine(engine_path, logger);
        std::println("└───────────────────────────────────────────────────────────────┘");

        // Quick inference test
        std::println("\n┌─ Inference Test ──────────────────────────────────────────────┐");
        {
            std::vector<int32_t> test_input(TrtEngine::SEQ_LENGTH, 1);  // Simple test input
            auto result = engine.infer(test_input, 1);
            std::println("│ Single inference test:");
            std::println("│   Value (win prob): {:.6f}", result.values[0]);

            // Show top-5 policy logits (argmax)
            auto& policy = result.policy_logits;
            std::vector<std::pair<int, float>> indexed_logits;
            indexed_logits.reserve(policy.size());
            for (size_t i = 0; i < policy.size(); ++i) {
                indexed_logits.emplace_back(static_cast<int>(i), policy[i]);
            }
            std::partial_sort(indexed_logits.begin(), indexed_logits.begin() + 5, indexed_logits.end(),
                              [](const auto& a, const auto& b) { return a.second > b.second; });

            std::println("│   Top-5 policy logits:");
            for (int i = 0; i < 5; ++i) {
                int from_sq = indexed_logits[i].first / 73;
                int to_sq = indexed_logits[i].first % 73;
                std::println("│     [{}] from={}, to={}: {:.4f}",
                             i + 1, from_sq, to_sq, indexed_logits[i].second);
            }
        }
        std::println("└───────────────────────────────────────────────────────────────┘");

        // Benchmark various batch sizes
        std::vector<int32_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

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
