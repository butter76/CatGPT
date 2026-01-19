/**
 * TensorRT Benchmark for CatGPT Chess Engine
 *
 * Loads a TensorRT engine and benchmarks inference at various batch sizes.
 * The model expects:
 *   - Input: int32 tensor of shape (batch, 64) - chess position tokens
 *   - Output: float32 tensor of shape (batch,) - win probability
 */

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
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

// RAII wrapper for CUDA memory
template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) : size_(count * sizeof(T)) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size_));
        }
    }

    ~CudaBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move only
    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size_bytes() const { return size_; }

    void copy_from_host(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// TensorRT Engine wrapper
class TrtEngine {
public:
    static constexpr int32_t SEQ_LENGTH = 64;
    static constexpr int32_t VOCAB_SIZE = 28;

    TrtEngine(const fs::path& engine_path, Logger& logger) : logger_(logger) {
        load_engine(engine_path);
        setup_io();
    }

    ~TrtEngine() = default;

    // Run inference on a batch of chess positions
    // Input: vector of int32 tokens, size = batch_size * SEQ_LENGTH
    // Output: vector of float32 win probabilities, size = batch_size
    std::vector<float> infer(std::span<const int32_t> input_tokens, int32_t batch_size) {
        if (static_cast<size_t>(batch_size * SEQ_LENGTH) != input_tokens.size()) {
            throw std::runtime_error("Input size mismatch");
        }

        // Resize GPU buffers if needed
        ensure_buffers(batch_size);

        // Copy input to GPU
        input_buffer_.copy_from_host(input_tokens.data(), input_tokens.size());

        // Set tensor addresses
        context_->setTensorAddress(input_name_.c_str(), input_buffer_.get());
        context_->setTensorAddress(output_name_.c_str(), output_buffer_.get());

        // Set input shape (dynamic batch)
        nvinfer1::Dims input_dims;
        input_dims.nbDims = 2;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = SEQ_LENGTH;
        context_->setInputShape(input_name_.c_str(), input_dims);

        // Execute
        if (!context_->enqueueV3(nullptr)) {
            throw std::runtime_error("TensorRT inference failed");
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy output back
        std::vector<float> output(batch_size);
        output_buffer_.copy_to_host(output.data(), batch_size);

        return output;
    }

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
    const std::string& output_name() const { return output_name_; }

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
                output_name_ = name;
                std::println("  Output '{}': {} {}", name, dims_str, dtype_str);
            }
        }

        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("Could not find input/output tensors");
        }
    }

    void ensure_buffers(int32_t batch_size) {
        size_t input_size = batch_size * SEQ_LENGTH;
        size_t output_size = batch_size;

        if (input_buffer_.size_bytes() < input_size * sizeof(int32_t)) {
            input_buffer_ = CudaBuffer<int32_t>(input_size);
        }
        if (output_buffer_.size_bytes() < output_size * sizeof(float)) {
            output_buffer_ = CudaBuffer<float>(output_size);
        }
    }

    Logger& logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string output_name_;

    CudaBuffer<int32_t> input_buffer_;
    CudaBuffer<float> output_buffer_;
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
            auto output = engine.infer(test_input, 1);
            std::println("│ Single inference test: value = {:.6f}", output[0]);
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

        std::println("\n★ Optimal batch size: {} ({:.0f} samples/sec)", best_batch, best_throughput);

    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
