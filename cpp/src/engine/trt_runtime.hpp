/**
 * Shared TensorRT / CUDA runtime scaffolding.
 *
 * Holds the bits used by every TRT-backed evaluator implementation:
 *   - TrtLogger:        nvinfer1::ILogger that prints WARNING+ to stderr.
 *   - CATGPT_CUDA_CHECK: macro that throws std::runtime_error on a
 *                        non-success cudaError_t.
 *
 * Heavyweight (pulls in <NvInfer.h> and <cuda_runtime.h>); only
 * include from files that already need the TRT/CUDA runtime
 * (i.e. the BatchEvaluator implementations).
 */

#ifndef CATGPT_ENGINE_TRT_RUNTIME_HPP
#define CATGPT_ENGINE_TRT_RUNTIME_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <print>
#include <stdexcept>
#include <string>

namespace catgpt {

#define CATGPT_CUDA_CHECK(call)                                                   \
    do {                                                                          \
        cudaError_t status = (call);                                              \
        if (status != cudaSuccess) {                                              \
            throw std::runtime_error(                                             \
                std::string("CUDA error: ") + cudaGetErrorString(status));        \
        }                                                                         \
    } while (0)

/**
 * TensorRT logger implementation. Prints WARNING and ERROR severity
 * messages to stderr, ignores INFO/VERBOSE.
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

}  // namespace catgpt

#endif  // CATGPT_ENGINE_TRT_RUNTIME_HPP
