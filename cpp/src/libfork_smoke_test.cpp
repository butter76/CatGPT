/**
 * libfork smoke tests for the GPU-batched-eval pattern we plan to use
 * in the MCTS self-play search.
 *
 * Two scenarios run back-to-back:
 *
 * --- Test 1: basic GPU queueing -----------------------------------
 *
 *   (1) An `lf::task` can `co_await` a custom awaitable that interfaces
 *       with an external "GPU" thread, and the suspended task is woken
 *       back onto a libfork worker via `pool.schedule(submit_handle)`.
 *
 *   (2) With a SINGLE worker thread, multiple coroutines can be queued
 *       on the GPU simultaneously - i.e. the worker is never blocked
 *       waiting on a GPU result, it freely runs other ready tasks (here:
 *       the driver continuation that forks more queries) and only goes
 *       idle once everything in flight is parked on the evaluator.
 *
 * --- Test 2: bounded concurrency via async semaphore --------------
 *
 *   (3) A custom `LfAsyncSemaphore` (counting semaphore conforming to
 *       `lf::context_switcher`) bounds the number of in-flight queries.
 *       The producer (driver) `co_await`s the semaphore BEFORE forking,
 *       so when permits are exhausted the driver itself suspends -
 *       no new child coroutine frame is created until a permit is free.
 *
 *   (4) The permit is a move-only RAII handle (`Permit`) passed by
 *       value into the forked child. When the child returns, the
 *       Permit destructor releases the slot, which wakes the driver's
 *       suspended acquire and lets the next fork proceed.
 *
 *       With K permits and N >> K queries, expect ceil(N/K) GPU batches
 *       of size K, not one batch of N.
 *
 * Output is timestamped + thread-labelled so you can read the
 * interleaving by eye:
 *   - "main " - the main thread (sync_wait blocks here)
 *   - "wkr"   - the libfork worker thread (only one in these tests)
 *   - "gpu"   - the fake batched evaluator thread
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <libfork/core.hpp>
#include <libfork/schedule.hpp>

#include "lf/async_semaphore.hpp"

namespace {

// ---------- Logging helpers ---------------------------------------------

using Clock = std::chrono::steady_clock;

Clock::time_point g_t0;
std::mutex g_log_mu;
std::thread::id g_main_tid;
std::thread::id g_gpu_tid;

void log(std::string_view msg) {
    auto t_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - g_t0)
                    .count();
    auto tid = std::this_thread::get_id();
    const char* label = "wkr ";
    if (tid == g_main_tid) {
        label = "main";
    } else if (tid == g_gpu_tid) {
        label = "gpu ";
    }
    std::lock_guard lk(g_log_mu);
    std::cout << "[" << (t_us / 1000) << "." << (t_us % 1000) / 100
              << "ms " << label << "] " << msg << '\n';
}

// ---------- Fake batched GPU evaluator ----------------------------------
//
// A single thread that pulls all queued requests, waits a "batch window"
// to let stragglers pile in, then "evaluates" the whole batch (compute
// result = input * input) and fires every callback.
//
// `submit()` is thread-safe and is intended to be called from libfork
// worker threads when a coroutine suspends on an EvalAwaitable.

class FakeGpuEvaluator {
   public:
    using Callback = std::move_only_function<void(int /*result*/)>;

    explicit FakeGpuEvaluator(
        std::chrono::milliseconds batch_window = std::chrono::milliseconds(20),
        std::chrono::milliseconds inference_time = std::chrono::milliseconds(40))
        : batch_window_(batch_window), inference_time_(inference_time) {
        thread_ = std::thread([this] { run(); });
    }

    ~FakeGpuEvaluator() {
        stop_.store(true, std::memory_order_release);
        thread_.join();
    }

    std::thread::id thread_id() const noexcept { return thread_.get_id(); }

    void submit(int input, Callback cb) {
        std::lock_guard lk(mu_);
        pending_.push_back(Request{input, std::move(cb)});
    }

   private:
    struct Request {
        int input;
        Callback cb;
    };

    void run() {
        while (!stop_.load(std::memory_order_acquire)) {
            // Drain whatever is queued right now.
            std::vector<Request> batch;
            {
                std::lock_guard lk(mu_);
                batch.swap(pending_);
            }
            if (batch.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Hold the batch window open so stragglers can pile in. This
            // is essential for the smoke test: it gives the libfork
            // worker time to spin through the rest of its forks (each
            // one suspending on the GPU) before we close the batch.
            std::this_thread::sleep_for(batch_window_);
            {
                std::lock_guard lk(mu_);
                std::move(pending_.begin(), pending_.end(),
                          std::back_inserter(batch));
                pending_.clear();
            }

            {
                std::ostringstream oss;
                oss << "GPU dispatching batch of " << batch.size();
                log(oss.str());
            }

            // Simulate inference latency.
            std::this_thread::sleep_for(inference_time_);

            for (auto& r : batch) {
                int result = r.input * r.input;
                r.cb(result);
            }
            log("GPU finished resuming all callbacks");
        }
    }

    std::chrono::milliseconds batch_window_;
    std::chrono::milliseconds inference_time_;
    std::mutex mu_;
    std::vector<Request> pending_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

// ---------- Libfork-aware GPU eval awaitable ----------------------------
//
// This is the integration point. Conforms to `lf::context_switcher`:
// `await_suspend` takes `lf::submit_handle` (NOT `std::coroutine_handle<>`).
//
// On suspension we hand the submit_handle to the GPU evaluator. The GPU
// thread - which is NOT a libfork worker - cannot call `lf::resume()`
// directly (that asserts it's on a worker). Instead it calls
// `pool.schedule(h)` which is documented as concurrent-safe and pushes
// the handle into a worker's submission queue, where the next worker
// loop iteration will pick it up and resume it.

class EvalAwaitable {
   public:
    EvalAwaitable(FakeGpuEvaluator& eval, lf::lazy_pool& pool, int input)
        : eval_(&eval), pool_(&pool), input_(input) {}

    bool await_ready() const noexcept { return false; }

    void await_suspend(lf::submit_handle h) noexcept {
        // The lambda captures `this` (stable: lives in the suspended
        // coroutine frame, which doesn't move) and `h` (a tagged pointer
        // into the same frame). Both remain valid until await_resume()
        // runs on a worker after pool_->schedule(h).
        eval_->submit(input_, [this, h](int result) noexcept {
            result_ = result;
            pool_->schedule(h);
        });
    }

    int await_resume() const noexcept { return result_; }

   private:
    FakeGpuEvaluator* eval_;
    lf::lazy_pool* pool_;
    int input_;
    int result_ = 0;
};

static_assert(lf::context_switcher<EvalAwaitable>,
              "EvalAwaitable must satisfy lf::context_switcher so it can "
              "be co_awaited inside an lf::task");

// ---------- Async semaphore + permit -----------------------------------
//
// Lifted into the shared header `cpp/src/lf/async_semaphore.hpp` so that
// the LKS port (and any future libfork-using code) can reuse them. We
// just import the names below for ergonomics.

using catgpt::lfsync::LfAsyncSemaphore;
using catgpt::lfsync::Permit;

// ---------- Async functions ---------------------------------------------

// One GPU query: suspend on the evaluator, return the result.
inline constexpr auto gpu_query =
    [](auto /*self*/, FakeGpuEvaluator* eval, lf::lazy_pool* pool, int x)
    -> lf::task<int> {
    {
        std::ostringstream oss;
        oss << "gpu_query(" << x << ") suspending on GPU";
        log(oss.str());
    }
    int r = co_await EvalAwaitable{*eval, *pool, x};
    {
        std::ostringstream oss;
        oss << "gpu_query(" << x << ") resumed, result = " << r;
        log(oss.str());
    }
    co_return r;
};

// Driver: fork N gpu_queries in parallel, join, return sum.
inline constexpr auto driver = [](auto /*self*/, FakeGpuEvaluator* eval,
                                  lf::lazy_pool* pool, int n,
                                  int* out_results) -> lf::task<int> {
    log("driver: starting fork loop");
    for (int i = 0; i < n; ++i) {
        co_await lf::fork[&out_results[i], gpu_query](eval, pool, i);
        std::ostringstream oss;
        oss << "driver: forked child " << i;
        log(oss.str());
    }
    log("driver: all forked, awaiting join");
    co_await lf::join;
    log("driver: join complete");
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += out_results[i];
    co_return sum;
};

// Permit-holding GPU query. Takes the Permit by value, so its lifetime
// is tied to this coroutine's frame. When the coroutine returns, the
// Permit destructor runs which calls sem.release(), waking the next
// producer that's parked on acquire().
inline constexpr auto gpu_query_with_permit =
    [](auto /*self*/, FakeGpuEvaluator* eval, lf::lazy_pool* pool,
       Permit /*permit*/, int x) -> lf::task<int> {
    {
        std::ostringstream oss;
        oss << "child(" << x << ") suspending on GPU [holding permit]";
        log(oss.str());
    }
    int r = co_await EvalAwaitable{*eval, *pool, x};
    {
        std::ostringstream oss;
        oss << "child(" << x << ") resumed (result=" << r
            << "), permit will release on co_return";
        log(oss.str());
    }
    co_return r;
    // ~Permit() runs as the coroutine frame is destroyed -> sem.release().
};

// Bounded driver: acquires a permit BEFORE forking the next child. When
// permits are exhausted, this coroutine itself suspends in acquire() -
// no new child frame gets allocated until a permit is free. That's the
// real backpressure point.
inline constexpr auto bounded_driver =
    [](auto /*self*/, FakeGpuEvaluator* eval, lf::lazy_pool* pool,
       LfAsyncSemaphore* sem, int n, int* out_results) -> lf::task<int> {
    log("bounded_driver: starting acquire-then-fork loop");
    for (int i = 0; i < n; ++i) {
        {
            std::ostringstream oss;
            oss << "bounded_driver: awaiting permit for child " << i;
            log(oss.str());
        }
        Permit p = co_await sem->acquire();
        {
            std::ostringstream oss;
            oss << "bounded_driver: got permit, forking child " << i;
            log(oss.str());
        }
        co_await lf::fork[&out_results[i], gpu_query_with_permit](
            eval, pool, std::move(p), i);
    }
    log("bounded_driver: all forked, awaiting join");
    co_await lf::join;
    log("bounded_driver: join complete");
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += out_results[i];
    co_return sum;
};

}  // namespace

// sum_{i=0..N-1} i^2 = (N-1) * N * (2N-1) / 6
static constexpr int sum_of_squares(int n) {
    return (n - 1) * n * (2 * n - 1) / 6;
}

int main() {
    g_t0 = Clock::now();
    g_main_tid = std::this_thread::get_id();

    log("================================================================");
    log("=== libfork smoke test ===");
    log("================================================================");

    FakeGpuEvaluator eval{
        std::chrono::milliseconds(30),  // batch_window
        std::chrono::milliseconds(40),  // inference_time
    };
    g_gpu_tid = eval.thread_id();
    log("GPU evaluator thread started (batch_window=30ms, inference=40ms)");

    // SINGLE worker. The smoke tests' whole point is that this one
    // thread never blocks waiting on the GPU - it just keeps forking
    // children until they're all parked, then idles.
    //
    // lazy_pool sleeps when idle (vs busy_pool which spins). For the
    // GPU-batched-eval pattern we expect frequent idle periods (worker
    // parks at lf::join, GPU thread eventually wakes it) so lazy_pool
    // is the right default.
    lf::lazy_pool pool{1};
    log("libfork lazy_pool started with 1 worker");

    int rc = 0;

    // ---------------- Test 1: unbounded fan-out ----------------
    {
        log("");
        log("---------------- TEST 1: unbounded fan-out ----------------");
        log("Expect: ONE batch of N (worker forks all N, then idles at join)");

        constexpr int N = 8;
        std::vector<int> results(N, 0);
        int sum = lf::sync_wait(pool, driver, &eval, &pool, N, results.data());

        std::ostringstream oss;
        oss << "Test 1: driver returned sum=" << sum
            << " (expected " << sum_of_squares(N) << ")";
        log(oss.str());
        if (sum != sum_of_squares(N)) {
            log("Test 1 FAIL: sum mismatch");
            rc = 1;
        } else {
            log("Test 1 PASS");
        }
    }

    // ---------------- Test 2: bounded by semaphore ----------------
    {
        log("");
        log("---------------- TEST 2: K-permit semaphore ---------------");
        constexpr int N = 12;
        constexpr int K = 3;
        log("N=12 queries, K=3 permits");
        log("Expect: ceil(N/K)=4 batches of K=3 (NOT one batch of N)");
        log("Producer suspends on acquire() once K permits are out, no");
        log("new child frame is allocated until a child returns its permit.");

        LfAsyncSemaphore sem{K, pool};

        std::vector<int> results(N, 0);
        int sum = lf::sync_wait(pool, bounded_driver, &eval, &pool, &sem, N,
                                results.data());

        std::ostringstream oss;
        oss << "Test 2: bounded_driver returned sum=" << sum
            << " (expected " << sum_of_squares(N) << ")";
        log(oss.str());
        if (sum != sum_of_squares(N)) {
            log("Test 2 FAIL: sum mismatch");
            rc = 1;
        } else {
            log("Test 2 PASS");
        }
    }

    log("");
    log(rc == 0 ? "OVERALL PASS" : "OVERALL FAIL");
    return rc;
}
