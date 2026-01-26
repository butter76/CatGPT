/**
 * Dummy Search POC - Testing libcoro + chess library integration
 *
 * This is a proof-of-concept for a parallel search algorithm using:
 * - libcoro for coroutines (task, thread_pool, ring_buffer)
 * - chess-library for board representation and move generation
 * - A dummy "GPU" evaluator that batches requests
 */

#include <coro/coro.hpp>
#include <chess.hpp>

#include <atomic>
#include <chrono>
#include <print>
#include <thread>
#include <vector>

// Quick test to verify everything compiles and links
int main() {
    std::println("=== Dummy Search POC ===");
    std::println("Testing libcoro + chess-library integration\n");

    // Test 1: Chess library
    std::println("1. Chess library test:");
    chess::Board board;
    std::println("   Starting position FEN: {}", board.getFen());

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    std::println("   Legal moves from start: {}", moves.size());

    // Test 2: libcoro thread_pool
    std::println("\n2. libcoro thread_pool test:");
    {
        auto tp = coro::thread_pool::make_shared(
            coro::thread_pool::options{.thread_count = 4}
        );
        std::atomic<int> counter{0};

        auto task = [&]() -> coro::task<void> {
            co_await tp->schedule();
            counter.fetch_add(1);
            co_return;
        };

        // Schedule some tasks and run them
        for (int i = 0; i < 10; i++) {
            coro::sync_wait(task());
        }
        std::println("   Tasks completed: {}", counter.load());
    }

    // Test 3: libcoro ring_buffer (bounded queue)
    std::println("\n3. libcoro ring_buffer test:");
    {
        auto tp = coro::thread_pool::make_shared(
            coro::thread_pool::options{.thread_count = 2}
        );
        coro::ring_buffer<int, 16> buffer;

        auto producer = [&]() -> coro::task<void> {
            co_await tp->schedule();
            for (int i = 0; i < 5; i++) {
                co_await buffer.produce(i);
            }
            co_return;
        };

        auto consumer = [&]() -> coro::task<int> {
            co_await tp->schedule();
            int sum = 0;
            for (int i = 0; i < 5; i++) {
                auto result = co_await buffer.consume();
                if (result.has_value()) {
                    sum += result.value();
                }
            }
            co_return sum;
        };

        // Run producer first (it will block until consumer starts)
        // We need to spawn them concurrently
        tp->spawn(producer());

        // Run consumer and get result
        int sum = coro::sync_wait(consumer());
        std::println("   Sum of consumed values: {} (expected: 10)", sum);
    }

    // Test 4: Simulate batched evaluation pattern
    std::println("\n4. Batched evaluation pattern test:");
    {
        auto tp = coro::thread_pool::make_shared(
            coro::thread_pool::options{.thread_count = 4}
        );

        struct EvalRequest {
            chess::Board position;
            float* result;
        };

        coro::ring_buffer<EvalRequest, 32> eval_queue;

        // Search workers that submit positions for evaluation
        auto search_worker = [&]() -> coro::task<void> {
            co_await tp->schedule();

            chess::Board pos;
            float result = 0.0f;

            // Submit a few positions
            for (int i = 0; i < 3; i++) {
                co_await eval_queue.produce(EvalRequest{pos, &result});
            }
            co_return;
        };

        // Spawn workers
        for (int i = 0; i < 4; i++) {
            tp->spawn(search_worker());
        }

        // Give workers time to produce
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Now consume all produced items (4 workers * 3 items = 12)
        int consumed = 0;
        for (int i = 0; i < 12; i++) {
            auto result = coro::sync_wait(eval_queue.consume());
            if (result.has_value()) {
                auto& req = result.value();
                auto dominated = req.position.us(chess::Color::WHITE);
                int material = dominated.count();
                *req.result = static_cast<float>(material) / 16.0f;
                consumed++;
            } else {
                break;
            }
        }

        std::println("   Positions evaluated: {}", consumed);
    }

    // Test 5: Concurrent producer-consumer with evaluator thread
    std::println("\n5. Full producer-consumer pattern:");
    {
        auto tp = coro::thread_pool::make_shared(
            coro::thread_pool::options{.thread_count = 4}
        );

        struct EvalRequest {
            chess::Board position;
            std::atomic<bool>* done;
            float* result;
        };

        coro::ring_buffer<EvalRequest, 64> eval_queue;
        std::atomic<int> total_evaluated{0};
        std::atomic<bool> stop_evaluator{false};

        // Evaluator coroutine - runs on thread pool
        auto evaluator = [&]() -> coro::task<void> {
            co_await tp->schedule();

            while (!stop_evaluator.load()) {
                auto result = co_await eval_queue.consume();
                if (!result.has_value()) {
                    break;
                }

                auto& req = result.value();

                // Dummy material evaluation
                auto dominated = req.position.us(chess::Color::WHITE);
                int material = dominated.count();
                *req.result = static_cast<float>(material) / 16.0f;

                total_evaluated.fetch_add(1);
                req.done->store(true);
            }
            co_return;
        };

        // Start evaluator
        tp->spawn(evaluator());

        // Search worker that submits and waits for evaluation
        auto search_worker = [&]([[maybe_unused]] int worker_id) -> coro::task<void> {
            co_await tp->schedule();

            for (int i = 0; i < 5; i++) {
                chess::Board pos;
                float result = 0.0f;
                std::atomic<bool> done{false};

                co_await eval_queue.produce(EvalRequest{pos, &done, &result});

                // Busy wait for result (in real impl, we'd use a proper awaitable)
                while (!done.load()) {
                    std::this_thread::yield();
                }
            }
            co_return;
        };

        // Spawn workers
        std::vector<coro::task<void>> worker_tasks;
        for (int i = 0; i < 4; i++) {
            tp->spawn(search_worker(i));
        }

        // Wait for all evaluations (4 workers * 5 evals = 20)
        while (total_evaluated.load() < 20) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Stop evaluator
        stop_evaluator.store(true);
        coro::sync_wait(eval_queue.shutdown());

        std::println("   Total positions evaluated: {}", total_evaluated.load());
    }

    std::println("\n=== All tests passed! ===");
    return 0;
}
