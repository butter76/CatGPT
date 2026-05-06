/**
 * Libfork-aware async counting semaphore + move-only RAII permit.
 *
 * Designed for the producer-side acquire pattern used by the LKS search:
 *
 *     // entering with a permit (passed in, or acquired from a wrapper)
 *     auto recursive_search = [](auto self, ..., Permit permit, ...)
 *         -> lf::task<void> { ... };
 *
 *     // root entry acquires the very first permit
 *     auto root_search = [](auto, ..., chess::Board b, float d)
 *         -> lf::task<void> {
 *         Permit p = co_await sem.acquire();
 *         co_await lf::call(recursive_search)(... std::move(p) ...);
 *     };
 *
 *     // inside recursive_search, fan-out:
 *     //   first child inherits via std::move(permit),
 *     //   siblings each co_await sem.acquire() before lf::fork.
 *
 * The semaphore conforms to lf::context_switcher so it can be co_awaited
 * inside any lf::task. The handoff from a non-libfork-worker thread is
 * via pool_->schedule(submit_handle) (NOT lf::resume(), which asserts on
 * the caller being a worker).
 *
 * Concretely hardcoded to lf::lazy_pool for the LKS port; the smoke test
 * uses the same concrete type.
 */

#ifndef CATGPT_LF_ASYNC_SEMAPHORE_HPP
#define CATGPT_LF_ASYNC_SEMAPHORE_HPP

#include <deque>
#include <mutex>
#include <utility>

#include <libfork/core.hpp>
#include <libfork/schedule.hpp>

namespace catgpt::lfsync {

class LfAsyncSemaphore;

/**
 * Move-only RAII handle for a single permit. The destructor releases
 * the permit back to the owning semaphore. Construct empty (no permit
 * owned) by default; default ctor is needed so we can lazily assign.
 */
class Permit {
   public:
    Permit() = default;
    explicit Permit(LfAsyncSemaphore& s) noexcept : sem_(&s) {}

    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;

    Permit(Permit&& o) noexcept : sem_(std::exchange(o.sem_, nullptr)) {}
    Permit& operator=(Permit&& o) noexcept {
        if (this != &o) {
            release();
            sem_ = std::exchange(o.sem_, nullptr);
        }
        return *this;
    }

    ~Permit() { release(); }

    [[nodiscard]] bool valid() const noexcept { return sem_ != nullptr; }

    // Manual early release. Idempotent; clears the owning pointer so
    // ~Permit() is a no-op afterwards.
    void release() noexcept;

   private:
    LfAsyncSemaphore* sem_ = nullptr;
};

class LfAsyncSemaphore {
   public:
    LfAsyncSemaphore(int initial_count, lf::lazy_pool& pool)
        : pool_(&pool), count_(initial_count) {}

    LfAsyncSemaphore(const LfAsyncSemaphore&) = delete;
    LfAsyncSemaphore& operator=(const LfAsyncSemaphore&) = delete;
    LfAsyncSemaphore(LfAsyncSemaphore&&) = delete;
    LfAsyncSemaphore& operator=(LfAsyncSemaphore&&) = delete;

    /**
     * Conforms to lf::context_switcher: await_suspend takes
     * lf::submit_handle, not std::coroutine_handle<>.
     */
    class AcquireAwaitable {
       public:
        explicit AcquireAwaitable(LfAsyncSemaphore* sem) noexcept : sem_(sem) {}

        // Fast-path: take a permit without suspending if one is free.
        bool await_ready() noexcept {
            std::lock_guard lk(sem_->mu_);
            if (sem_->count_ > 0) {
                --sem_->count_;
                return true;
            }
            return false;
        }

        // Slow-path: re-check under the lock to close the race window
        // between await_ready and await_suspend, then either take it
        // and immediately reschedule, or park on the waiter list.
        void await_suspend(lf::submit_handle h) noexcept {
            bool resume_now = false;
            {
                std::lock_guard lk(sem_->mu_);
                if (sem_->count_ > 0) {
                    --sem_->count_;
                    resume_now = true;
                } else {
                    sem_->waiters_.push_back(h);
                }
            }
            if (resume_now) {
                sem_->pool_->schedule(h);
            }
        }

        // We always have a permit by this point: either await_ready took
        // one, or await_suspend took one, or release() handed us one.
        Permit await_resume() noexcept { return Permit{*sem_}; }

       private:
        LfAsyncSemaphore* sem_;
    };

    [[nodiscard]] AcquireAwaitable acquire() noexcept {
        return AcquireAwaitable{this};
    }

    void release() noexcept {
        lf::submit_handle to_wake = nullptr;
        {
            std::lock_guard lk(mu_);
            if (!waiters_.empty()) {
                // Hand the slot directly to a waiter; count_ stays put.
                to_wake = waiters_.front();
                waiters_.pop_front();
            } else {
                ++count_;
            }
        }
        if (to_wake) {
            // Concurrent-safe per worker_context::schedule's contract.
            pool_->schedule(to_wake);
        }
    }

   private:
    lf::lazy_pool* pool_;
    std::mutex mu_;
    int count_;
    std::deque<lf::submit_handle> waiters_;
};

inline void Permit::release() noexcept {
    if (auto* s = std::exchange(sem_, nullptr)) {
        s->release();
    }
}

static_assert(lf::context_switcher<LfAsyncSemaphore::AcquireAwaitable>,
              "AcquireAwaitable must satisfy lf::context_switcher so it "
              "can be co_awaited inside an lf::task");

}  // namespace catgpt::lfsync

#endif  // CATGPT_LF_ASYNC_SEMAPHORE_HPP
