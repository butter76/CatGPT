/**
 * In-flight eval dedup table — "claim the eval before you run it".
 *
 * Problem this solves
 * -------------------
 * The LKS descent evaluates an unexpanded position by submitting it to the
 * GPU and *suspending* (`co_await EvalAwaitable`). Only after the result
 * returns does it claim the TT slot (`SearchArena::find_or_claim`). Between
 * the initial TT `find()` miss and the post-eval `find_or_claim`, any other
 * coroutine — on the same worker or, far more commonly, on another worker
 * sharing the TT — can also miss and submit its own GPU eval for the SAME
 * position. The TT later dedups the *storage* (one CAS winner, the loser
 * orphans its bytes), but both GPU evals already ran. That is wasted GPU
 * time, the scarcest resource in the search.
 *
 * The fix is to dedup the *work*, not just the storage: publish the intent
 * to evaluate a position BEFORE suspending, and make every later arrival
 * block on that intent instead of starting its own eval.
 *
 * Protocol
 * --------
 *   - On a TT miss, a coroutine calls `claim_or_join(key, secondary)`.
 *     Exactly one caller per (key, secondary) wins the race and becomes the
 *     LEADER (`role.leader == true`); it runs the GPU eval and publishes to
 *     the TT exactly as before. Every other caller becomes a FOLLOWER; it
 *     does NOT evaluate.
 *   - A follower `co_await`s `EvalWaitAwaitable(slot, my_pool)`, which parks
 *     its coroutine on the slot's lock-free waiter stack (it does not spin —
 *     parking returns control to the worker's pool so it stays productive).
 *   - When the leader finishes publishing to the TT it calls `wake(slot)`,
 *     which atomically closes the waiter stack and hands each parked
 *     follower's `lf::submit_handle` back to that follower's OWN
 *     `lf::lazy_pool` via `pool->schedule(...)` — exactly the cross-thread
 *     wake path the GPU thread already uses. Followers then resume, read the
 *     now-published TT entry, and continue. No follower ever runs a
 *     duplicate GPU eval.
 *
 * Match is on the (key, secondary) pair, mirroring the TT: a genuine 64-bit
 * primary collision with a different secondary is a different position and
 * gets its own leader (worst case: one extra eval, never a wrong result).
 *
 * Lost-wakeup race
 * ----------------
 * The waiter stack is a Treiber stack with a `kInflightClosed` sentinel.
 * The leader `exchange`s the head to the sentinel; a follower's push CAS
 * observes the sentinel iff the leader already closed. So either the push
 * lands before the close (leader schedules it) or the follower sees the
 * sentinel and self-reschedules. No follower is ever lost or double-runs.
 *
 * Lifetime / reclamation
 * ----------------------
 * A slot is only needed during [leader claims] -> [leader publishes]; after
 * the leader removes it from the discovery map, new arrivals hit the TT and
 * never see it. Live slots are therefore bounded by peak in-flight evals
 * (~= total permits across workers), NOT by the lifetime eval count K. Slots
 * are recycled through a per-shard freelist guarded by an atomic refcount:
 * a follower pins the slot under the shard lock in `claim_or_join`; the slot
 * is returned to the freelist only when the refcount hits zero, so a slot is
 * never freed while a follower is mid-park. Ownership of the backing memory
 * is tracked in `all_slots_` and released in the destructor.
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_V2_INFLIGHT_TABLE_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_V2_INFLIGHT_TABLE_HPP

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <libfork/core.hpp>
#include <libfork/schedule.hpp>

namespace catgpt::v2 {

/**
 * One parked follower. Lives in the follower's coroutine frame (inside
 * `EvalWaitAwaitable`), so there is no separate allocation and its lifetime
 * spans push -> schedule. The leader reads `next` before scheduling, since
 * scheduling may resume-and-destroy the frame.
 */
struct InflightWaiter {
    lf::submit_handle continuation = nullptr;
    lf::lazy_pool*    pool         = nullptr;
    InflightWaiter*   next         = nullptr;
};

/**
 * Sentinel stored in `InflightSlot::waiters` once the leader has published
 * its result to the TT: "do not park, the answer is in the TT now". Distinct
 * from nullptr (the empty stack) and from any real `InflightWaiter*`.
 */
inline InflightWaiter* const kInflightClosed =
    reinterpret_cast<InflightWaiter*>(1);

/**
 * A single in-flight eval. Discovered via the table's sharded map; the
 * waiter stack is manipulated lock-free so wakeups never take a shard lock.
 */
struct InflightSlot {
    uint64_t key       = 0;
    uint32_t secondary = 0;
    // Pins outstanding from the leader (1 at claim) plus each follower that
    // looked it up under the shard lock. Slot returns to the freelist when
    // this hits 0.
    std::atomic<int>             refcount{0};
    // Treiber stack of parked followers; kInflightClosed after the leader
    // publishes. Reset to nullptr on (re)claim.
    std::atomic<InflightWaiter*> waiters{nullptr};
};

/** Result of `claim_or_join`. `slot` is never null. */
struct InflightRole {
    InflightSlot* slot;
    bool          leader;
};

class InflightTable {
public:
    /**
     * @param peak_concurrency_hint  Rough upper bound on simultaneously
     *                               in-flight evals (~= total permits across
     *                               workers). Used only to size the shard
     *                               array; correctness does not depend on it.
     */
    explicit InflightTable(int peak_concurrency_hint = 4096)
        : num_shards_(pick_num_shards(peak_concurrency_hint))
        , shard_mask_(num_shards_ - 1)
        , shards_(new Shard[num_shards_]) {}

    ~InflightTable() {
        for (InflightSlot* s : all_slots_) delete s;
    }

    InflightTable(const InflightTable&)            = delete;
    InflightTable& operator=(const InflightTable&) = delete;
    InflightTable(InflightTable&&)                 = delete;
    InflightTable& operator=(InflightTable&&)      = delete;

    /**
     * Claim the right to evaluate (key, secondary), or join an existing
     * in-flight eval for it.
     *
     * Returns `{slot, leader=true}` for exactly one caller per
     * (key, secondary): that caller must run the eval, publish to the TT,
     * then call `wake(slot)`. Every other concurrent caller gets
     * `{slot, leader=false}` and must `co_await EvalWaitAwaitable(slot, pool)`
     * then `release(slot)`. The returned slot is pinned on behalf of the
     * caller (leader or follower); `wake`/`release` drop that pin.
     */
    [[nodiscard]] InflightRole claim_or_join(uint64_t key,
                                             uint32_t secondary) {
        const Composite c = composite(key, secondary);
        Shard& s = shard_for(key);
        std::lock_guard<std::mutex> lk(s.mu);
        if (auto it = s.map.find(c); it != s.map.end()) {
            InflightSlot* slot = it->second;
            slot->refcount.fetch_add(1, std::memory_order_relaxed);
            return {slot, /*leader=*/false};
        }
        InflightSlot* slot = alloc_slot(s);
        slot->key       = key;
        slot->secondary = secondary;
        slot->waiters.store(nullptr, std::memory_order_relaxed);
        slot->refcount.store(1, std::memory_order_relaxed);  // leader pin
        s.map.emplace(c, slot);
        return {slot, /*leader=*/true};
    }

    /**
     * Leader: call exactly once, AFTER publishing the result to the TT.
     * Wakes every parked follower and removes the slot from the discovery
     * map (so new arrivals hit the now-published TT entry), then drops the
     * leader's pin.
     */
    void wake(InflightSlot* slot) noexcept {
        // 1. Close the stack and drain it. Reading `next` before scheduling
        //    is mandatory: schedule() may resume-and-destroy the follower
        //    frame (and with it the InflightWaiter) before we return.
        InflightWaiter* w =
            slot->waiters.exchange(kInflightClosed, std::memory_order_acq_rel);
        while (w != nullptr) {
            InflightWaiter* next = w->next;
            w->pool->schedule(w->continuation);
            w = next;
        }
        // 2. Remove from the discovery map.
        Shard& s = shard_for(slot->key);
        {
            std::lock_guard<std::mutex> lk(s.mu);
            s.map.erase(composite(slot->key, slot->secondary));
        }
        // 3. Drop the leader's pin.
        release(slot);
    }

    /**
     * Follower: drop the pin taken by `claim_or_join`, AFTER the
     * `EvalWaitAwaitable` co_await has resumed. Returns the slot to the
     * freelist when the last pin is gone.
     */
    void release(InflightSlot* slot) noexcept {
        if (slot->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            Shard& s = shard_for(slot->key);
            std::lock_guard<std::mutex> lk(s.mu);
            s.freelist.push_back(slot);
        }
    }

    [[nodiscard]] int num_shards() const noexcept { return num_shards_; }

private:
    __extension__ using Composite = unsigned __int128;

    struct CompositeHash {
        std::size_t operator()(Composite c) const noexcept {
            return static_cast<std::size_t>(c ^ (c >> 64));
        }
    };

    struct Shard {
        std::mutex mu;
        std::unordered_map<Composite, InflightSlot*, CompositeHash> map;
        std::vector<InflightSlot*> freelist;  // recycled slots, reused first
    };

    static Composite composite(uint64_t key, uint32_t secondary) noexcept {
        return (static_cast<Composite>(key) << 32) |
               static_cast<Composite>(secondary);
    }

    Shard& shard_for(uint64_t key) noexcept {
        // Zobrist keys are uniformly random; high bits decorrelate from the
        // low bits the per-shard map hashes on.
        return shards_[(key >> 40) & shard_mask_];
    }

    // Caller must hold `s.mu`. Reuses a freelisted slot when available,
    // otherwise allocates and records ownership in `all_slots_`.
    InflightSlot* alloc_slot(Shard& s) {
        if (!s.freelist.empty()) {
            InflightSlot* slot = s.freelist.back();
            s.freelist.pop_back();
            return slot;
        }
        auto* slot = new InflightSlot();
        {
            std::lock_guard<std::mutex> lk(all_mu_);
            all_slots_.push_back(slot);
        }
        return slot;
    }

    static int pick_num_shards(int peak_concurrency_hint) {
        int want = peak_concurrency_hint / 8;
        if (want < 64) want = 64;
        if (want > 4096) want = 4096;
        return static_cast<int>(std::bit_ceil(static_cast<unsigned>(want)));
    }

    int               num_shards_;
    uint64_t          shard_mask_;
    std::unique_ptr<Shard[]> shards_;

    std::mutex                  all_mu_;
    std::vector<InflightSlot*>  all_slots_;  // owns every slot ever created
};

/**
 * Awaitable that parks the calling coroutine on an `InflightSlot` until the
 * leader publishes and calls `InflightTable::wake`. Conforms to
 * `lf::context_switcher` (await_suspend takes `lf::submit_handle`) so it can
 * be co_awaited inside any `lf::task`, mirroring `EvalAwaitable` and
 * `LfAsyncSemaphore::AcquireAwaitable`.
 *
 * `pool` must be the follower's own worker pool — the pool the resumed
 * coroutine will run on. The embedded `InflightWaiter` lives in the
 * coroutine frame for the duration of the suspension.
 */
class EvalWaitAwaitable {
public:
    EvalWaitAwaitable(InflightSlot* slot, lf::lazy_pool* pool) noexcept
        : slot_(slot), pool_(pool) {}

    // Fast path: leader already published; don't suspend at all.
    [[nodiscard]] bool await_ready() const noexcept {
        return slot_->waiters.load(std::memory_order_acquire) == kInflightClosed;
    }

    void await_suspend(lf::submit_handle h) noexcept {
        waiter_.continuation = h;
        waiter_.pool         = pool_;
        InflightWaiter* head = slot_->waiters.load(std::memory_order_acquire);
        while (true) {
            if (head == kInflightClosed) {
                // Leader closed the stack between await_ready and now;
                // resume immediately on our own pool.
                pool_->schedule(h);
                return;
            }
            waiter_.next = head;
            if (slot_->waiters.compare_exchange_weak(
                    head, &waiter_,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                return;  // parked; leader's wake() will schedule us
            }
            // head reloaded by the CAS; retry (may now be kInflightClosed).
        }
    }

    void await_resume() const noexcept {}

private:
    InflightSlot*  slot_;
    lf::lazy_pool* pool_;
    InflightWaiter waiter_{};
};

static_assert(lf::context_switcher<EvalWaitAwaitable>,
              "EvalWaitAwaitable must satisfy lf::context_switcher so it can "
              "be co_awaited inside an lf::task");

}  // namespace catgpt::v2

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_V2_INFLIGHT_TABLE_HPP
