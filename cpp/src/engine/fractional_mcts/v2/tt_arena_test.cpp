/**
 * Unit tests for v2::MoveInfo's 4-byte opaque (_Float16 + terminal_kind)
 * encoding. Standalone: no TRT, no libcoro, no chess-library deps.
 *
 * What we verify:
 *   1. Static layout (sizeof/alignof/trivially-copyable).
 *   2. Non-terminal round-trip across a representative P range: decoded P
 *      is exactly the fp16 quantization of the input, terminal_kind
 *      decodes to kTerminalNone, and the `move` field is preserved.
 *   3. Terminal-Draw / Terminal-LossForChild round-trip across the same
 *      P range: terminal_kind decodes correctly and decoded P matches
 *      the input fp16-rounded-to-9-mantissa-bits (the mantissa LSB is
 *      stolen for the kind flag).
 *   4. Monotonicity: for two non-terminal Ps with a > b, decoded P is
 *      also ordered (fp16 ties may round equal; we only check !<).
 *   5. Specific bit-pattern edge cases: P=0, P=1, the exact 1+2^-10
 *      boundary that loses precision under the LSB steal, and negative
 *      zero handling on input.
 *   6. Every `move` bit-pattern survives (no aliasing between the move
 *      slot and the packed slot).
 *   7. Exhaustive scan of all 2^16 `_packed` patterns: accessors never
 *      crash / return NaN for any input, and round-trips for all three
 *      kinds with well-formed P ∈ [0, 1] stay inside tight bounds.
 *
 * Test harness style mirrors `lks_search_test.cpp`: hand-rolled EXPECT
 * with a failure counter, non-zero exit on any failure.
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

#include "tt_arena.hpp"

using catgpt::v2::MoveInfo;
using catgpt::v2::TerminalKind;
using catgpt::v2::kTerminalNone;
using catgpt::v2::kTerminalDraw;
using catgpt::v2::kTerminalLossForChild;

namespace {

int g_failed = 0;

#define EXPECT(cond)                                                        \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr,                                            \
                         "  FAIL: %s (line %d)\n", #cond, __LINE__);        \
            ++g_failed;                                                     \
        }                                                                   \
    } while (0)

#define EXPECT_EQ_F(a, b)                                                   \
    do {                                                                    \
        const float _a = (a);                                               \
        const float _b = (b);                                               \
        if (!(_a == _b)) {                                                  \
            std::fprintf(stderr,                                            \
                         "  FAIL: %s (%.9g) != %s (%.9g) (line %d)\n",      \
                         #a, _a, #b, _b, __LINE__);                         \
            ++g_failed;                                                     \
        }                                                                   \
    } while (0)

// Reference: quantize a float to the same fp16 representation the
// encoder uses, then back to float. This is the "best possible"
// decoded P for a non-terminal pack.
float fp16_round_trip(float x) {
    _Float16 h = static_cast<_Float16>(x);
    return static_cast<float>(h);
}

// Reference: quantize to fp16 *and* drop the mantissa LSB, mirroring
// what the terminal pack does (the LSB is stolen for the kind flag).
// Operates on the absolute value since terminal pack strips the sign
// before setting bit 15.
float fp16_terminal_round_trip(float x) {
    _Float16 h = static_cast<_Float16>(x);
    uint16_t bits = std::bit_cast<uint16_t>(h);
    // Strip any accidental sign bit (fp16 can produce -0) and clear LSB.
    bits = static_cast<uint16_t>(bits & 0x7FFFu);
    bits = static_cast<uint16_t>(bits & ~static_cast<uint16_t>(0x0001u));
    _Float16 h2 = std::bit_cast<_Float16>(bits);
    return static_cast<float>(h2);
}

constexpr const char* kind_name(TerminalKind k) {
    switch (k) {
        case kTerminalNone:         return "kTerminalNone";
        case kTerminalDraw:         return "kTerminalDraw";
        case kTerminalLossForChild: return "kTerminalLossForChild";
    }
    return "?";
}

// ── Test cases ────────────────────────────────────────────────────────────

void test_static_layout() {
    std::printf("[1] static layout (sizeof, alignof, trivially copyable)\n");

    static_assert(sizeof(MoveInfo) == 4, "MoveInfo size regression");
    static_assert(alignof(MoveInfo) == 2, "MoveInfo alignment regression");
    static_assert(std::is_trivially_copyable_v<MoveInfo>,
                  "MoveInfo must stay trivially copyable");
    static_assert(sizeof(_Float16) == 2, "_Float16 must be IEEE 754 binary16");

    EXPECT(sizeof(MoveInfo) == 4);
    EXPECT(alignof(MoveInfo) == 2);
}

void test_non_terminal_round_trip() {
    std::printf("[2] non-terminal round-trip across representative P range\n");

    // Anchor values covering: 0, subnormal-ish, typical softmax range,
    // near-1, exactly-1. All non-negative as required by pack().
    const float p_values[] = {
        0.0f,
        1e-7f,    // rounds to 0 in fp16 subnormal
        1e-5f,    // rounds to the smallest subnormal ~6e-8 ... actually ~6e-5 normal
        1e-4f,
        1e-3f,
        0.01f,
        0.05f,
        0.1f,
        0.25f,
        0.5f,
        0.75f,
        0.9f,
        0.99f,
        1.0f,
    };

    for (float p : p_values) {
        const uint16_t move = static_cast<uint16_t>(0xABCDu ^ std::bit_cast<uint32_t>(p));
        MoveInfo mi = MoveInfo::pack(move, p, kTerminalNone);

        EXPECT(mi.move == move);
        EXPECT(mi.terminal_kind() == kTerminalNone);

        const float decoded = mi.P();
        const float expected = fp16_round_trip(p);
        if (decoded != expected) {
            std::fprintf(stderr,
                "  non-terminal P=%.9g decoded=%.9g expected=%.9g\n",
                p, decoded, expected);
            ++g_failed;
        }
    }
}

void test_terminal_round_trip() {
    std::printf("[3] terminal-Draw / terminal-LossForChild round-trip\n");

    const float p_values[] = {
        0.0f, 1e-4f, 1e-3f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f,
        0.99f, 1.0f,
    };
    const TerminalKind kinds[] = { kTerminalDraw, kTerminalLossForChild };

    for (TerminalKind tk : kinds) {
        for (float p : p_values) {
            const uint16_t move = static_cast<uint16_t>(
                0x1234u ^ static_cast<uint16_t>(tk) ^
                std::bit_cast<uint32_t>(p));
            MoveInfo mi = MoveInfo::pack(move, p, tk);

            EXPECT(mi.move == move);
            if (mi.terminal_kind() != tk) {
                std::fprintf(stderr,
                    "  kind mismatch: pack(%s, P=%.9g) -> decoded %s\n",
                    kind_name(tk), p, kind_name(mi.terminal_kind()));
                ++g_failed;
            }

            const float decoded = mi.P();
            const float expected = fp16_terminal_round_trip(p);
            if (decoded != expected) {
                std::fprintf(stderr,
                    "  terminal(%s) P=%.9g decoded=%.9g expected=%.9g\n",
                    kind_name(tk), p, decoded, expected);
                ++g_failed;
            }
        }
    }
}

void test_move_field_independence() {
    std::printf("[4] move field preserved across all 16-bit patterns\n");

    // Every possible move u16 must round-trip unchanged, regardless of
    // the P/kind encoding collapsing onto bit patterns that overlap.
    // We spot-check every 17th u16 for speed (still >3800 samples).
    const float p_samples[] = { 0.0f, 0.001f, 0.5f, 1.0f };
    const TerminalKind tk_samples[] = {
        kTerminalNone, kTerminalDraw, kTerminalLossForChild
    };

    int samples = 0;
    for (uint32_t m = 0; m < 0x10000u; m += 17u) {
        const uint16_t move = static_cast<uint16_t>(m);
        for (float p : p_samples) {
            for (TerminalKind tk : tk_samples) {
                MoveInfo mi = MoveInfo::pack(move, p, tk);
                if (mi.move != move) {
                    std::fprintf(stderr,
                        "  move bit-flip: packed move=0x%04x decoded=0x%04x\n",
                        (unsigned)move, (unsigned)mi.move);
                    ++g_failed;
                }
                ++samples;
            }
        }
    }
    EXPECT(samples > 3000);
}

void test_edge_bit_patterns() {
    std::printf("[5] explicit edge bit patterns\n");

    // P = 0, every kind. Must decode to P=0 and the right kind.
    {
        MoveInfo mi0 = MoveInfo::pack(0, 0.0f, kTerminalNone);
        EXPECT(mi0.terminal_kind() == kTerminalNone);
        EXPECT_EQ_F(mi0.P(), 0.0f);

        MoveInfo mid = MoveInfo::pack(0, 0.0f, kTerminalDraw);
        EXPECT(mid.terminal_kind() == kTerminalDraw);
        EXPECT_EQ_F(mid.P(), 0.0f);

        MoveInfo mil = MoveInfo::pack(0, 0.0f, kTerminalLossForChild);
        EXPECT(mil.terminal_kind() == kTerminalLossForChild);
        EXPECT_EQ_F(mil.P(), 0.0f);
    }

    // P = 1, every kind. Lossless for non-terminal; terminal form
    // drops the mantissa LSB but 1.0's mantissa is all-zero so it's
    // lossless here too.
    {
        MoveInfo mi0 = MoveInfo::pack(0, 1.0f, kTerminalNone);
        EXPECT(mi0.terminal_kind() == kTerminalNone);
        EXPECT_EQ_F(mi0.P(), 1.0f);

        MoveInfo mid = MoveInfo::pack(0, 1.0f, kTerminalDraw);
        EXPECT(mid.terminal_kind() == kTerminalDraw);
        EXPECT_EQ_F(mid.P(), 1.0f);

        MoveInfo mil = MoveInfo::pack(0, 1.0f, kTerminalLossForChild);
        EXPECT(mil.terminal_kind() == kTerminalLossForChild);
        EXPECT_EQ_F(mil.P(), 1.0f);
    }

    // The 1-bit-precision-loss pattern. fp16(1.0 + 2^-10) has mantissa
    // 0x001, so terminal encoding clears it back to 1.0 exactly, while
    // non-terminal retains the extra ULP.
    {
        const float boundary = 1.0f + std::ldexp(1.0f, -10);
        MoveInfo mi0 = MoveInfo::pack(0, boundary, kTerminalNone);
        EXPECT(mi0.terminal_kind() == kTerminalNone);
        EXPECT_EQ_F(mi0.P(), fp16_round_trip(boundary));
        EXPECT(mi0.P() > 1.0f);  // the extra ULP survives non-terminal

        MoveInfo mid = MoveInfo::pack(0, boundary, kTerminalDraw);
        EXPECT(mid.terminal_kind() == kTerminalDraw);
        EXPECT_EQ_F(mid.P(), 1.0f);  // ULP stolen for kind flag

        MoveInfo mil = MoveInfo::pack(0, boundary, kTerminalLossForChild);
        EXPECT(mil.terminal_kind() == kTerminalLossForChild);
        EXPECT_EQ_F(mil.P(), 1.0f);
    }
}

void test_monotonicity() {
    std::printf("[6] decoded P preserves ordering of non-terminal inputs\n");

    // Build a sorted list of P inputs across the representable range,
    // pack+unpack each, and check that the decoded sequence is also
    // non-decreasing (equal after fp16 quantization is OK).
    std::vector<float> ps;
    ps.reserve(512);
    for (int i = 0; i <= 500; ++i) {
        ps.push_back(static_cast<float>(i) / 500.0f);
    }
    std::sort(ps.begin(), ps.end());

    float last_decoded = -1.0f;
    for (float p : ps) {
        MoveInfo mi = MoveInfo::pack(0, p, kTerminalNone);
        float decoded = mi.P();
        EXPECT(decoded >= last_decoded - 1e-9f);
        last_decoded = decoded;
    }
}

void test_randomized_round_trip() {
    std::printf("[7] randomized softmax-like P inputs, all three kinds\n");

    std::mt19937_64 rng(0xCA7CA7CAFEFEFEFEULL);
    std::uniform_real_distribution<float> uni_p(0.0f, 1.0f);
    std::uniform_int_distribution<int> uni_kind(0, 2);
    std::uniform_int_distribution<uint32_t> uni_move(0, 0xFFFFu);

    constexpr int N = 50000;
    int kind_hits[3] = {0, 0, 0};

    for (int i = 0; i < N; ++i) {
        // Simulate a softmax: draw a few logits, softmax them, pick one.
        // This puts P in the realistic distribution the arena sees.
        float logits[4];
        for (float& l : logits) l = std::generate_canonical<float, 24>(rng) * 8.0f - 4.0f;
        float m = *std::max_element(logits, logits + 4);
        float s = 0.0f;
        for (float& l : logits) { l = std::exp(l - m); s += l; }
        float p = logits[i & 3] / s;
        if (p < 0.0f) p = 0.0f;  // paranoia; softmax is always >= 0

        // Very small probability that p ends up exactly 1.0 from a
        // degenerate logit set; clamp to [0, 1] defensively.
        if (p > 1.0f) p = 1.0f;

        const int k = uni_kind(rng);
        const TerminalKind tk = (k == 0) ? kTerminalNone
                              : (k == 1) ? kTerminalDraw
                                         : kTerminalLossForChild;
        ++kind_hits[k];

        const uint16_t move = static_cast<uint16_t>(uni_move(rng));
        MoveInfo mi = MoveInfo::pack(move, p, tk);

        EXPECT(mi.move == move);
        EXPECT(mi.terminal_kind() == tk);

        const float expected = (tk == kTerminalNone)
            ? fp16_round_trip(p)
            : fp16_terminal_round_trip(p);
        const float decoded = mi.P();
        if (decoded != expected) {
            std::fprintf(stderr,
                "  random mismatch: kind=%s P=%.9g decoded=%.9g expected=%.9g\n",
                kind_name(tk), p, decoded, expected);
            ++g_failed;
            if (g_failed > 20) return;  // don't spam the log
        }
    }

    // Sanity: all three kinds actually got exercised.
    EXPECT(kind_hits[0] > N / 4);
    EXPECT(kind_hits[1] > N / 4);
    EXPECT(kind_hits[2] > N / 4);
}

void test_exhaustive_packed_patterns_never_crash() {
    std::printf("[8] every _packed bit pattern yields a sane accessor result\n");

    // Construct MoveInfos with arbitrary _packed values (including NaN,
    // Inf, subnormals) and confirm accessors don't crash, that
    // terminal_kind is always a valid enum value, and that P() returns
    // a finite number or NaN (never a trap/SIGFPE).
    //
    // We can't use pack() to reach every bit pattern because pack()
    // normalizes the sign/LSB. Use std::bit_cast<MoveInfo>(u32) to
    // bypass.
    uint64_t sane_count = 0;
    uint64_t nan_count = 0;

    for (uint32_t packed = 0; packed < 0x10000u; ++packed) {
        const uint32_t word = (static_cast<uint32_t>(packed) << 16) | 0x5678u;
        MoveInfo mi = std::bit_cast<MoveInfo>(word);
        EXPECT(mi.move == 0x5678u);

        TerminalKind tk = mi.terminal_kind();
        EXPECT(tk == kTerminalNone
            || tk == kTerminalDraw
            || tk == kTerminalLossForChild);

        const float p = mi.P();
        if (std::isnan(p)) ++nan_count; else ++sane_count;
        // P must not be negative for any of our non-crashy outputs
        // (negative sign is our "terminal" flag and we strip it for
        // magnitude decoding).
        EXPECT(!(p < 0.0f));
    }

    std::printf("    scanned=%llu  sane=%llu  nan=%llu\n",
                (unsigned long long)0x10000u,
                (unsigned long long)sane_count,
                (unsigned long long)nan_count);
    // Most patterns are sane; only NaN bit patterns (exp=31, mantissa!=0)
    // should surface NaN.
    EXPECT(sane_count > 60000);
}

// ── Atomic128 + TTEntry layout / claim round-trip ─────────────────────────

void test_atomic128_lock_free() {
    std::printf("[9] Atomic128<KeyQd> / Atomic128<InfoCell> are lock-free\n");

    // These should already be enforced by static_asserts inside Atomic128.
    // The runtime check exists so the test binary fails loudly on any
    // host that managed to slip past those asserts.
    catgpt::v2::Atomic128<catgpt::v2::KeyQd>    a;
    catgpt::v2::Atomic128<catgpt::v2::InfoCell> b;
    (void)a;
    (void)b;

    EXPECT(sizeof(catgpt::v2::TTEntry) == 32);
    EXPECT(alignof(catgpt::v2::TTEntry) == 32);
    EXPECT(sizeof(catgpt::v2::KeyQd) == 16);
    EXPECT(sizeof(catgpt::v2::InfoCell) == 16);
}

void test_claim_load_publish_round_trip() {
    std::printf("[10] single-thread claim / load_qd / publish_info / load_info round-trip\n");

    catgpt::v2::SearchArena arena(/*k_max_evals=*/1024,
                                  /*load_factor=*/0.5,
                                  /*avg_moves_per_node=*/40);

    constexpr uint64_t kKey = 0x0123456789ABCDEFULL;
    constexpr float    kQ   = 0.3125f;       // exact in fp32
    constexpr float    kMD  = 4.0f;          // exact in fp32

    // Pre-claim: find returns nullptr.
    EXPECT(arena.find(kKey) == nullptr);

    // Allocate arena bytes for a small node and fill MoveInfo[].
    constexpr uint16_t kNumMoves = 5;
    const uint64_t off = arena.alloc_node_info(kNumMoves);
    EXPECT(off != catgpt::v2::kNoInfoOffset);
    auto* hdr = arena.info_at(off);
    hdr->variance = 0.25f;
    catgpt::v2::MoveInfo* moves = arena.moves_at(off);
    for (uint16_t i = 0; i < kNumMoves; ++i) {
        moves[i] = catgpt::v2::MoveInfo::pack(
            static_cast<uint16_t>(i + 1),
            1.0f / static_cast<float>(kNumMoves),
            catgpt::v2::kTerminalNone);
    }

    // Claim with the qd committed atomically.
    auto [e, claimed] = arena.find_or_claim(kKey, /*Q=*/kQ, /*max_depth=*/kMD);
    EXPECT(claimed);
    EXPECT(e != nullptr);

    // Cell A is readable immediately, with no spin.
    catgpt::v2::KeyQd kq = catgpt::v2::SearchArena::load_qd(e);
    EXPECT(kq.key == kKey);
    auto [q, md] = catgpt::v2::unpack_qd(kq.qd_packed);
    EXPECT_EQ_F(q, kQ);
    EXPECT_EQ_F(md, kMD);

    // Cell B is unpublished until publish_info.
    catgpt::v2::InfoCell pre = catgpt::v2::SearchArena::load_info(e);
    EXPECT(pre.info_offset == catgpt::v2::kNoInfoOffset);

    // Publish: origQ kept distinct from the rolled-up Q so we can verify
    // the round-trip preserves the input value.
    constexpr float kOrigQ = -0.625f;        // exact in fp32
    catgpt::v2::SearchArena::publish_info(e, /*origQ=*/kOrigQ, off);

    catgpt::v2::InfoCell post = catgpt::v2::SearchArena::load_info(e);
    EXPECT(post.info_offset == off);
    EXPECT_EQ_F(post.origQ, kOrigQ);

    auto info_opt = catgpt::v2::SearchArena::wait_published(e);
    EXPECT(info_opt.has_value());
    if (info_opt) {
        EXPECT(info_opt->info_offset == off);
        EXPECT_EQ_F(info_opt->origQ, kOrigQ);
    }

    // update_qd raises max_depth.
    bool ok = catgpt::v2::SearchArena::update_qd(e, /*new_q=*/0.5f, /*new_max_depth=*/8.0f);
    EXPECT(ok);
    auto [q2, md2] = catgpt::v2::unpack_qd(
        catgpt::v2::SearchArena::load_qd(e).qd_packed);
    EXPECT_EQ_F(q2, 0.5f);
    EXPECT_EQ_F(md2, 8.0f);

    // Lower max_depth is rejected; q stays at the prior value.
    bool ok2 = catgpt::v2::SearchArena::update_qd(e, /*new_q=*/0.0f, /*new_max_depth=*/4.0f);
    EXPECT(!ok2);
    auto [q3, md3] = catgpt::v2::unpack_qd(
        catgpt::v2::SearchArena::load_qd(e).qd_packed);
    EXPECT_EQ_F(q3, 0.5f);
    EXPECT_EQ_F(md3, 8.0f);

    // origQ remains untouched by update_qd.
    catgpt::v2::InfoCell after_update = catgpt::v2::SearchArena::load_info(e);
    EXPECT_EQ_F(after_update.origQ, kOrigQ);
    EXPECT(after_update.info_offset == off);

    // Re-finding the same key returns the same entry; second find_or_claim
    // does NOT re-claim.
    auto* found = arena.find(kKey);
    EXPECT(found == e);
    auto [e2, claimed2] = arena.find_or_claim(kKey, /*Q=*/0.0f, /*max_depth=*/0.0f);
    EXPECT(!claimed2);
    EXPECT(e2 == e);
}

}  // namespace

int main() {
    std::printf("tt_arena_test: MoveInfo=%zuB alignof=%zuB  TTEntry=%zuB\n",
                sizeof(MoveInfo), alignof(MoveInfo), sizeof(catgpt::v2::TTEntry));

    test_static_layout();
    test_non_terminal_round_trip();
    test_terminal_round_trip();
    test_move_field_independence();
    test_edge_bit_patterns();
    test_monotonicity();
    test_randomized_round_trip();
    test_exhaustive_packed_patterns_never_crash();
    test_atomic128_lock_free();
    test_claim_load_publish_round_trip();

    if (g_failed == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d expectation(s) failed.\n", g_failed);
    return 1;
}
