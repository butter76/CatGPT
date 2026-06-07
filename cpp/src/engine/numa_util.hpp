/**
 * Dependency-free NUMA topology, thread-pinning, and large-buffer
 * allocation helpers.
 *
 * Targets the TCEC GPU box (2-socket EPYC, 8x RTX 5090, Ubuntu 22.04)
 * but degrades gracefully to a single-node, no-op topology on any
 * machine whose `/sys` interface is missing or unreadable (e.g. a local
 * dev box, a container). Uses only raw Linux syscalls + `/sys` parsing:
 * no libnuma, no hwloc, no extra link dependencies.
 *
 * What it provides:
 *   - `detect_topology()` / `host_topology()`: NUMA node -> CPU map,
 *     built from /sys, distinguishing physical-core "primary" hyperthreads
 *     from their SMT siblings.
 *   - `pin_this_thread_to_cpu` / `pin_this_thread_to_node`: affinity via
 *     `sched_setaffinity`.
 *   - `alloc_interleaved` / `parallel_prefault` / `free_mapped`: mmap a
 *     huge buffer, request THP (madvise) + MPOL_INTERLEAVE placement
 *     across all NUMA nodes, then fault every page in parallel across all
 *     physical cores. Anonymous mmap pages fault in zeroed, so callers
 *     that need zeroed memory (e.g. the all-zero TT sentinels) get it for
 *     free without an extra memset.
 *   - `gpu_numa_node` (only when <cuda_runtime.h> is in scope): maps a
 *     CUDA device to the NUMA node its PCIe root complex hangs off.
 *
 * NUMA placement rationale: the shared structures this allocator is built
 * for (the TT + arena) are accessed by hash, i.e. uniformly at random by
 * every worker on every socket. The bandwidth-optimal policy for such a
 * globally-shared random-access table is MPOL_INTERLEAVE (round-robin
 * pages across all controllers), NOT first-touch-local. The parallel
 * prefault is purely to spread the page-fault cost across cores and
 * front-load it out of the search; the *placement* is decided by the
 * interleave policy, independent of which thread touches a page.
 */

#ifndef CATGPT_ENGINE_NUMA_UTIL_HPP
#define CATGPT_ENGINE_NUMA_UTIL_HPP

#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// MPOL_INTERLEAVE lives in <linux/mempolicy.h> / <numaif.h>; define it
// ourselves so we depend on neither (we call mbind via syscall() anyway).
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif

namespace catgpt::numa {

namespace fs = std::filesystem;

/**
 * Hierarchical CPU view of the host. All indices are OS logical-CPU ids
 * and OS NUMA-node ids unless otherwise noted.
 */
struct Topology {
    // OS node ids actually present (e.g. {0, 1}); not assumed contiguous.
    std::vector<int> node_ids;
    // node_cpus[i] = every logical CPU on node_ids[i] (primaries then
    // SMT siblings, in /sys order otherwise).
    std::vector<std::vector<int>> node_cpus;
    // node_phys_cpus[i] = one logical CPU per physical core on node_ids[i]
    // (the lowest-numbered sibling of each core).
    std::vector<std::vector<int>> node_phys_cpus;
    // sibling_of[cpu] = the SMT sibling logical CPU of `cpu`, or `cpu`
    // itself if it has no sibling / is unknown. Sized to max_cpu + 1.
    std::vector<int> sibling_of;

    [[nodiscard]] int num_nodes() const noexcept {
        return static_cast<int>(node_ids.size());
    }

    // Map an OS node id to its index in node_ids, or -1 if absent.
    [[nodiscard]] int index_of_node(int os_node) const noexcept {
        for (int i = 0; i < num_nodes(); ++i) {
            if (node_ids[i] == os_node) return i;
        }
        return -1;
    }
};

// ── /sys parsing helpers ──────────────────────────────────────────────

/**
 * Parse a Linux cpulist / nodelist string ("0-15,32,40-47") into a sorted
 * vector of ids. Tolerant of trailing whitespace / empty input.
 */
inline std::vector<int> parse_id_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // Trim.
        while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back()))) tok.pop_back();
        std::size_t b = 0;
        while (b < tok.size() && std::isspace(static_cast<unsigned char>(tok[b]))) ++b;
        tok = tok.substr(b);
        if (tok.empty()) continue;
        const auto dash = tok.find('-');
        try {
            if (dash == std::string::npos) {
                out.push_back(std::stoi(tok));
            } else {
                const int lo = std::stoi(tok.substr(0, dash));
                const int hi = std::stoi(tok.substr(dash + 1));
                for (int v = lo; v <= hi; ++v) out.push_back(v);
            }
        } catch (...) {
            // Skip malformed token.
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

inline bool read_file_trimmed(const fs::path& p, std::string& out) {
    std::ifstream f(p);
    if (!f) return false;
    std::getline(f, out);
    return true;
}

/**
 * Build a single-node fallback topology covering every online CPU with no
 * SMT-sibling information. Used when /sys is unreadable.
 */
inline Topology fallback_topology() {
    Topology t;
    unsigned hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 1;
    t.node_ids = {0};
    t.node_cpus.resize(1);
    t.node_phys_cpus.resize(1);
    t.sibling_of.resize(hc);
    for (unsigned c = 0; c < hc; ++c) {
        t.node_cpus[0].push_back(static_cast<int>(c));
        t.node_phys_cpus[0].push_back(static_cast<int>(c));
        t.sibling_of[c] = static_cast<int>(c);
    }
    return t;
}

/**
 * Discover NUMA topology from /sys. Returns a single-node fallback if the
 * sysfs node hierarchy is missing.
 */
inline Topology detect_topology() {
    const fs::path node_root = "/sys/devices/system/node";
    std::error_code ec;
    if (!fs::exists(node_root, ec)) return fallback_topology();

    Topology t;
    int max_cpu = -1;

    // First pass: collect nodes + their cpu lists.
    std::vector<std::pair<int, std::vector<int>>> nodes;
    for (const auto& entry : fs::directory_iterator(node_root, ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.rfind("node", 0) != 0) continue;
        int node_id = -1;
        try { node_id = std::stoi(name.substr(4)); } catch (...) { continue; }
        std::string cpulist;
        if (!read_file_trimmed(entry.path() / "cpulist", cpulist)) continue;
        std::vector<int> cpus = parse_id_list(cpulist);
        if (cpus.empty()) continue;
        for (int c : cpus) max_cpu = std::max(max_cpu, c);
        nodes.emplace_back(node_id, std::move(cpus));
    }

    if (nodes.empty()) return fallback_topology();
    std::sort(nodes.begin(), nodes.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    t.sibling_of.assign(static_cast<std::size_t>(max_cpu) + 1, 0);
    for (int c = 0; c <= max_cpu; ++c) t.sibling_of[c] = c;

    for (auto& [node_id, cpus] : nodes) {
        t.node_ids.push_back(node_id);
        std::vector<int> phys;
        for (int c : cpus) {
            // thread_siblings_list: the SMT group this cpu belongs to.
            const fs::path sib = "/sys/devices/system/cpu/cpu" +
                                 std::to_string(c) + "/topology/thread_siblings_list";
            std::string sl;
            int primary = c;
            int sibling = c;
            if (read_file_trimmed(sib, sl)) {
                std::vector<int> group = parse_id_list(sl);
                if (!group.empty()) {
                    primary = group.front();  // lowest id in the SMT group
                    if (group.size() >= 2) {
                        // Pick the first group member != c as the sibling.
                        for (int g : group) {
                            if (g != c) { sibling = g; break; }
                        }
                    }
                }
            }
            if (c <= max_cpu) t.sibling_of[c] = sibling;
            if (primary == c) phys.push_back(c);  // c is its core's primary
        }
        std::sort(phys.begin(), phys.end());
        t.node_cpus.push_back(std::move(cpus));
        t.node_phys_cpus.push_back(std::move(phys));
    }

    // Guard against a degenerate parse (no physical cores identified).
    bool any_phys = false;
    for (const auto& v : t.node_phys_cpus) any_phys = any_phys || !v.empty();
    if (!any_phys) return fallback_topology();

    return t;
}

/**
 * Process-wide cached topology (scanned once on first use).
 */
inline const Topology& host_topology() {
    static const Topology t = detect_topology();
    return t;
}

// ── Thread pinning ────────────────────────────────────────────────────

/**
 * Pin the calling thread to a single logical CPU. Returns false on
 * failure (e.g. invalid cpu, restricted cgroup) without throwing — NUMA
 * pinning is a performance hint, never a correctness requirement.
 */
inline bool pin_this_thread_to_cpu(int cpu) {
    if (cpu < 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

/**
 * Pin the calling thread to the full CPU set of a NUMA node (by OS node
 * id). Looser than a single-cpu pin: lets the scheduler move the thread
 * within the node but never off it.
 */
inline bool pin_this_thread_to_node(const Topology& topo, int os_node) {
    const int idx = topo.index_of_node(os_node);
    if (idx < 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : topo.node_cpus[idx]) {
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    }
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// ── Large-buffer allocation (mmap + THP + interleave) ─────────────────

/**
 * mmap an anonymous private buffer and request THP backing. The mapping
 * is page-aligned and faults in zeroed. Returns nullptr on failure.
 */
inline void* alloc_mapped(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    // THP is set to `madvise` on the TCEC box, so this is required to get
    // 2 MiB pages for the giant TT/arena (TLB pressure on 200 GB+).
    ::madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

/**
 * Apply MPOL_INTERLEAVE across every NUMA node in `topo` to an existing
 * mapping. No-op (and harmless) on a single-node host. Best-effort: a
 * failure just leaves the kernel default (first-touch) policy in place.
 */
inline void interleave_mbind(void* base, std::size_t bytes, const Topology& topo) {
    if (base == nullptr || bytes == 0) return;
    if (topo.num_nodes() <= 1) return;

    int max_node = 0;
    for (int n : topo.node_ids) max_node = std::max(max_node, n);

    // nodemask is an array of unsigned long bitfields; bit k => node k.
    const std::size_t bits = static_cast<std::size_t>(max_node) + 1;
    const std::size_t words = (bits + (8 * sizeof(unsigned long) - 1)) /
                              (8 * sizeof(unsigned long));
    std::vector<unsigned long> mask(words, 0UL);
    for (int n : topo.node_ids) {
        mask[static_cast<std::size_t>(n) / (8 * sizeof(unsigned long))] |=
            (1UL << (static_cast<std::size_t>(n) % (8 * sizeof(unsigned long))));
    }

    // maxnode = number of bits in the mask (man mbind: includes the +1).
    const unsigned long maxnode = static_cast<unsigned long>(words * 8 * sizeof(unsigned long));
    ::syscall(SYS_mbind, base, static_cast<unsigned long>(bytes),
              MPOL_INTERLEAVE, mask.data(), maxnode, 0u);
}

/**
 * Fault every page of `[base, base+bytes)` in parallel across all
 * physical cores, each pinned to its core. Writes one zero byte per
 * system page (preserving the zeroed contents of an anonymous mapping)
 * so the page is committed under whatever mempolicy is in effect.
 *
 * Spreading the faults across both sockets' cores both parallelizes the
 * (otherwise multi-second) commit of a 200 GB+ buffer and, under
 * MPOL_INTERLEAVE, lets the kernel place pages round-robin as fast as the
 * aggregate memory bandwidth allows.
 */
inline void parallel_prefault(void* base, std::size_t bytes, const Topology& topo) {
    if (base == nullptr || bytes == 0) return;

    const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    char* p = static_cast<char*>(base);

    // Gather (cpu) workers: one per physical core across all nodes.
    std::vector<int> cpus;
    for (const auto& node : topo.node_phys_cpus) {
        for (int c : node) cpus.push_back(c);
    }
    if (cpus.empty()) {
        unsigned hc = std::thread::hardware_concurrency();
        for (unsigned i = 0; i < std::max(1u, hc); ++i) cpus.push_back(-1);
    }

    const std::size_t nthreads = cpus.size();
    const std::size_t chunk = (bytes + nthreads - 1) / nthreads;

    std::vector<std::thread> ts;
    ts.reserve(nthreads);
    for (std::size_t ti = 0; ti < nthreads; ++ti) {
        const std::size_t begin = ti * chunk;
        if (begin >= bytes) break;
        const std::size_t end = std::min(begin + chunk, bytes);
        const int cpu = cpus[ti];
        ts.emplace_back([p, begin, end, page, cpu] {
            if (cpu >= 0) pin_this_thread_to_cpu(cpu);
            for (std::size_t off = begin; off < end; off += page) {
                // Write 0 (matches the already-zero anonymous page) to
                // commit the page without altering its contents.
                reinterpret_cast<volatile char*>(p)[off] = 0;
            }
        });
    }
    for (auto& t : ts) t.join();
}

/**
 * Allocate `bytes`, request THP, interleave across all NUMA nodes, and
 * prefault in parallel. Returns nullptr on mmap failure. The returned
 * memory is zeroed.
 */
inline void* alloc_interleaved(std::size_t bytes, const Topology& topo) {
    void* p = alloc_mapped(bytes);
    if (p == nullptr) return nullptr;
    interleave_mbind(p, bytes, topo);
    parallel_prefault(p, bytes, topo);
    return p;
}

inline void free_mapped(void* p, std::size_t bytes) {
    if (p != nullptr && bytes != 0) ::munmap(p, bytes);
}

// ── CUDA device -> NUMA node (only when cuda_runtime is in scope) ──────

/**
 * Read the NUMA node a PCI device hangs off, from
 * /sys/bus/pci/devices/<bdf>/numa_node. Returns -1 if unknown.
 * `pci_bus_id` is the CUDA-style "0000:65:00.0" string (any case).
 */
inline int numa_node_for_pci_bus(std::string pci_bus_id) {
    std::transform(pci_bus_id.begin(), pci_bus_id.end(), pci_bus_id.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const fs::path p = "/sys/bus/pci/devices/" + pci_bus_id + "/numa_node";
    std::string s;
    if (!read_file_trimmed(p, s)) return -1;
    try {
        return std::stoi(s);
    } catch (...) {
        return -1;
    }
}

#if defined(CUDART_VERSION)
/**
 * Map a CUDA device to its local NUMA node (OS node id). Falls back to a
 * round-robin assignment across the topology's nodes when sysfs reports
 * no affinity (numa_node == -1, common in VMs / single-node hosts).
 */
inline int gpu_numa_node(int device_id, const Topology& topo) {
    auto round_robin = [&]() -> int {
        if (topo.num_nodes() <= 0) return 0;
        return topo.node_ids[static_cast<std::size_t>(device_id) % topo.node_ids.size()];
    };
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device_id) != cudaSuccess) {
        return round_robin();
    }
    const int n = numa_node_for_pci_bus(bus);
    if (n < 0 || topo.index_of_node(n) < 0) return round_robin();
    return n;
}
#endif  // CUDART_VERSION

}  // namespace catgpt::numa

#endif  // CATGPT_ENGINE_NUMA_UTIL_HPP
