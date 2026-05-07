/**
 * .network file loader: a packed bundle of per-bucket TensorRT engines.
 *
 * Each .network is built by `scripts/pack_network.py`. It contains N engine
 * blobs, one per batch-size bucket, each tuned for exactly that shape (single
 * profile, min == opt == max). The C++ side deserializes one ICudaEngine per
 * bucket from these blobs, avoiding the kernel-selection compromise that a
 * combined multi-profile engine forces on TRT.
 *
 * Format (all little-endian, no padding) — must match scripts/pack_network.py:
 *
 *   offset  size       field
 *   ------  ----       -----
 *   0       16         magic    = "CATGPT_NETWORK\0\0"
 *   16      4          version  = uint32 (1)
 *   20      4          num_engines = uint32 (N)
 *   24      N * 20     TOC entries:
 *                        uint32 bucket_size
 *                        uint64 offset      // byte offset from start of file
 *                        uint64 size        // engine blob size in bytes
 *   24+N*20 ...        concatenated engine blobs (in TOC order)
 */

#ifndef CATGPT_ENGINE_NETWORK_FILE_HPP
#define CATGPT_ENGINE_NETWORK_FILE_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace catgpt {

class NetworkFile {
public:
    static constexpr std::size_t kMagicSize = 16;
    static constexpr char kMagic[kMagicSize + 1] = "CATGPT_NETWORK\0\0";
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::size_t kHeaderSize = 24;          // magic + version + num_engines
    static constexpr std::size_t kTocEntrySize = 4 + 8 + 8; // bucket + offset + size

    struct SubEngine {
        int bucket_size;
        std::span<const std::byte> blob;
    };

    explicit NetworkFile(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error(
                std::format("NetworkFile: failed to open {}", path.string()));
        }
        const auto file_size = static_cast<std::size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        if (file_size < kHeaderSize) {
            throw std::runtime_error(std::format(
                "NetworkFile: {} is too small ({} bytes < header {})",
                path.string(), file_size, kHeaderSize));
        }

        data_.resize(file_size);
        if (!file.read(reinterpret_cast<char*>(data_.data()),
                       static_cast<std::streamsize>(file_size))) {
            throw std::runtime_error(
                std::format("NetworkFile: short read from {}", path.string()));
        }

        // Validate magic.
        if (std::memcmp(data_.data(), kMagic, kMagicSize) != 0) {
            throw std::runtime_error(std::format(
                "NetworkFile: bad magic in {} (not a CATGPT .network file)",
                path.string()));
        }

        // Version + N.
        std::uint32_t version = read_u32(kMagicSize);
        std::uint32_t n = read_u32(kMagicSize + 4);
        if (version != kVersion) {
            throw std::runtime_error(std::format(
                "NetworkFile: unsupported version {} in {} (loader expects {})",
                version, path.string(), kVersion));
        }
        if (n == 0) {
            throw std::runtime_error(std::format(
                "NetworkFile: {} has zero engines", path.string()));
        }

        const std::size_t toc_bytes = static_cast<std::size_t>(n) * kTocEntrySize;
        if (file_size < kHeaderSize + toc_bytes) {
            throw std::runtime_error(std::format(
                "NetworkFile: {} truncated TOC (file {} bytes, header+TOC {})",
                path.string(), file_size, kHeaderSize + toc_bytes));
        }

        sub_engines_.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::size_t toc_off = kHeaderSize + i * kTocEntrySize;
            const std::uint32_t bucket = read_u32(toc_off);
            const std::uint64_t blob_off = read_u64(toc_off + 4);
            const std::uint64_t blob_sz = read_u64(toc_off + 12);

            if (bucket == 0) {
                throw std::runtime_error(std::format(
                    "NetworkFile: {} has zero bucket size at TOC index {}",
                    path.string(), i));
            }
            if (blob_off + blob_sz > file_size || blob_off < kHeaderSize + toc_bytes) {
                throw std::runtime_error(std::format(
                    "NetworkFile: {} TOC entry {} (bucket {}) blob range "
                    "[{}, {}) out of file bounds [{}, {})",
                    path.string(), i, bucket,
                    blob_off, blob_off + blob_sz,
                    kHeaderSize + toc_bytes, file_size));
            }

            sub_engines_.push_back(SubEngine{
                .bucket_size = static_cast<int>(bucket),
                .blob = std::span<const std::byte>(data_.data() + blob_off, blob_sz),
            });
        }
    }

    NetworkFile(const NetworkFile&) = delete;
    NetworkFile& operator=(const NetworkFile&) = delete;
    NetworkFile(NetworkFile&&) = default;
    NetworkFile& operator=(NetworkFile&&) = default;

    [[nodiscard]] const std::vector<SubEngine>& sub_engines() const noexcept {
        return sub_engines_;
    }

    [[nodiscard]] std::size_t file_size() const noexcept { return data_.size(); }

private:
    std::uint32_t read_u32(std::size_t off) const {
        std::uint32_t v;
        std::memcpy(&v, data_.data() + off, sizeof(v));
        return v;
    }
    std::uint64_t read_u64(std::size_t off) const {
        std::uint64_t v;
        std::memcpy(&v, data_.data() + off, sizeof(v));
        return v;
    }

    std::vector<std::byte> data_;          // Whole file in memory (blobs are spans into this).
    std::vector<SubEngine> sub_engines_;   // In TOC order.
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_NETWORK_FILE_HPP
