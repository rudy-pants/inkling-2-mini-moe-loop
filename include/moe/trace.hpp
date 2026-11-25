// Binary trace of router decisions. Format: docs/trace-format.md.
#pragma once

#include "moe/layout.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static_assert(std::endian::native == std::endian::little, "trace format assumes a little-endian host");

namespace moe {

inline constexpr char kTraceMagic[8] = {'M', 'O', 'E', 'T', 'R', 'C', '\0', '\1'};
inline constexpr uint32_t kTraceVersion = 1;
inline constexpr int kMaxK = 16;

enum TraceFlags : uint32_t {
    TRACE_HAS_WEIGHTS = 1u << 0, // records carry the k normalized router weights (f32)
};

enum class Phase : uint8_t { Prefill = 0, Decode = 1 };

struct TraceHeader {
    uint32_t version  = kTraceVersion;
    uint32_t flags    = 0;
    uint32_t n_expert = 0;
    uint32_t k        = 0;
    LoopLayout layout;          // stored as 4 x u32
    uint64_t model_size = 0;    // bytes of the GGUF file
    uint64_t model_hash = 0;    // FNV-1a 64 of the GGUF header region, see model_fingerprint()

    bool has_weights() const { return flags & TRACE_HAS_WEIGHTS; }
    size_t record_size() const { return 16 + k + (has_weights() ? 4 * k : 0); }
};

// One router decision: which k experts token `pos` of sequence `seq` used at graph slot `slot`.
struct TraceRecord {
    uint32_t seq   = 0;
    uint32_t pos   = 0;
    int32_t  token = 0;  // input token at this position
    uint8_t  slot  = 0;
    Phase    phase = Phase::Prefill;
    std::array<uint8_t, kMaxK> ids{};
    std::array<float, kMaxK> weights{};
};

class TraceWriter {
public:
    TraceWriter(const std::string & path, const TraceHeader & header);
    ~TraceWriter();
    TraceWriter(const TraceWriter &) = delete;
    TraceWriter & operator=(const TraceWriter &) = delete;

    void write(const TraceRecord & r);
    void close();
    uint64_t n_records() const { return n_; }

private:
    FILE * f_ = nullptr;
    TraceHeader h_;
    std::vector<uint8_t> buf_;
    uint64_t n_ = 0;
};

struct Trace {
    TraceHeader header;
    std::vector<TraceRecord> records;
};

// Throws std::runtime_error on bad magic, unsupported version, invalid header or a truncated file.
Trace read_trace(const std::string & path);

// Cheap identity of a GGUF file: its size and an FNV-1a hash of the first `n_bytes` (header + tensor table).
struct Fingerprint {
    uint64_t size = 0;
    uint64_t hash = 0;
};
Fingerprint model_fingerprint(const std::string & path, size_t n_bytes = 16u << 20);

} // namespace moe
