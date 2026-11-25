#include "moe/trace.hpp"

#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace moe {

namespace {

constexpr size_t kHeaderBytes = 8 + 4 * 8 + 8 * 2; // magic + 8 x u32 + 2 x u64

template <typename T> void put(uint8_t *& p, T v) {
    std::memcpy(p, &v, sizeof(T));
    p += sizeof(T);
}
template <typename T> T get(const uint8_t *& p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}

struct FileCloser {
    void operator()(FILE * f) const { if (f) std::fclose(f); }
};
using FilePtr = std::unique_ptr<FILE, FileCloser>;

void validate_header(const TraceHeader & h) {
    if (h.k == 0 || h.k > (uint32_t) kMaxK) throw std::runtime_error("trace: invalid k=" + std::to_string(h.k));
    if (h.n_expert == 0 || h.n_expert > 256) {
        throw std::runtime_error("trace: n_expert=" + std::to_string(h.n_expert) + " does not fit the u8 expert ids");
    }
    h.layout.validate();
}

} // namespace

TraceWriter::TraceWriter(const std::string & path, const TraceHeader & header) : h_(header) {
    validate_header(h_);
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("trace: cannot open " + path + " for writing");
    uint8_t hdr[kHeaderBytes];
    uint8_t * p = hdr;
    std::memcpy(p, kTraceMagic, 8);
    p += 8;
    put<uint32_t>(p, h_.version);
    put<uint32_t>(p, h_.flags);
    put<uint32_t>(p, h_.n_expert);
    put<uint32_t>(p, h_.k);
    put<uint32_t>(p, (uint32_t) h_.layout.n_phys);
    put<uint32_t>(p, (uint32_t) h_.layout.core_start);
    put<uint32_t>(p, (uint32_t) h_.layout.core_end);
    put<uint32_t>(p, (uint32_t) h_.layout.n_passes);
    put<uint64_t>(p, h_.model_size);
    put<uint64_t>(p, h_.model_hash);
    if (std::fwrite(hdr, 1, kHeaderBytes, f_) != kHeaderBytes) throw std::runtime_error("trace: header write failed");
    buf_.resize(h_.record_size());
}

TraceWriter::~TraceWriter() {
    try { close(); } catch (...) {}
}

void TraceWriter::write(const TraceRecord & r) {
    if (!f_) throw std::runtime_error("trace: write after close");
    uint8_t * p = buf_.data();
    put<uint32_t>(p, r.seq);
    put<uint32_t>(p, r.pos);
    put<int32_t>(p, r.token);
    put<uint8_t>(p, r.slot);
    put<uint8_t>(p, static_cast<uint8_t>(r.phase));
    put<uint16_t>(p, 0);
    for (uint32_t i = 0; i < h_.k; ++i) put<uint8_t>(p, r.ids[i]);
    if (h_.has_weights()) {
        for (uint32_t i = 0; i < h_.k; ++i) put<float>(p, r.weights[i]);
    }
    if (std::fwrite(buf_.data(), 1, buf_.size(), f_) != buf_.size()) throw std::runtime_error("trace: record write failed");
    ++n_;
}

void TraceWriter::close() {
    if (!f_) return;
    const bool ok = std::fclose(f_) == 0;
    f_ = nullptr;
    if (!ok) throw std::runtime_error("trace: close failed");
}

Trace read_trace(const std::string & path) {
    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) throw std::runtime_error("trace: cannot open " + path);
    const auto file_size = std::filesystem::file_size(path);
    if (file_size < kHeaderBytes) throw std::runtime_error("trace: " + path + " is too short to hold a header");

    uint8_t hdr[kHeaderBytes];
    if (std::fread(hdr, 1, kHeaderBytes, f.get()) != kHeaderBytes) throw std::runtime_error("trace: header read failed");
    if (std::memcmp(hdr, kTraceMagic, 8) != 0) throw std::runtime_error("trace: " + path + " has a wrong magic (not a moe-loop trace)");
    const uint8_t * p = hdr + 8;
    Trace t;
    auto & h = t.header;
    h.version = get<uint32_t>(p);
    if (h.version != kTraceVersion) {
        throw std::runtime_error("trace: unsupported version " + std::to_string(h.version) + " (expected " +
                                 std::to_string(kTraceVersion) + ")");
    }
    h.flags = get<uint32_t>(p);
    h.n_expert = get<uint32_t>(p);
    h.k = get<uint32_t>(p);
    h.layout.n_phys = (int) get<uint32_t>(p);
    h.layout.core_start = (int) get<uint32_t>(p);
    h.layout.core_end = (int) get<uint32_t>(p);
    h.layout.n_passes = (int) get<uint32_t>(p);
    h.model_size = get<uint64_t>(p);
    h.model_hash = get<uint64_t>(p);
    validate_header(h);

    const size_t rs = h.record_size();
    const uint64_t body = file_size - kHeaderBytes;
    if (body % rs != 0) {
        throw std::runtime_error("trace: " + path + " is truncated (" + std::to_string(body % rs) +
                                 " trailing bytes do not form a whole record)");
    }
    const uint64_t n = body / rs;
    t.records.resize(n);
    std::vector<uint8_t> buf(rs * 4096);
    uint64_t done = 0;
    while (done < n) {
        const uint64_t chunk = std::min<uint64_t>(4096, n - done);
        if (std::fread(buf.data(), rs, chunk, f.get()) != chunk) throw std::runtime_error("trace: record read failed");
        for (uint64_t i = 0; i < chunk; ++i) {
            const uint8_t * q = buf.data() + i * rs;
            auto & r = t.records[done + i];
            r.seq = get<uint32_t>(q);
            r.pos = get<uint32_t>(q);
            r.token = get<int32_t>(q);
            r.slot = get<uint8_t>(q);
            const uint8_t phase = get<uint8_t>(q);
            if (phase > 1) throw std::runtime_error("trace: invalid phase byte in record " + std::to_string(done + i));
            r.phase = static_cast<Phase>(phase);
            q += 2;
            for (uint32_t j = 0; j < h.k; ++j) r.ids[j] = get<uint8_t>(q);
            if (h.has_weights()) {
                for (uint32_t j = 0; j < h.k; ++j) r.weights[j] = get<float>(q);
            }
        }
        done += chunk;
    }
    return t;
}

Fingerprint model_fingerprint(const std::string & path, size_t n_bytes) {
    Fingerprint fp;
    fp.size = std::filesystem::file_size(path);
    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> buf(1u << 20);
    uint64_t h = 1469598103934665603ull;
    size_t left = std::min<uint64_t>(n_bytes, fp.size);
    while (left > 0) {
        const size_t want = std::min(left, buf.size());
        const size_t got = std::fread(buf.data(), 1, want, f.get());
        if (got == 0) break;
        for (size_t i = 0; i < got; ++i) {
            h ^= buf[i];
            h *= 1099511628211ull;
        }
        left -= got;
    }
    fp.hash = h;
    return fp;
}

} // namespace moe
