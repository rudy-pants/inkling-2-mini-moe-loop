#include "moe/gguf_permute.hpp"

#include "moe/gguf_layout.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace moe {

namespace {

struct Fd {
    int fd = -1;
    Fd(const std::string & path, int flags, mode_t mode = 0644) : fd(::open(path.c_str(), flags, mode)) {
        if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }
    ~Fd() { if (fd >= 0) ::close(fd); }
    Fd(const Fd &) = delete;
    Fd & operator=(const Fd &) = delete;
};

void pread_all(int fd, void * buf, size_t n, uint64_t off) {
    auto * p = static_cast<uint8_t *>(buf);
    while (n) {
        const ssize_t r = ::pread(fd, p, n, (off_t) off);
        if (r <= 0) throw std::runtime_error(std::string("read failed: ") + (r < 0 ? std::strerror(errno) : "unexpected end of file"));
        p += r;
        n -= (size_t) r;
        off += (uint64_t) r;
    }
}

void write_all(int fd, const void * buf, size_t n) {
    const auto * p = static_cast<const uint8_t *>(buf);
    while (n) {
        const ssize_t w = ::write(fd, p, n);
        if (w <= 0) throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
        p += w;
        n -= (size_t) w;
    }
}

// Copies [off, off + n) of `in` to the current position of `out`.
void copy_range(int in, int out, uint64_t off, uint64_t n, std::vector<uint8_t> & buf) {
    while (n) {
        const size_t c = (size_t) std::min<uint64_t>(n, buf.size());
        pread_all(in, buf.data(), c, off);
        write_all(out, buf.data(), c);
        off += c;
        n -= c;
    }
}

int n_expert_of(const GgufModel & m) {
    const auto arch = m.kv_str("general.architecture");
    if (!arch) throw std::runtime_error(m.path + ": no general.architecture");
    const auto n = m.kv_int(*arch + ".expert_count");
    if (!n || *n <= 0) throw std::runtime_error(m.path + ": no " + *arch + ".expert_count");
    return (int) *n;
}

// Tensors sorted by file offset, with their expert classification resolved and validated.
struct Plan {
    const TensorInfo * t;
    ExpertTensorClass cls;
    uint64_t chunk = 0;  // bytes per expert, 0 for tensors copied verbatim
};

std::vector<Plan> make_plan(const GgufModel & m, const Permutation & perm, int n_expert) {
    if (perm.n_expert != n_expert) {
        throw std::runtime_error("permutation is for " + std::to_string(perm.n_expert) + " experts, model has " +
                                 std::to_string(n_expert));
    }
    perm.validate();
    std::vector<Plan> plan;
    for (const auto & t : m.tensors) {
        Plan p{&t, classify_tensor(t.name), 0};
        if (p.cls.kind != ExpertTensorKind::None) {
            if (p.cls.layer < 0 || p.cls.layer >= (int) perm.new_to_old.size()) {
                throw std::runtime_error("no permutation row for layer " + std::to_string(p.cls.layer) + " (" + t.name + ")");
            }
            p.chunk = expert_chunk_bytes(t, n_expert);
        }
        plan.push_back(p);
    }
    std::sort(plan.begin(), plan.end(), [](const Plan & a, const Plan & b) { return a.t->offset < b.t->offset; });
    for (size_t i = 1; i < plan.size(); ++i) {
        if (plan[i - 1].t->offset + plan[i - 1].t->size > plan[i].t->offset) {
            throw std::runtime_error("overlapping tensors " + plan[i - 1].t->name + " and " + plan[i].t->name);
        }
    }
    return plan;
}

} // namespace

PermuteReport permute_gguf(const std::string & in, const std::string & out, const Permutation & perm) {
    if (std::filesystem::exists(out)) throw std::runtime_error("refusing to overwrite existing file " + out);
    const auto m = open_gguf(in);
    const int n_expert = n_expert_of(m);
    const auto plan = make_plan(m, perm, n_expert);

    Fd fi(in, O_RDONLY);
    Fd fo(out, O_WRONLY | O_CREAT | O_EXCL);
    std::vector<uint8_t> buf(64u << 20);
    PermuteReport rep;
    rep.bytes_total = m.file_size;
    try {
        uint64_t cur = 0;
        for (const auto & p : plan) {
            copy_range(fi.fd, fo.fd, cur, p.t->offset - cur, buf);  // header / padding before this tensor
            if (p.chunk == 0) {
                copy_range(fi.fd, fo.fd, p.t->offset, p.t->size, buf);
                ++rep.tensors_copied;
            } else {
                const auto & row = perm.new_to_old[p.cls.layer];
                std::vector<uint8_t> chunk(p.chunk);
                for (int i = 0; i < n_expert; ++i) {
                    pread_all(fi.fd, chunk.data(), p.chunk, p.t->offset + (uint64_t) row[i] * p.chunk);
                    write_all(fo.fd, chunk.data(), p.chunk);
                }
                ++rep.tensors_permuted;
                rep.bytes_permuted += p.t->size;
                switch (p.cls.kind) {
                    case ExpertTensorKind::RouterWeight: ++rep.router_weights; break;
                    case ExpertTensorKind::RouterBias: ++rep.router_biases; break;
                    case ExpertTensorKind::ExpertWeight: ++rep.expert_weights; break;
                    case ExpertTensorKind::ExpertScale: ++rep.expert_scales; break;
                    default: break;
                }
            }
            cur = p.t->offset + p.t->size;
        }
        copy_range(fi.fd, fo.fd, cur, m.file_size - cur, buf);  // trailing bytes
        if (::fsync(fo.fd) != 0) throw std::runtime_error(std::string("fsync failed: ") + std::strerror(errno));
    } catch (...) {
        ::unlink(out.c_str());  // never leave a half-written model behind
        throw;
    }
    return rep;
}

VerifyReport verify_permuted(const std::string & in, const std::string & out, const Permutation & perm) {
    VerifyReport v;
    const auto m = open_gguf(in);
    const int n_expert = n_expert_of(m);
    const auto plan = make_plan(m, perm, n_expert);
    if (std::filesystem::file_size(out) != m.file_size) {
        v.first_error = "file sizes differ";
        return v;
    }
    Fd fi(in, O_RDONLY);
    Fd fo(out, O_RDONLY);
    std::vector<uint8_t> a(16u << 20), b(16u << 20);
    auto same_range = [&](uint64_t off_in, uint64_t off_out, uint64_t n) {
        while (n) {
            const size_t c = (size_t) std::min<uint64_t>(n, a.size());
            pread_all(fi.fd, a.data(), c, off_in);
            pread_all(fo.fd, b.data(), c, off_out);
            if (std::memcmp(a.data(), b.data(), c) != 0) return false;
            off_in += c;
            off_out += c;
            n -= c;
        }
        return true;
    };
    uint64_t cur = 0;
    for (const auto & p : plan) {
        if (!same_range(cur, cur, p.t->offset - cur)) {
            v.first_error = "bytes before " + p.t->name + " differ";
            return v;
        }
        if (p.chunk == 0) {
            if (!same_range(p.t->offset, p.t->offset, p.t->size)) {
                v.first_error = "non-expert tensor " + p.t->name + " differs";
                return v;
            }
        } else {
            const auto & row = perm.new_to_old[p.cls.layer];
            for (int i = 0; i < n_expert; ++i) {
                if (!same_range(p.t->offset + (uint64_t) row[i] * p.chunk, p.t->offset + (uint64_t) i * p.chunk, p.chunk)) {
                    v.first_error = "expert tensor " + p.t->name + " position " + std::to_string(i) +
                                    " does not hold original expert " + std::to_string(row[i]);
                    return v;
                }
            }
        }
        ++v.tensors_checked;
        cur = p.t->offset + p.t->size;
    }
    if (!same_range(cur, cur, m.file_size - cur)) {
        v.first_error = "trailing bytes differ";
        return v;
    }
    v.ok = true;
    return v;
}

} // namespace moe
