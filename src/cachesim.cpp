#include "moe/cachesim.hpp"

#include <queue>
#include <stdexcept>
#include <tuple>

namespace moe {

namespace {

// Fenwick tree over access times, holding the size of each object at the time of its most recent access.
class Fenwick {
public:
    explicit Fenwick(size_t n) : t_(n + 1, 0) {}
    void add(size_t i, int64_t v) {
        for (++i; i < t_.size(); i += i & (~i + 1)) t_[i] += v;
    }
    int64_t prefix(size_t i) const {  // sum of [0, i)
        int64_t s = 0;
        for (; i > 0; i -= i & (~i + 1)) s += t_[i];
        return s;
    }

private:
    std::vector<int64_t> t_;
};

} // namespace

double HitStats::hit_rate(int kind) const {
    uint64_t t = 0, h = 0;
    for (size_t k = 0; k < bytes_total.size(); ++k) {
        if (kind >= 0 && (int) k != kind) continue;
        t += bytes_total[k];
        h += bytes_hit[k];
    }
    return t ? (double) h / (double) t : 0.0;
}

std::vector<uint64_t> stack_distances(const std::vector<Access> & stream, const std::vector<uint64_t> & object_size) {
    constexpr size_t kNever = std::numeric_limits<size_t>::max();
    std::vector<size_t> last(object_size.size(), kNever);
    std::vector<uint64_t> d(stream.size());
    Fenwick fw(stream.size());
    for (size_t t = 0; t < stream.size(); ++t) {
        const uint32_t o = stream[t].obj;
        if (o >= object_size.size()) throw std::out_of_range("access to unknown object");
        const auto sz = (int64_t) object_size[o];
        if (last[o] == kNever) {
            d[t] = kInfDistance;
        } else {
            d[t] = (uint64_t) (fw.prefix(t) - fw.prefix(last[o] + 1));
            fw.add(last[o], -sz);
        }
        fw.add(t, sz);
        last[o] = t;
    }
    return d;
}

std::vector<HitStats> lru_hits(const std::vector<Access> & stream, const std::vector<uint64_t> & object_size,
                               const std::vector<uint64_t> & distances, const std::vector<uint64_t> & capacities,
                               int n_kinds) {
    std::vector<HitStats> out(capacities.size());
    for (size_t c = 0; c < capacities.size(); ++c) {
        out[c].capacity = capacities[c];
        out[c].bytes_total.assign(n_kinds, 0);
        out[c].bytes_hit.assign(n_kinds, 0);
    }
    for (size_t t = 0; t < stream.size(); ++t) {
        const auto & a = stream[t];
        const uint64_t sz = object_size[a.obj];
        for (auto & s : out) {
            s.bytes_total[a.kind] += sz;
            if (distances[t] != kInfDistance && distances[t] + sz <= s.capacity) s.bytes_hit[a.kind] += sz;
        }
    }
    return out;
}

// GCC 13 reports a false -Wfree-nonheap-object inside std::priority_queue's vector when this is inlined at -O2.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif
HitStats belady_hits(const std::vector<Access> & stream, const std::vector<uint64_t> & object_size, uint64_t capacity,
                     int n_kinds) {
    constexpr size_t kNever = std::numeric_limits<size_t>::max();
    // next use of the object accessed at time t
    std::vector<size_t> next(stream.size(), kNever);
    {
        std::vector<size_t> upcoming(object_size.size(), kNever);
        for (size_t t = stream.size(); t-- > 0;) {
            next[t] = upcoming[stream[t].obj];
            upcoming[stream[t].obj] = t;
        }
    }
    HitStats s;
    s.capacity = capacity;
    s.bytes_total.assign(n_kinds, 0);
    s.bytes_hit.assign(n_kinds, 0);

    std::vector<char> resident(object_size.size(), 0);
    std::vector<size_t> res_next(object_size.size(), kNever);  // current next-use key of resident objects
    std::priority_queue<std::pair<size_t, uint32_t>> heap;      // (next use, obj), lazily invalidated
    uint64_t used = 0;

    for (size_t t = 0; t < stream.size(); ++t) {
        const uint32_t o = stream[t].obj;
        const uint64_t sz = object_size[o];
        s.bytes_total[stream[t].kind] += sz;
        if (resident[o]) {
            s.bytes_hit[stream[t].kind] += sz;
        } else {
            if (sz > capacity) continue;  // can never fit
            resident[o] = 1;
            used += sz;
        }
        res_next[o] = next[t];
        heap.emplace(next[t], o);
        while (used > capacity) {
            const auto [key, victim] = heap.top();
            heap.pop();
            if (!resident[victim] || res_next[victim] != key) continue;  // stale entry
            resident[victim] = 0;
            used -= object_size[victim];
        }
    }
    return s;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

} // namespace moe
