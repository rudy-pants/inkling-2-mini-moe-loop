// Object-level cache simulation for streams of whole-object reads.
//
// Every access reads one object (an expert slab, a layer's dense weights, a KV chunk) completely and sequentially.
// For such streams a fully associative LRU cache of capacity C keeps an object between two uses exactly when
//     D + size <= C
// where D is the number of distinct bytes of *other* objects touched in between (the byte stack distance):
// the object's own lines are all re-touched during its scan, so each of its lines has reuse distance D + size.
// This is exact for line-granular LRU on full-object scans and lets one pass give the hit rate at every capacity.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace moe {

struct Access {
    uint32_t obj;   // object id, index into object_size
    uint8_t kind;   // caller-defined stream category (weights, experts, KV, ...)
};

inline constexpr uint64_t kInfDistance = std::numeric_limits<uint64_t>::max();

// Byte stack distance of every access (kInfDistance for the first access of an object).
std::vector<uint64_t> stack_distances(const std::vector<Access> & stream, const std::vector<uint64_t> & object_size);

struct HitStats {
    uint64_t capacity = 0;
    std::vector<uint64_t> bytes_total;  // per kind
    std::vector<uint64_t> bytes_hit;    // per kind
    double hit_rate(int kind = -1) const;
};

// LRU hit statistics for several capacities from precomputed stack distances.
std::vector<HitStats> lru_hits(const std::vector<Access> & stream, const std::vector<uint64_t> & object_size,
                               const std::vector<uint64_t> & distances, const std::vector<uint64_t> & capacities,
                               int n_kinds);

// Offline size-aware Belady: on a miss the object is inserted, then objects whose next use is furthest away are
// evicted until everything fits (this may evict the new object itself = bypass). A strong upper-bound heuristic;
// exact optimality is NP-hard with variable object sizes.
HitStats belady_hits(const std::vector<Access> & stream, const std::vector<uint64_t> & object_size, uint64_t capacity,
                     int n_kinds);

} // namespace moe
