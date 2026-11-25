// Small statistics used by moe-analyze and moe-permute. Header-only and dependency-free for unit tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace moe {

// Number of experts two top-k selections share.
inline int overlap(const uint8_t * a, const uint8_t * b, int k) {
    int n = 0;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) n += a[i] == b[j];
    return n;
}

// Weighted overlap: sum over shared experts of min(weight in a, weight in b). 1.0 = identical weighted selection.
inline double weighted_overlap(const uint8_t * a, const float * wa, const uint8_t * b, const float * wb, int k) {
    double s = 0;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            if (a[i] == b[j]) s += std::min(wa[i], wb[j]);
    return s;
}

// Expected overlap of two independent uniformly random k-subsets of n experts: k*k/n.
inline double random_overlap(int k, int n) { return (double) k * k / n; }

// Number of maximal runs of consecutive integers among the given (distinct) positions.
// E.g. {3,4,5,9} -> 2. Positions are sorted in place.
inline int count_runs(std::vector<int> & pos) {
    if (pos.empty()) return 0;
    std::sort(pos.begin(), pos.end());
    int runs = 1;
    for (size_t i = 1; i < pos.size(); ++i) runs += pos[i] != pos[i - 1] + 1;
    return runs;
}

// Shannon entropy in bits of a count histogram.
inline double entropy_bits(const std::vector<uint64_t> & counts) {
    double total = 0;
    for (auto c : counts) total += (double) c;
    if (total <= 0) return 0;
    double h = 0;
    for (auto c : counts) {
        if (c == 0) continue;
        const double p = (double) c / total;
        h -= p * std::log2(p);
    }
    return h;
}

// Jensen-Shannon divergence (bits, in [0,1]) between two count histograms of equal length.
inline double js_divergence_bits(const std::vector<uint64_t> & a, const std::vector<uint64_t> & b) {
    double ta = 0, tb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ta += (double) a[i]; tb += (double) b[i]; }
    if (ta <= 0 || tb <= 0) return 0;
    double js = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double p = (double) a[i] / ta, q = (double) b[i] / tb, m = 0.5 * (p + q);
        if (p > 0) js += 0.5 * p * std::log2(p / m);
        if (q > 0) js += 0.5 * q * std::log2(q / m);
    }
    return js;
}

// Gini coefficient of a histogram (0 = perfectly even use, -> 1 = one expert gets everything).
inline double gini(std::vector<uint64_t> counts) {
    if (counts.empty()) return 0;
    std::sort(counts.begin(), counts.end());
    double cum = 0, weighted = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        cum += (double) counts[i];
        weighted += (double) (i + 1) * (double) counts[i];
    }
    if (cum <= 0) return 0;
    const double n = (double) counts.size();
    return (2.0 * weighted) / (n * cum) - (n + 1.0) / n;
}

} // namespace moe
