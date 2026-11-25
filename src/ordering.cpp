#include "moe/ordering.hpp"

#include "moe/stats.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace moe {

uint64_t path_weight(const CoMatrix & c, const std::vector<int> & order) {
    uint64_t w = 0;
    for (size_t i = 1; i < order.size(); ++i) w += c[order[i - 1]][order[i]];
    return w;
}

double mean_runs(const std::vector<std::vector<uint8_t>> & selections, const std::vector<int> & order) {
    if (selections.empty()) return 0;
    std::vector<int> pos_of(order.size());
    for (size_t i = 0; i < order.size(); ++i) pos_of[order[i]] = (int) i;
    double total = 0;
    std::vector<int> pos;
    for (const auto & s : selections) {
        pos.clear();
        for (uint8_t e : s) pos.push_back(pos_of[e]);
        total += count_runs(pos);
    }
    return total / (double) selections.size();
}

std::vector<int> greedy_chain(const CoMatrix & c, int start) {
    const int n = (int) c.size();
    std::vector<char> used(n, 0);
    std::vector<int> order{start};
    used[start] = 1;
    // grow at whichever end has the strongest unused neighbour
    while ((int) order.size() < n) {
        int best = -1;
        bool at_back = true;
        uint64_t best_w = 0;
        for (int v = 0; v < n; ++v) {
            if (used[v]) continue;
            const uint64_t wb = c[order.back()][v], wf = c[order.front()][v];
            if (best < 0 || wb > best_w) { best = v; best_w = wb; at_back = true; }
            if (wf > best_w) { best = v; best_w = wf; at_back = false; }
        }
        used[best] = 1;
        if (at_back) order.push_back(best);
        else order.insert(order.begin(), best);
    }
    return order;
}

int two_opt(const CoMatrix & c, std::vector<int> & order) {
    const int n = (int) order.size();
    int moves = 0;
    bool improved = true;
    auto w = [&](int a, int b) -> int64_t { return (int64_t) c[order[a]][order[b]]; };
    while (improved) {
        improved = false;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                // reverse order[i..j]; edges (i-1,i) and (j,j+1) become (i-1,j) and (i,j+1)
                int64_t before = 0, after = 0;
                if (i > 0) { before += w(i - 1, i); after += w(i - 1, j); }
                if (j < n - 1) { before += w(j, j + 1); after += w(i, j + 1); }
                if (after > before) {
                    std::reverse(order.begin() + i, order.begin() + j + 1);
                    ++moves;
                    improved = true;
                }
            }
        }
    }
    return moves;
}

std::vector<int> cluster_order(const CoMatrix & c) {
    const int n = (int) c.size();
    if (n == 0) return {};
    std::vector<int> best;
    uint64_t best_w = 0;
    for (int s = 0; s < n; ++s) {
        auto o = greedy_chain(c, s);
        two_opt(c, o);
        const uint64_t w = path_weight(c, o);
        if (best.empty() || w > best_w) { best = std::move(o); best_w = w; }
    }
    return best;
}

std::vector<int> random_order(int n, std::mt19937_64 & rng) {
    std::vector<int> o(n);
    std::iota(o.begin(), o.end(), 0);
    std::shuffle(o.begin(), o.end(), rng);
    return o;
}

} // namespace moe
