#include "moe/cachesim.hpp"
#include "moe/layout.hpp"
#include "moe/ordering.hpp"
#include "moe/stats.hpp"

#include "doctest/doctest.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <set>

using doctest::Approx;

TEST_CASE("overlap and weighted overlap") {
    const uint8_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint8_t b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    const uint8_t c[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    const uint8_t d[8] = {1, 2, 3, 4, 20, 21, 22, 23};
    CHECK(moe::overlap(a, b, 8) == 8);
    CHECK(moe::overlap(a, c, 8) == 0);
    CHECK(moe::overlap(a, d, 8) == 4);
    const float w[8] = {0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};
    CHECK(moe::weighted_overlap(a, w, b, w, 8) == Approx(1.0));
    CHECK(moe::weighted_overlap(a, w, d, w, 8) == Approx(0.5));
    CHECK(moe::random_overlap(8, 128) == Approx(0.5));
}

TEST_CASE("uniform random routing gives the analytic overlap") {
    std::mt19937 rng(11);
    std::vector<int> all(128);
    std::iota(all.begin(), all.end(), 0);
    double sum = 0;
    const int n = 20000;
    for (int i = 0; i < n; ++i) {
        uint8_t x[8], y[8];
        std::shuffle(all.begin(), all.end(), rng);
        for (int e = 0; e < 8; ++e) x[e] = (uint8_t) all[e];
        std::shuffle(all.begin(), all.end(), rng);
        for (int e = 0; e < 8; ++e) y[e] = (uint8_t) all[e];
        sum += moe::overlap(x, y, 8);
    }
    CHECK(sum / n == Approx(0.5).epsilon(0.05));
}

TEST_CASE("count_runs") {
    std::vector<int> v{3, 4, 5, 9};
    CHECK(moe::count_runs(v) == 2);
    v = {7, 1, 3, 5};
    CHECK(moe::count_runs(v) == 4);
    v = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(moe::count_runs(v) == 1);
    v = {};
    CHECK(moe::count_runs(v) == 0);
}

TEST_CASE("entropy, JS divergence, gini") {
    CHECK(moe::entropy_bits({1, 1, 1, 1}) == Approx(2.0));
    CHECK(moe::entropy_bits({5, 0, 0, 0}) == Approx(0.0));
    CHECK(moe::js_divergence_bits({1, 0}, {0, 1}) == Approx(1.0));
    CHECK(moe::js_divergence_bits({3, 7}, {30, 70}) == Approx(0.0));
    CHECK(moe::gini({5, 5, 5, 5}) == Approx(0.0));
    CHECK(moe::gini({0, 0, 0, 12}) == Approx(0.75));
}

// ---------------------------------------------------------------------------------------------------- cache sim

TEST_CASE("stack distances and LRU on a hand-computed stream") {
    // objects: A=0 (size 10), B=1 (size 20), C=2 (size 5)
    const std::vector<uint64_t> size{10, 20, 5};
    const std::vector<moe::Access> s{{0, 0}, {1, 0}, {0, 0}, {2, 0}, {1, 0}, {0, 0}};
    const auto d = moe::stack_distances(s, size);
    CHECK(d[0] == moe::kInfDistance);
    CHECK(d[1] == moe::kInfDistance);
    CHECK(d[2] == 20);  // B in between
    CHECK(d[3] == moe::kInfDistance);
    CHECK(d[4] == 15);  // A, C in between (A counted once)
    CHECK(d[5] == 25);  // C, B in between
    // capacity 30: hits where d + size <= 30 -> access 2 (20+10), access 4 (15+20 = 35: miss), access 5 (25+10 = 35: miss)
    const auto h = moe::lru_hits(s, size, d, {30, 35, 1000}, 1);
    CHECK(h[0].bytes_hit[0] == 10);
    CHECK(h[1].bytes_hit[0] == 10 + 20 + 10);
    CHECK(h[2].bytes_hit[0] == 40);
    CHECK(h[0].bytes_total[0] == 10 + 20 + 10 + 5 + 20 + 10);
}

TEST_CASE("Belady beats LRU on a cyclic pattern (unit sizes)") {
    // A B C A B C with room for 2: LRU never hits; Belady keeps A and B -> 2 hits
    const std::vector<uint64_t> size{1, 1, 1};
    const std::vector<moe::Access> s{{0, 0}, {1, 0}, {2, 0}, {0, 0}, {1, 0}, {2, 0}};
    const auto d = moe::stack_distances(s, size);
    CHECK(moe::lru_hits(s, size, d, {2}, 1)[0].bytes_hit[0] == 0);
    CHECK(moe::belady_hits(s, size, 2, 1).bytes_hit[0] == 2);
    CHECK(moe::belady_hits(s, size, 3, 1).bytes_hit[0] == 3);
}

TEST_CASE("LRU hit rate never decreases with capacity; Belady >= LRU for unit sizes") {
    std::mt19937 rng(5);
    std::vector<uint64_t> size(200, 1);
    std::vector<moe::Access> s;
    for (int i = 0; i < 20000; ++i) s.push_back({(uint32_t) (rng() % 200 < 150 ? rng() % 30 : rng() % 200), 0});
    const auto d = moe::stack_distances(s, size);
    std::vector<uint64_t> caps{1, 2, 4, 8, 16, 32, 64, 128, 256};
    const auto h = moe::lru_hits(s, size, d, caps, 1);
    for (size_t i = 1; i < h.size(); ++i) CHECK(h[i].hit_rate() >= h[i - 1].hit_rate());
    for (size_t i = 0; i < caps.size(); ++i) CHECK(moe::belady_hits(s, size, caps[i], 1).hit_rate() >= h[i].hit_rate());

    // variable sizes: still monotone for LRU
    std::vector<uint64_t> vsize(200);
    for (auto & v : vsize) v = 1 + rng() % 50;
    const auto dv = moe::stack_distances(s, vsize);
    const auto hv = moe::lru_hits(s, vsize, dv, {10, 100, 1000, 10000}, 1);
    for (size_t i = 1; i < hv.size(); ++i) CHECK(hv[i].hit_rate() >= hv[i - 1].hit_rate());
}

TEST_CASE("synthetic loop routing: identical routing in every pass") {
    const moe::LoopLayout L{24, 6, 18, 3};
    // one token; every slot of a physical block picks experts {0..7}; distinct objects per (block, expert)
    std::vector<uint64_t> size(24 * 128, 1);
    std::vector<moe::Access> s;
    int looped_accesses = 0;
    for (int slot : L.exec_order()) {
        const int b = L.phys(slot);
        for (int e = 0; e < 8; ++e) {
            s.push_back({(uint32_t) (b * 128 + e), (uint8_t) (L.is_core(slot) ? 1 : 0)});
            looped_accesses += L.is_core(slot);
        }
    }
    // overlap between passes is the full 8
    const uint8_t ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(moe::overlap(ids, ids, 8) == 8);
    // cold, unlimited cache: pass 0 misses, passes 1 and 2 hit -> 2/3 of the looped expert bytes
    const auto d = moe::stack_distances(s, size);
    const auto h = moe::lru_hits(s, size, d, {1ull << 40}, 2);
    CHECK(looped_accesses == 12 * 3 * 8);
    CHECK(h[0].hit_rate(1) == Approx(2.0 / 3.0));
    CHECK(h[0].hit_rate(0) == Approx(0.0));  // prelude/coda blocks run once
}

// ---------------------------------------------------------------------------------------------------- ordering

TEST_CASE("cluster ordering recovers a hidden chain") {
    // experts are secretly arranged in a chain; consecutive chain members are selected together
    const int n = 32;
    std::mt19937_64 rng(9);
    auto hidden = moe::random_order(n, rng);
    std::vector<std::vector<uint8_t>> sel;
    for (int rep = 0; rep < 50; ++rep)
        for (int start = 0; start + 4 <= n; ++start) {
            std::vector<uint8_t> s;
            for (int j = 0; j < 4; ++j) s.push_back((uint8_t) hidden[start + j]);
            sel.push_back(s);
        }
    moe::CoMatrix c(n, std::vector<uint64_t>(n, 0));
    for (const auto & s : sel)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (i != j) ++c[s[i]][s[j]];
    const auto order = moe::cluster_order(c);
    CHECK(std::set<int>(order.begin(), order.end()).size() == (size_t) n);
    CHECK(moe::mean_runs(sel, order) == Approx(1.0));         // every selection is one contiguous run
    CHECK(moe::mean_runs(sel, moe::random_order(n, rng)) > 3.0);
    CHECK(moe::path_weight(c, order) >= moe::path_weight(c, hidden));
}

TEST_CASE("2-opt never lowers the path weight") {
    std::mt19937_64 rng(4);
    const int n = 40;
    moe::CoMatrix c(n, std::vector<uint64_t>(n, 0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) c[i][j] = c[j][i] = rng() % 100;
    auto o = moe::random_order(n, rng);
    const auto before = moe::path_weight(c, o);
    moe::two_opt(c, o);
    CHECK(moe::path_weight(c, o) >= before);
    CHECK(std::set<int>(o.begin(), o.end()).size() == (size_t) n);
}
