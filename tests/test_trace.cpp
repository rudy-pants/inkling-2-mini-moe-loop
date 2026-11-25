#include "moe/permutation.hpp"
#include "moe/trace.hpp"

#include "doctest/doctest.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

fs::path tmp(const std::string & name) {
    return fs::temp_directory_path() / ("moe-loop-test-" + std::to_string(::getpid()) + "-" + name);
}

moe::TraceHeader header(bool weights) {
    moe::TraceHeader h;
    h.flags = weights ? static_cast<uint32_t>(moe::TRACE_HAS_WEIGHTS) : 0u;
    h.n_expert = 128;
    h.k = 8;
    h.layout = {24, 6, 18, 3};
    h.model_size = 123456789;
    h.model_hash = 0xdeadbeefcafef00dull;
    return h;
}

std::vector<moe::TraceRecord> synthetic(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<moe::TraceRecord> v(n);
    for (int i = 0; i < n; ++i) {
        auto & r = v[i];
        r.seq = static_cast<uint32_t>(rng() % 5);
        r.pos = static_cast<uint32_t>(rng() % 10000);
        r.token = (int32_t) (rng() % 200000) - 10;
        r.slot = (uint8_t) (rng() % 48);
        r.phase = rng() % 2 ? moe::Phase::Decode : moe::Phase::Prefill;
        for (int e = 0; e < 8; ++e) {
            r.ids[e] = (uint8_t) (rng() % 128);
            r.weights[e] = (float) (rng() % 1000) / 1000.0f;
        }
    }
    return v;
}

} // namespace

TEST_CASE("trace round-trip with and without weights") {
    for (bool weights : {true, false}) {
        const auto path = tmp(weights ? "rt_w.mtrace" : "rt_nw.mtrace");
        const auto recs = synthetic(10000, 7);
        {
            moe::TraceWriter w(path.string(), header(weights));
            for (const auto & r : recs) w.write(r);
            CHECK(w.n_records() == recs.size());
        }
        const auto t = moe::read_trace(path.string());
        CHECK(t.header.k == 8);
        CHECK(t.header.n_expert == 128);
        CHECK(t.header.layout.n_slots() == 48);
        CHECK(t.header.model_hash == 0xdeadbeefcafef00dull);
        CHECK(t.header.has_weights() == weights);
        REQUIRE(t.records.size() == recs.size());
        bool same = true;
        for (size_t i = 0; i < recs.size(); ++i) {
            const auto & a = recs[i];
            const auto & b = t.records[i];
            same &= a.seq == b.seq && a.pos == b.pos && a.token == b.token && a.slot == b.slot && a.phase == b.phase;
            for (int e = 0; e < 8; ++e) {
                same &= a.ids[e] == b.ids[e];
                if (weights) same &= a.weights[e] == b.weights[e];
            }
        }
        CHECK(same);
        fs::remove(path);
    }
}

TEST_CASE("trace reader rejects bad files") {
    const auto path = tmp("bad.mtrace");
    {
        moe::TraceWriter w(path.string(), header(true));
        for (const auto & r : synthetic(10, 1)) w.write(r);
    }
    const auto good_size = fs::file_size(path);

    SUBCASE("truncated") {
        fs::resize_file(path, good_size - 3);
        CHECK_THROWS_WITH(moe::read_trace(path.string()), doctest::Contains("truncated"));
    }
    SUBCASE("wrong magic") {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(0);
        f.write("NOTATRCE", 8);
        f.close();
        CHECK_THROWS_WITH(moe::read_trace(path.string()), doctest::Contains("magic"));
    }
    SUBCASE("wrong version") {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(8);
        const uint32_t v = 99;
        f.write(reinterpret_cast<const char *>(&v), 4);
        f.close();
        CHECK_THROWS_WITH(moe::read_trace(path.string()), doctest::Contains("version"));
    }
    SUBCASE("too short") {
        fs::resize_file(path, 10);
        CHECK_THROWS(moe::read_trace(path.string()));
    }
    fs::remove(path);
}

TEST_CASE("trace writer rejects headers the format cannot hold") {
    auto h = header(false);
    h.n_expert = 300;  // ids are stored as u8
    CHECK_THROWS(moe::TraceWriter(tmp("x.mtrace").string(), h));
    h = header(false);
    h.k = 0;
    CHECK_THROWS(moe::TraceWriter(tmp("x.mtrace").string(), h));
    fs::remove(tmp("x.mtrace"));
}

TEST_CASE("permutation file round-trip and validation") {
    std::mt19937_64 rng(3);
    auto p = moe::Permutation::identity(24, 128);
    for (auto & row : p.new_to_old) std::shuffle(row.begin(), row.end(), rng);
    const auto path = tmp("perm.txt");
    moe::write_permutation(path.string(), p);
    const auto q = moe::read_permutation(path.string());
    CHECK(q.n_expert == 128);
    CHECK(q.new_to_old == p.new_to_old);
    for (int l = 0; l < 24; ++l) {
        const auto inv = q.old_to_new(l);
        for (int i = 0; i < 128; ++i) CHECK(inv[q.new_to_old[l][i]] == i);
    }
    p.new_to_old[3][5] = p.new_to_old[3][6];  // duplicate -> not a permutation
    CHECK_THROWS_AS(moe::write_permutation(path.string(), p), std::invalid_argument);
    fs::remove(path);
}
