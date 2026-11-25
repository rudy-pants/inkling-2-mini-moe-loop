#include "moe/layout.hpp"

#include "doctest/doctest.h"

#include <algorithm>
#include <set>

using moe::LoopLayout;

namespace {
const LoopLayout kInkling{24, 6, 18, 3};
}

TEST_CASE("slot count matches the graph") {
    CHECK(kInkling.core_len() == 12);
    CHECK(kInkling.n_slots() == 48);
    CHECK_NOTHROW(kInkling.validate());
}

TEST_CASE("slot to physical block and pass (PLAN.md table)") {
    for (int il = 0; il < 6; ++il) {
        CHECK(kInkling.phys(il) == il);
        CHECK(kInkling.pass(il) == -1);
        CHECK_FALSE(kInkling.is_core(il));
    }
    for (int il = 6; il < 18; ++il) {
        CHECK(kInkling.phys(il) == il);
        CHECK(kInkling.pass(il) == 0);
        CHECK(kInkling.is_core(il));
    }
    for (int il = 18; il < 24; ++il) {
        CHECK(kInkling.phys(il) == il);
        CHECK(kInkling.pass(il) == -1);
    }
    for (int il = 24; il < 36; ++il) {
        CHECK(kInkling.phys(il) == il - 18);
        CHECK(kInkling.pass(il) == 1);
    }
    for (int il = 36; il < 48; ++il) {
        CHECK(kInkling.phys(il) == il - 30);
        CHECK(kInkling.pass(il) == 2);
    }
    CHECK(kInkling.phys(24) == 6);
    CHECK(kInkling.phys(35) == 17);
    CHECK(kInkling.phys(36) == 6);
    CHECK(kInkling.phys(47) == 17);
}

TEST_CASE("slot() inverts (phys, pass)") {
    for (int il = 0; il < kInkling.n_slots(); ++il) {
        const int p = std::max(kInkling.pass(il), 0);
        CHECK(kInkling.slot(kInkling.phys(il), p) == il);
    }
    CHECK_THROWS_AS(kInkling.slot(6, 3), std::out_of_range);
    CHECK_THROWS_AS(kInkling.slot(24, 0), std::out_of_range);
}

TEST_CASE("execution order") {
    const auto order = kInkling.exec_order();
    REQUIRE(order.size() == 48);

    std::vector<int> expected;
    for (int i = 0; i < 18; ++i) expected.push_back(i);
    for (int i = 24; i < 48; ++i) expected.push_back(i);
    for (int i = 18; i < 24; ++i) expected.push_back(i);
    CHECK(order == expected);

    // edge cases called out in QA Q2: pass 0 hands over to pass 1, the last pass hands over to the coda
    const auto idx = kInkling.exec_index();
    CHECK(idx[24] == idx[17] + 1);
    CHECK(idx[18] == idx[47] + 1);
    CHECK(idx[0] == 0);
    CHECK(idx[23] == 47);

    // a permutation of all slots
    CHECK(std::set<int>(order.begin(), order.end()).size() == 48);
}

TEST_CASE("each physical core block appears once per pass") {
    for (int b = 6; b < 18; ++b) {
        std::vector<int> passes;
        for (int il = 0; il < kInkling.n_slots(); ++il) {
            if (kInkling.phys(il) == b) passes.push_back(kInkling.pass(il));
        }
        CHECK(passes == std::vector<int>{0, 1, 2});
    }
}

TEST_CASE("out of range and invalid layouts") {
    CHECK_THROWS_AS(kInkling.phys(-1), std::out_of_range);
    CHECK_THROWS_AS(kInkling.phys(48), std::out_of_range);
    CHECK_THROWS_AS(kInkling.pass(48), std::out_of_range);
    CHECK_THROWS_AS((LoopLayout{24, 18, 6, 3}).validate(), std::invalid_argument);
    CHECK_THROWS_AS((LoopLayout{24, 6, 30, 3}).validate(), std::invalid_argument);
    CHECK_THROWS_AS((LoopLayout{24, 6, 18, 0}).validate(), std::invalid_argument);
}

TEST_CASE("single-pass layout degenerates to a plain stack") {
    const LoopLayout plain{24, 6, 18, 1};
    CHECK(plain.n_slots() == 24);
    std::vector<int> expected(24);
    for (int i = 0; i < 24; ++i) expected[i] = i;
    CHECK(plain.exec_order() == expected);
}
