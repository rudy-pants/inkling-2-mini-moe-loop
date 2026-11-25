// Expert orderings for one layer, chosen so that experts the router tends to pick together are stored next to
// each other.
//
// For a selection of k experts at storage positions P, the number of contiguous memory runs is
//     runs = k - #{ adjacent position pairs (i, i+1) that are both selected }.
// So minimizing the expected runs is the same as maximizing sum_i C[order[i]][order[i+1]], where C[a][b] counts how
// often a and b were selected together: a maximum-weight Hamiltonian path. We use greedy chains from every start
// node followed by 2-opt segment reversals.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace moe {

using CoMatrix = std::vector<std::vector<uint64_t>>;  // symmetric co-selection counts [n][n]

// Sum of C over adjacent pairs of the order (the quantity being maximized).
uint64_t path_weight(const CoMatrix & c, const std::vector<int> & order);

// Mean number of contiguous runs over the given selections when experts are stored in `order`
// (order[i] = expert at position i).
double mean_runs(const std::vector<std::vector<uint8_t>> & selections, const std::vector<int> & order);

std::vector<int> greedy_chain(const CoMatrix & c, int start);
// Improves the order in place with 2-opt reversals until no reversal helps. Returns the number of improving moves.
int two_opt(const CoMatrix & c, std::vector<int> & order);
// Best of greedy chains from every start node, each refined with 2-opt.
std::vector<int> cluster_order(const CoMatrix & c);

std::vector<int> random_order(int n, std::mt19937_64 & rng);

} // namespace moe
