// Per-layer expert permutations, as written by moe-permute.
//
// Convention: new_to_old[layer][i] = the original expert stored at position i of the permuted model.
// So a trace of the permuted model reporting expert i at layer L used the original expert new_to_old[L][i].
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe {

struct Permutation {
    int n_expert = 0;
    std::vector<std::vector<int>> new_to_old;  // [n_layer][n_expert]; empty = identity everywhere

    bool empty() const { return new_to_old.empty(); }
    int to_original(int layer, int new_id) const { return empty() ? new_id : new_to_old.at(layer).at(new_id); }
    std::vector<int> old_to_new(int layer) const;
    // Throws std::invalid_argument when a row is not a permutation of [0, n_expert).
    void validate() const;

    static Permutation identity(int n_layer, int n_expert);
};

// Text format: header line "moe-loop-permutation v1 <n_layer> <n_expert>", then one line per layer: "<layer>: id id ...".
void write_permutation(const std::string & path, const Permutation & p);
Permutation read_permutation(const std::string & path);

} // namespace moe
