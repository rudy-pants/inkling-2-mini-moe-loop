// Rewrite a GGUF so that each layer's experts are stored in a new order, and verify such a rewrite.
//
// For layer L with permutation P (P.new_to_old[L][i] = original expert at new position i), the expert-indexed tensors
// (classified by name, see classify_tensor) are rewritten so that chunk i of the output = chunk P[i] of the input:
//   ffn_gate_inp.weight rows, exp_probs_b.bias entries, ffn_*_exps.weight slabs, ffn_*_exps.scale entries.
// Every other byte (header, metadata, other tensors, padding) is copied unchanged, so the tensor table and the file
// size stay identical and an identity permutation reproduces the input byte for byte.
#pragma once

#include "moe/permutation.hpp"

#include <cstdint>
#include <string>

namespace moe {

struct PermuteReport {
    int tensors_permuted = 0;
    int tensors_copied = 0;
    int router_weights = 0, router_biases = 0, expert_weights = 0, expert_scales = 0;
    uint64_t bytes_permuted = 0;
    uint64_t bytes_total = 0;
};

// Throws if `out` already exists, if a layer with expert tensors has no permutation row, or if any expert tensor's
// outermost dimension is not the expert count.
PermuteReport permute_gguf(const std::string & in, const std::string & out, const Permutation & perm);

struct VerifyReport {
    bool ok = false;
    std::string first_error;
    int tensors_checked = 0;
};

// Re-reads both files and checks: identical size, identical bytes outside expert tensors, and every expert chunk
// moved exactly as the permutation says.
VerifyReport verify_permuted(const std::string & in, const std::string & out, const Permutation & perm);

} // namespace moe
