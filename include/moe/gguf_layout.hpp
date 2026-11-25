// Tensor table of a GGUF file (no tensor data loaded) plus helpers to classify the MoE tensors.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace moe {

struct TensorInfo {
    std::string name;
    int type = 0;              // enum ggml_type
    std::string type_name;
    uint64_t offset = 0;       // absolute byte offset in the file
    uint64_t size = 0;         // bytes
    int n_dims = 0;
    std::array<int64_t, 4> ne{1, 1, 1, 1};
};

struct GgufModel {
    std::string path;
    uint64_t file_size = 0;
    uint64_t data_offset = 0;  // absolute offset of the tensor data section
    uint64_t alignment = 0;
    std::vector<TensorInfo> tensors;  // in file order
    std::unordered_map<std::string, size_t> index;

    const TensorInfo * find(const std::string & name) const;
    const TensorInfo & at(const std::string & name) const;  // throws when missing
    std::vector<float> read_f32(const std::string & name) const;
    std::optional<std::string> kv_str(const std::string & key) const;
    std::optional<int64_t> kv_int(const std::string & key) const;

    std::unordered_map<std::string, std::string> kv_strings;  // string-valued metadata
    std::unordered_map<std::string, int64_t> kv_ints;         // integer-valued metadata
    std::unordered_map<std::string, std::vector<int64_t>> kv_int_arrays;  // bool/integer arrays
};

GgufModel open_gguf(const std::string & path);

// How a tensor is indexed by expert. Classification is by name only: shape is never used to decide,
// because unrelated tensors (e.g. attn_q_clip [128, 24], 128 = head dim) can share the expert count.
enum class ExpertTensorKind {
    None,
    RouterWeight,   // blk.N.ffn_gate_inp.weight      [n_embd, n_expert]       expert axis = ne1 (rows)
    RouterBias,     // blk.N.exp_probs_b.bias         [n_expert]
    ExpertWeight,   // blk.N.ffn_{gate_up,gate,up,down}_exps.weight [.., .., n_expert] expert axis = ne2
    ExpertScale,    // blk.N.ffn_{...}_exps.scale     [n_expert]
};

struct ExpertTensorClass {
    ExpertTensorKind kind = ExpertTensorKind::None;
    int layer = -1;
    std::string role;  // "gate_up", "down", ... for weights/scales
};

ExpertTensorClass classify_tensor(const std::string & name);

// Bytes per expert for a tensor whose outermost dimension is the expert axis; throws if it is not.
uint64_t expert_chunk_bytes(const TensorInfo & t, int n_expert);

} // namespace moe
