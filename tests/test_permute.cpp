// End-to-end tests of the GGUF expert rewriter on a tiny synthetic model (QA Q5).

#include "moe/gguf_layout.hpp"
#include "moe/gguf_permute.hpp"
#include "moe/permutation.hpp"

#include "doctest/doctest.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr int kNE = 4;  // experts in the toy model

fs::path tmp(const std::string & name) {
    return fs::temp_directory_path() / ("moe-loop-test-" + std::to_string(::getpid()) + "-" + name);
}

std::vector<char> slurp(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

// Layer 0: NVFP4-like layer with .scale tensors (stored as Q8_0 / F32 here; the rewriter never decodes data).
// Layer 1: Q8_0 layer without .scale tensors, like blocks 20-23 of Inkling 2 Mini.
// attn_q_clip has a dimension equal to the expert count but is NOT an expert tensor and must stay untouched.
fs::path make_toy_gguf(const std::string & name) {
    ggml_init_params ip{64u << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(17);
    std::vector<ggml_tensor *> ts;
    auto add = [&](ggml_tensor * t, const char * n) {
        ggml_set_name(t, n);
        auto * p = static_cast<uint8_t *>(t->data);
        for (size_t i = 0; i < ggml_nbytes(t); ++i) p[i] = (uint8_t) (rng() & 0xff);
        ts.push_back(t);
    };
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 10), "token_embd.weight");
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, kNE), "blk.0.ffn_gate_inp.weight");
    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNE), "blk.0.exp_probs_b.bias");
    add(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 32, 16, kNE), "blk.0.ffn_gate_up_exps.weight");
    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNE), "blk.0.ffn_gate_up_exps.scale");
    add(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 8, kNE), "blk.0.ffn_down_exps.weight");
    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNE), "blk.0.ffn_down_exps.scale");
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kNE, kNE), "blk.0.attn_q_clip.weight");  // trap: shape [4, 4]
    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8), "blk.0.attn_norm.weight");
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, kNE), "blk.1.ffn_gate_inp.weight");
    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kNE), "blk.1.exp_probs_b.bias");
    add(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 32, 16, kNE), "blk.1.ffn_gate_up_exps.weight");
    add(ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 32, 8, kNE), "blk.1.ffn_down_exps.weight");
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 10), "output.weight");

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "toy");
    gguf_set_val_u32(g, "toy.expert_count", kNE);
    gguf_set_val_u32(g, "toy.block_count", 2);
    for (auto * t : ts) gguf_add_tensor(g, t);
    const auto path = tmp(name);
    fs::remove(path);
    REQUIRE(gguf_write_to_file(g, path.string().c_str(), false));
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

std::vector<uint8_t> tensor_bytes(const moe::GgufModel & m, const std::string & name) {
    const auto & t = m.at(name);
    std::ifstream f(m.path, std::ios::binary);
    f.seekg((std::streamoff) t.offset);
    std::vector<uint8_t> v(t.size);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize) t.size);
    return v;
}

} // namespace

TEST_CASE("classification is by name, not by shape") {
    CHECK(moe::classify_tensor("blk.3.ffn_gate_inp.weight").kind == moe::ExpertTensorKind::RouterWeight);
    CHECK(moe::classify_tensor("blk.3.exp_probs_b.bias").kind == moe::ExpertTensorKind::RouterBias);
    CHECK(moe::classify_tensor("blk.13.ffn_gate_up_exps.weight").kind == moe::ExpertTensorKind::ExpertWeight);
    CHECK(moe::classify_tensor("blk.13.ffn_gate_up_exps.weight").layer == 13);
    CHECK(moe::classify_tensor("blk.13.ffn_down_exps.scale").kind == moe::ExpertTensorKind::ExpertScale);
    CHECK(moe::classify_tensor("blk.0.attn_q_clip.weight").kind == moe::ExpertTensorKind::None);
    CHECK(moe::classify_tensor("blk.0.ffn_gate_inp_shexp.weight").kind == moe::ExpertTensorKind::None);
    CHECK(moe::classify_tensor("output.weight").kind == moe::ExpertTensorKind::None);
}

TEST_CASE("identity permutation reproduces the file byte for byte") {
    const auto in = make_toy_gguf("toy_id_in.gguf");
    const auto out = tmp("toy_id_out.gguf");
    fs::remove(out);
    const auto rep = moe::permute_gguf(in.string(), out.string(), moe::Permutation::identity(2, kNE));
    CHECK(rep.tensors_permuted == 10);  // 2 router weights, 2 biases, 4 expert weights, 2 scales
    CHECK(rep.expert_scales == 2);
    CHECK(rep.router_weights == 2);
    CHECK(slurp(in) == slurp(out));
    CHECK(moe::verify_permuted(in.string(), out.string(), moe::Permutation::identity(2, kNE)).ok);
    fs::remove(in);
    fs::remove(out);
}

TEST_CASE("a real permutation moves exactly the expert chunks") {
    const auto in = make_toy_gguf("toy_p_in.gguf");
    const auto out = tmp("toy_p_out.gguf");
    fs::remove(out);
    moe::Permutation p = moe::Permutation::identity(2, kNE);
    p.new_to_old[0] = {2, 0, 3, 1};
    p.new_to_old[1] = {3, 2, 1, 0};
    moe::permute_gguf(in.string(), out.string(), p);
    CHECK(fs::file_size(in) == fs::file_size(out));

    const auto mi = moe::open_gguf(in.string());
    const auto mo = moe::open_gguf(out.string());
    // untouched tensors, including the attn_q_clip trap
    for (const char * n : {"token_embd.weight", "blk.0.attn_q_clip.weight", "blk.0.attn_norm.weight", "output.weight"}) {
        CHECK_MESSAGE(tensor_bytes(mi, n) == tensor_bytes(mo, n), n);
    }
    // expert tensors: chunk i of out == chunk p[i] of in
    for (const char * n : {"blk.0.ffn_gate_inp.weight", "blk.0.exp_probs_b.bias", "blk.0.ffn_gate_up_exps.weight",
                           "blk.0.ffn_gate_up_exps.scale", "blk.0.ffn_down_exps.weight", "blk.0.ffn_down_exps.scale",
                           "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias", "blk.1.ffn_gate_up_exps.weight",
                           "blk.1.ffn_down_exps.weight"}) {
        const auto a = tensor_bytes(mi, n), b = tensor_bytes(mo, n);
        const int layer = moe::classify_tensor(n).layer;
        const size_t chunk = a.size() / kNE;
        bool ok = a != b;  // must actually change (all permutations here move every chunk)
        for (int i = 0; i < kNE; ++i) {
            ok &= std::memcmp(b.data() + i * chunk, a.data() + (size_t) p.new_to_old[layer][i] * chunk, chunk) == 0;
        }
        CHECK_MESSAGE(ok, n);
    }
    CHECK(moe::verify_permuted(in.string(), out.string(), p).ok);

    // verify must notice a wrong permutation
    auto wrong = p;
    std::swap(wrong.new_to_old[1][0], wrong.new_to_old[1][1]);
    const auto v = moe::verify_permuted(in.string(), out.string(), wrong);
    CHECK_FALSE(v.ok);
    CHECK(v.first_error.find("blk.1") != std::string::npos);
    fs::remove(in);
    fs::remove(out);
}

TEST_CASE("rewriter refuses unsafe situations") {
    const auto in = make_toy_gguf("toy_r_in.gguf");
    const auto out = tmp("toy_r_out.gguf");
    { std::ofstream(out) << "existing"; }
    CHECK_THROWS_WITH(moe::permute_gguf(in.string(), out.string(), moe::Permutation::identity(2, kNE)),
                      doctest::Contains("refusing to overwrite"));
    CHECK(slurp(out) == std::vector<char>{'e', 'x', 'i', 's', 't', 'i', 'n', 'g'});
    fs::remove(out);
    // permutation missing layer 1
    CHECK_THROWS_WITH(moe::permute_gguf(in.string(), out.string(), moe::Permutation::identity(1, kNE)),
                      doctest::Contains("no permutation row for layer 1"));
    CHECK_FALSE(fs::exists(out));  // nothing half-written left behind
    // wrong expert count
    CHECK_THROWS(moe::permute_gguf(in.string(), out.string(), moe::Permutation::identity(2, 8)));
    CHECK_FALSE(fs::exists(out));
    fs::remove(in);
}
