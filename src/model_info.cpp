#include "moe/model_info.hpp"

#include "llama.h"

#include <charconv>
#include <stdexcept>
#include <vector>

namespace moe {

std::optional<std::string> meta_str(const llama_model * model, const std::string & key) {
    std::vector<char> buf(256);
    int32_t n = llama_model_meta_val_str(model, key.c_str(), buf.data(), buf.size());
    if (n < 0) return std::nullopt;
    if ((size_t) n >= buf.size()) {
        buf.resize(n + 1);
        n = llama_model_meta_val_str(model, key.c_str(), buf.data(), buf.size());
    }
    return std::string(buf.data(), n);
}

static int meta_int(const llama_model * model, const std::string & key) {
    const auto s = meta_str(model, key);
    if (!s) throw std::runtime_error("missing GGUF key: " + key);
    int v = 0;
    const auto [p, ec] = std::from_chars(s->data(), s->data() + s->size(), v);
    if (ec != std::errc() || p != s->data() + s->size()) {
        throw std::runtime_error("GGUF key " + key + " is not an integer: '" + *s + "'");
    }
    return v;
}

ModelInfo read_model_info(const llama_model * model) {
    ModelInfo info;
    const auto arch = meta_str(model, "general.architecture");
    if (!arch) throw std::runtime_error("missing GGUF key: general.architecture");
    info.arch = *arch;

    const std::string a = info.arch + ".";
    info.n_expert      = meta_int(model, a + "expert_count");
    info.n_expert_used = meta_int(model, a + "expert_used_count");
    info.layout.n_phys     = meta_int(model, a + "block_count");
    info.layout.core_start = meta_int(model, a + "loop.start");
    info.layout.core_end   = meta_int(model, a + "loop.end");
    info.layout.n_passes   = meta_int(model, a + "loop.count");
    info.n_slots_graph = llama_model_n_layer(model);
    return info;
}

void check_model_info(const ModelInfo & info, const Expected & e) {
    std::string err;
    auto expect = [&](const char * what, auto got, auto want) {
        if (got != want) err += std::string("\n  ") + what + ": got " + std::to_string(got) + ", expected " + std::to_string(want);
    };
    if (info.arch != e.arch) err += "\n  architecture: got '" + info.arch + "', expected '" + e.arch + "'";
    expect("expert_count", info.n_expert, e.n_expert);
    expect("expert_used_count", info.n_expert_used, e.n_expert_used);
    expect("block_count", info.layout.n_phys, e.layout.n_phys);
    expect("loop.start", info.layout.core_start, e.layout.core_start);
    expect("loop.end", info.layout.core_end, e.layout.core_end);
    expect("loop.count", info.layout.n_passes, e.layout.n_passes);
    try {
        info.layout.validate();
        expect("graph slots (llama_model_n_layer)", info.n_slots_graph, info.layout.n_slots());
    } catch (const std::exception & ex) {
        err += std::string("\n  ") + ex.what();
    }
    if (!err.empty()) throw std::runtime_error("model does not match what moe-loop expects:" + err);
}

} // namespace moe
