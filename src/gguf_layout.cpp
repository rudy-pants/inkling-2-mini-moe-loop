#include "moe/gguf_layout.hpp"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <regex>
#include <stdexcept>

namespace moe {

const TensorInfo * GgufModel::find(const std::string & name) const {
    const auto it = index.find(name);
    return it == index.end() ? nullptr : &tensors[it->second];
}

const TensorInfo & GgufModel::at(const std::string & name) const {
    const auto * t = find(name);
    if (!t) throw std::runtime_error("tensor not found in " + path + ": " + name);
    return *t;
}

std::vector<float> GgufModel::read_f32(const std::string & name) const {
    const auto & t = at(name);
    if (t.type != GGML_TYPE_F32) throw std::runtime_error("tensor " + name + " is " + t.type_name + ", not f32");
    std::vector<float> v(t.size / sizeof(float));
    std::unique_ptr<FILE, int (*)(FILE *)> f(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (fseeko(f.get(), (off_t) t.offset, SEEK_SET) != 0 || std::fread(v.data(), 1, t.size, f.get()) != t.size) {
        throw std::runtime_error("read failed for tensor " + name);
    }
    return v;
}

std::optional<std::string> GgufModel::kv_str(const std::string & key) const {
    const auto it = kv_strings.find(key);
    return it == kv_strings.end() ? std::nullopt : std::optional<std::string>(it->second);
}

std::optional<int64_t> GgufModel::kv_int(const std::string & key) const {
    const auto it = kv_ints.find(key);
    return it == kv_ints.end() ? std::nullopt : std::optional<int64_t>(it->second);
}

GgufModel open_gguf(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params params{/*no_alloc =*/true, /*ctx =*/&meta};
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) throw std::runtime_error("not a readable GGUF file: " + path);

    GgufModel m;
    m.path = path;
    m.file_size = std::filesystem::file_size(path);
    m.data_offset = gguf_get_data_offset(ctx);
    m.alignment = gguf_get_alignment(ctx);

    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i) {
        const std::string key = gguf_get_key(ctx, i);
        switch (gguf_get_kv_type(ctx, i)) {
            case GGUF_TYPE_STRING: m.kv_strings[key] = gguf_get_val_str(ctx, i); break;
            case GGUF_TYPE_UINT8:  m.kv_ints[key] = gguf_get_val_u8(ctx, i); break;
            case GGUF_TYPE_INT8:   m.kv_ints[key] = gguf_get_val_i8(ctx, i); break;
            case GGUF_TYPE_UINT16: m.kv_ints[key] = gguf_get_val_u16(ctx, i); break;
            case GGUF_TYPE_INT16:  m.kv_ints[key] = gguf_get_val_i16(ctx, i); break;
            case GGUF_TYPE_UINT32: m.kv_ints[key] = gguf_get_val_u32(ctx, i); break;
            case GGUF_TYPE_INT32:  m.kv_ints[key] = gguf_get_val_i32(ctx, i); break;
            case GGUF_TYPE_UINT64: m.kv_ints[key] = (int64_t) gguf_get_val_u64(ctx, i); break;
            case GGUF_TYPE_INT64:  m.kv_ints[key] = gguf_get_val_i64(ctx, i); break;
            case GGUF_TYPE_ARRAY: {
                const auto at = gguf_get_arr_type(ctx, i);
                if (at == GGUF_TYPE_STRING || at == GGUF_TYPE_ARRAY) break;  // gguf_get_arr_data aborts on these
                const size_t n = gguf_get_arr_n(ctx, i);
                const void * d = gguf_get_arr_data(ctx, i);
                std::vector<int64_t> v;
                auto take = [&](auto * ptr) { for (size_t j = 0; j < n; ++j) v.push_back((int64_t) ptr[j]); };
                switch (at) {
                    case GGUF_TYPE_BOOL:   take(static_cast<const int8_t *>(d)); break;
                    case GGUF_TYPE_UINT8:  take(static_cast<const uint8_t *>(d)); break;
                    case GGUF_TYPE_INT8:   take(static_cast<const int8_t *>(d)); break;
                    case GGUF_TYPE_UINT16: take(static_cast<const uint16_t *>(d)); break;
                    case GGUF_TYPE_INT16:  take(static_cast<const int16_t *>(d)); break;
                    case GGUF_TYPE_UINT32: take(static_cast<const uint32_t *>(d)); break;
                    case GGUF_TYPE_INT32:  take(static_cast<const int32_t *>(d)); break;
                    case GGUF_TYPE_INT64:  take(static_cast<const int64_t *>(d)); break;
                    default: break;
                }
                if (!v.empty() || n == 0) m.kv_int_arrays[key] = std::move(v);
            } break;
            case GGUF_TYPE_BOOL: m.kv_ints[key] = gguf_get_val_bool(ctx, i); break;
            default: break;
        }
    }

    const int64_t n = gguf_get_n_tensors(ctx);
    m.tensors.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        TensorInfo t;
        t.name = gguf_get_tensor_name(ctx, i);
        t.type = gguf_get_tensor_type(ctx, i);
        t.type_name = ggml_type_name((ggml_type) t.type);
        t.offset = m.data_offset + gguf_get_tensor_offset(ctx, i);
        t.size = gguf_get_tensor_size(ctx, i);
        const ggml_tensor * gt = ggml_get_tensor(meta, t.name.c_str());
        if (!gt) throw std::runtime_error("tensor metadata missing for " + t.name);
        t.n_dims = ggml_n_dims(gt);
        for (int d = 0; d < 4; ++d) t.ne[d] = gt->ne[d];
        m.index[t.name] = m.tensors.size();
        m.tensors.push_back(std::move(t));
    }
    gguf_free(ctx);
    ggml_free(meta);
    return m;
}

std::vector<std::string> read_string_array(const std::string & path, const std::string & key) {
    gguf_init_params params{/*no_alloc =*/true, /*ctx =*/nullptr};
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) throw std::runtime_error("not a readable GGUF file: " + path);
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_STRING) {
        gguf_free(ctx);
        throw std::runtime_error(path + ": no string array " + key);
    }
    std::vector<std::string> out(gguf_get_arr_n(ctx, id));
    for (size_t i = 0; i < out.size(); ++i) out[i] = gguf_get_arr_str(ctx, id, i);
    gguf_free(ctx);
    return out;
}

ExpertTensorClass classify_tensor(const std::string & name) {
    static const std::regex re_router(R"(^blk\.(\d+)\.ffn_gate_inp\.weight$)");
    static const std::regex re_bias(R"(^blk\.(\d+)\.exp_probs_b\.bias$)");
    static const std::regex re_exps(R"(^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.(weight|scale)$)");
    std::smatch m;
    ExpertTensorClass c;
    if (std::regex_match(name, m, re_router)) {
        c.kind = ExpertTensorKind::RouterWeight;
        c.layer = std::stoi(m[1]);
    } else if (std::regex_match(name, m, re_bias)) {
        c.kind = ExpertTensorKind::RouterBias;
        c.layer = std::stoi(m[1]);
    } else if (std::regex_match(name, m, re_exps)) {
        c.kind = m[3] == "weight" ? ExpertTensorKind::ExpertWeight : ExpertTensorKind::ExpertScale;
        c.layer = std::stoi(m[1]);
        c.role = m[2];
    }
    return c;
}

uint64_t expert_chunk_bytes(const TensorInfo & t, int n_expert) {
    if (t.n_dims < 1 || t.ne[t.n_dims - 1] != n_expert) {
        throw std::runtime_error("tensor " + t.name + ": outermost dimension is " +
                                 std::to_string(t.n_dims ? t.ne[t.n_dims - 1] : 0) + ", expected the expert count " +
                                 std::to_string(n_expert));
    }
    if (t.size % (uint64_t) n_expert != 0) {
        throw std::runtime_error("tensor " + t.name + ": size " + std::to_string(t.size) + " is not divisible by " +
                                 std::to_string(n_expert));
    }
    return t.size / (uint64_t) n_expert;
}

} // namespace moe
