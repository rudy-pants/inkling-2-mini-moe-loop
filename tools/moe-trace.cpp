// moe-trace: record which experts the router picks for every token at every graph slot.
//
// Hooks the `ffn_moe_topk-<il>` (and `ffn_moe_probs-<il>`) tensors through llama.cpp's cb_eval callback,
// writes a binary trace (docs/trace-format.md) and a JSON sidecar with per-sequence metadata.
//
// Prompts come from -p, --prompt-file or --prompts-dir (every *.txt, named <category>__<name>.txt).
// Prompt text is tokenized with special-token parsing, so chat-rendered prompts work as-is.

#include "moe/gguf_layout.hpp"
#include "moe/json.hpp"
#include "moe/model_info.hpp"
#include "moe/trace.hpp"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MOE_DEFAULT_MODEL
#define MOE_DEFAULT_MODEL ""
#endif
#ifndef MOE_BACKEND_DIR
#define MOE_BACKEND_DIR ""
#endif

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string model = MOE_DEFAULT_MODEL;
    std::string out;                 // trace path; sidecar is <out>.json
    std::vector<std::string> prompt_files;
    std::string prompt;
    int n_predict = 256;
    int n_ctx = 8192;
    int ngl = 99;
    float temp = 1.0f;
    uint32_t seed = 1234;
    bool no_trace = false;           // generate without the callback (QA: no-interference check)
    bool check_router = true;        // recompute top-k from probs + bias on the CPU and count mismatches
    bool verbose = false;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -o trace.mtrace (-p text | --prompt-file f.txt | --prompts-dir dir) [options]\n"
        "  -m model.gguf        model (default %s)\n"
        "  -n N                 tokens to generate per prompt (default 256; 0 = prefill only)\n"
        "  -c N                 context size (default 8192)\n"
        "  --temp T             sampling temperature (default 1.0, as start-inkling.sh; 0 = greedy)\n"
        "  --seed S             sampler seed (default 1234)\n"
        "  --no-trace           do not install the callback (generation only, for the no-interference check)\n"
        "  --no-check           skip the CPU top-k cross-check\n"
        "  -v                   llama.cpp logs\n",
        argv0, MOE_DEFAULT_MODEL);
}

Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + s);
            return argv[++i];
        };
        if (s == "-m") a.model = next();
        else if (s == "-o") a.out = next();
        else if (s == "-p") a.prompt = next();
        else if (s == "--prompt-file") a.prompt_files.push_back(next());
        else if (s == "--prompts-dir") {
            const fs::path dir = next();
            std::vector<std::string> files;
            for (const auto & e : fs::directory_iterator(dir)) {
                if (e.is_regular_file() && e.path().extension() == ".txt") files.push_back(e.path().string());
            }
            std::sort(files.begin(), files.end());
            a.prompt_files.insert(a.prompt_files.end(), files.begin(), files.end());
        }
        else if (s == "-n") a.n_predict = std::stoi(next());
        else if (s == "-c") a.n_ctx = std::stoi(next());
        else if (s == "-ngl") a.ngl = std::stoi(next());
        else if (s == "--temp") a.temp = std::stof(next());
        else if (s == "--seed") a.seed = (uint32_t) std::stoul(next());
        else if (s == "--no-trace") a.no_trace = true;
        else if (s == "--no-check") a.check_router = false;
        else if (s == "-v") a.verbose = true;
        else if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown argument: " + s);
    }
    if (a.out.empty()) throw std::invalid_argument("no output given (-o)");
    if (a.prompt.empty() && a.prompt_files.empty()) throw std::invalid_argument("no prompt given");
    return a;
}

struct Prompt {
    std::string name;
    std::string category;
    std::string text;
};

std::vector<Prompt> load_prompts(const Args & a) {
    std::vector<Prompt> out;
    if (!a.prompt.empty()) out.push_back({"cli", "cli", a.prompt});
    for (const auto & f : a.prompt_files) {
        std::ifstream in(f, std::ios::binary);
        if (!in) throw std::runtime_error("cannot read prompt file " + f);
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string stem = fs::path(f).stem().string();
        const auto sep = stem.find("__");
        out.push_back({stem, sep == std::string::npos ? "uncategorized" : stem.substr(0, sep), ss.str()});
    }
    return out;
}

bool g_verbose = false;
void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// ---------------------------------------------------------------------------------------------------------------
// Callback state

constexpr const char * kTopk = "ffn_moe_topk-";
constexpr const char * kProbs = "ffn_moe_probs-";

struct Tracer {
    moe::LoopLayout layout;
    int n_expert = 0;
    int k = 0;
    bool check = true;
    moe::TraceWriter * writer = nullptr;
    std::vector<std::vector<float>> bias;  // per physical block, from exp_probs_b

    // the batch currently being processed
    bool active = false;
    uint32_t seq = 0;
    moe::Phase phase = moe::Phase::Prefill;
    std::vector<llama_token> tokens;
    std::vector<uint32_t> positions;
    std::vector<int> consumed;                 // tokens of the current batch already seen, per slot
    std::vector<std::vector<float>> probs;     // pending probs per slot (ubatch-sized, [n_tokens][n_expert])
    std::vector<int> probs_tokens;

    // statistics
    uint64_t n_records = 0;
    uint64_t n_checked = 0;
    uint64_t n_set_mismatch = 0;   // records whose top-k set differs from the CPU recomputation
    uint64_t n_tie_mismatch = 0;   // ... of which the boundary scores tie within 1e-6
    uint64_t n_invalid = 0;        // out-of-range or duplicate ids
    uint64_t n_unexpected = 0;     // callbacks outside a batch, or more tokens than the batch holds
    std::vector<uint64_t> per_slot_records;

    void begin(uint32_t s, moe::Phase ph, const llama_token * t, int n, uint32_t pos0) {
        active = true;
        seq = s;
        phase = ph;
        tokens.assign(t, t + n);
        positions.resize(n);
        for (int i = 0; i < n; ++i) positions[i] = pos0 + (uint32_t) i;
        std::fill(consumed.begin(), consumed.end(), 0);
    }

    // Returns the number of slots that did not see every token of the batch (should be 0).
    int end() {
        active = false;
        int incomplete = 0;
        for (int c : consumed) incomplete += c != (int) tokens.size();
        return incomplete;
    }

    static int parse_il(const char * name, const char * prefix) {
        const size_t n = std::strlen(prefix);
        if (std::strncmp(name, prefix, n) != 0) return -1;
        char * end = nullptr;
        const long il = std::strtol(name + n, &end, 10);
        return (end && *end == '\0') ? (int) il : -1;
    }

    // Strided read of a 2D tensor [ne0, ne1] into a dense host buffer.
    template <typename T> static std::vector<T> read2d(ggml_tensor * t) {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<T> out((size_t) t->ne[0] * t->ne[1]);
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            for (int64_t i = 0; i < t->ne[0]; ++i) {
                std::memcpy(&out[j * t->ne[0] + i], raw.data() + i * t->nb[0] + j * t->nb[1], sizeof(T));
            }
        }
        return out;
    }

    bool on_tensor(ggml_tensor * t, bool ask) {
        const int il_topk = parse_il(t->name, kTopk);
        const int il_probs = check ? parse_il(t->name, kProbs) : -1;
        if (ask) return il_topk >= 0 || il_probs >= 0;
        if (!active) { ++n_unexpected; return true; }

        if (il_probs >= 0) {
            if (t->type != GGML_TYPE_F32 || t->ne[0] != n_expert) throw std::runtime_error("unexpected probs tensor shape");
            probs[il_probs] = read2d<float>(t);
            probs_tokens[il_probs] = (int) t->ne[1];
            return true;
        }
        if (il_topk < 0) return true;

        const int il = il_topk;
        if (!layout.valid_slot(il)) throw std::runtime_error("callback slot out of range: " + std::string(t->name));
        if (t->type != GGML_TYPE_I32 || t->ne[0] != k) {
            throw std::runtime_error(std::string("unexpected topk tensor: ") + t->name + " type " + ggml_type_name(t->type) +
                                     " ne0 " + std::to_string(t->ne[0]));
        }
        const int n_tok = (int) t->ne[1];
        const auto ids = read2d<int32_t>(t);
        const bool have_probs = check && probs_tokens[il] == n_tok;
        const int phys = layout.phys(il);

        for (int j = 0; j < n_tok; ++j) {
            const int bi = consumed[il] + j;
            if (bi >= (int) tokens.size()) { ++n_unexpected; continue; }
            moe::TraceRecord r;
            r.seq = seq;
            r.pos = positions[bi];
            r.token = tokens[bi];
            r.slot = (uint8_t) il;
            r.phase = phase;
            uint64_t seen_mask[4] = {0, 0, 0, 0};
            bool invalid = false;
            for (int e = 0; e < k; ++e) {
                const int id = ids[(size_t) j * k + e];
                if (id < 0 || id >= n_expert || (seen_mask[id >> 6] >> (id & 63) & 1)) invalid = true;
                else seen_mask[id >> 6] |= 1ull << (id & 63);
                r.ids[e] = (uint8_t) std::clamp(id, 0, n_expert - 1);
            }
            n_invalid += invalid;

            if (have_probs) {
                const float * p = &probs[il][(size_t) j * n_expert];
                // weights: the router's normalized sigmoid probabilities of the chosen experts
                float sum = 0;
                for (int e = 0; e < k; ++e) sum += p[r.ids[e]];
                for (int e = 0; e < k; ++e) r.weights[e] = sum > 0 ? p[r.ids[e]] / sum : 0.0f;
                // CPU recomputation of the selection: top-k of (probs + exp_probs_b)
                std::vector<int> order(n_expert);
                std::iota(order.begin(), order.end(), 0);
                const auto & b = bias[phys];
                auto score = [&](int e) { return p[e] + b[e]; };
                std::partial_sort(order.begin(), order.begin() + k + 1, order.end(),
                                  [&](int x, int y) { return score(x) > score(y); });
                uint64_t cpu_mask[4] = {0, 0, 0, 0};
                for (int e = 0; e < k; ++e) cpu_mask[order[e] >> 6] |= 1ull << (order[e] & 63);
                ++n_checked;
                if (std::memcmp(cpu_mask, seen_mask, sizeof(cpu_mask)) != 0) {
                    ++n_set_mismatch;
                    if (std::fabs(score(order[k - 1]) - score(order[k])) < 1e-6f) ++n_tie_mismatch;
                }
            }
            if (writer) writer->write(r);
            ++n_records;
            ++per_slot_records[il];
        }
        consumed[il] += n_tok;
        probs_tokens[il] = 0;
        return true;
    }
};

bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    return static_cast<Tracer *>(ud)->on_tensor(t, ask);
}

std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) return "";
    return std::string(buf, n);
}

struct SeqResult {
    Prompt prompt;
    int n_prompt = 0;
    std::vector<llama_token> generated;
    std::string text;
    double prefill_ms = 0;
    double decode_tps = 0;
    bool hit_eog = false;
};

int run(const Args & a) {
    g_verbose = a.verbose;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    if (*MOE_BACKEND_DIR) ggml_backend_load_all_from_path(MOE_BACKEND_DIR);
    else ggml_backend_load_all();

    const auto prompts = load_prompts(a);
    std::fprintf(stderr, "moe-trace: %zu prompt(s), %d tokens each, temp %.2f, seed %u\n", prompts.size(), a.n_predict,
                 a.temp, a.seed);

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = a.ngl;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mparams);
    if (!model) throw std::runtime_error("failed to load model: " + a.model);
    const auto info = moe::read_model_info(model);
    moe::check_model_info(info);

    Tracer tr;
    tr.layout = info.layout;
    tr.n_expert = info.n_expert;
    tr.k = info.n_expert_used;
    tr.check = a.check_router;
    tr.consumed.assign(info.layout.n_slots(), 0);
    tr.probs.assign(info.layout.n_slots(), {});
    tr.probs_tokens.assign(info.layout.n_slots(), 0);
    tr.per_slot_records.assign(info.layout.n_slots(), 0);
    if (tr.check) {
        const auto g = moe::open_gguf(a.model);
        for (int b = 0; b < info.layout.n_phys; ++b) {
            tr.bias.push_back(g.read_f32("blk." + std::to_string(b) + ".exp_probs_b.bias"));
        }
    }

    std::unique_ptr<moe::TraceWriter> writer;
    if (!a.no_trace) {
        const auto fp = moe::model_fingerprint(a.model);
        moe::TraceHeader h;
        h.flags = a.check_router ? static_cast<uint32_t>(moe::TRACE_HAS_WEIGHTS) : 0u;
        h.n_expert = (uint32_t) info.n_expert;
        h.k = (uint32_t) info.n_expert_used;
        h.layout = info.layout;
        h.model_size = fp.size;
        h.model_hash = fp.hash;
        writer = std::make_unique<moe::TraceWriter>(a.out, h);
        tr.writer = writer.get();
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = a.n_ctx;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;
    if (!a.no_trace) {
        cparams.cb_eval = eval_cb;
        cparams.cb_eval_user_data = &tr;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) throw std::runtime_error("failed to create context");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (a.temp <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(a.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(a.seed));
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    auto run_batch = [&](uint32_t seq, moe::Phase ph, const llama_token * t, int n, uint32_t pos0) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; ++i) {
            const int idx = llama_batch_ext_add_token(batch, 0, t[i]);
            if (idx < 0) throw std::runtime_error("batch add failed: " + std::to_string(idx));
            const llama_pos pos = (llama_pos) (pos0 + i);
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n - 1, true);
        tr.begin(seq, ph, t, n, pos0);
        if (const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch); rc != 0) {
            throw std::runtime_error("llama_process failed: " + std::to_string(rc));
        }
    };

    std::vector<SeqResult> results;
    uint64_t incomplete_batches = 0;
    for (size_t si = 0; si < prompts.size(); ++si) {
        SeqResult res;
        res.prompt = prompts[si];
        const auto & text = prompts[si].text;
        const int n_prompt = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, true);
        std::vector<llama_token> toks(n_prompt);
        if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n_prompt, true, true) < 0) {
            throw std::runtime_error("tokenize failed for " + prompts[si].name);
        }
        if (n_prompt + a.n_predict > a.n_ctx) throw std::runtime_error(prompts[si].name + ": prompt + generation exceeds -c");
        res.n_prompt = n_prompt;

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);

        const int64_t t0 = ggml_time_us();
        // prefill in chunks of n_batch
        for (int off = 0; off < n_prompt; off += (int) cparams.n_batch) {
            const int n = std::min<int>((int) cparams.n_batch, n_prompt - off);
            run_batch((uint32_t) si, moe::Phase::Prefill, toks.data() + off, n, (uint32_t) off);
            if (off + n < n_prompt) {
                llama_synchronize(ctx);
                incomplete_batches += tr.end();
            }
        }
        uint32_t pos = (uint32_t) n_prompt;
        std::vector<int64_t> t_sampled;
        for (int g = 0; g < a.n_predict; ++g) {
            const llama_token next = llama_sampler_sample(smpl, ctx, -1);  // waits for the GPU
            incomplete_batches += tr.end();
            t_sampled.push_back(ggml_time_us());
            if (llama_vocab_is_eog(vocab, next)) { res.hit_eog = true; break; }
            res.generated.push_back(next);
            res.text += piece(vocab, next);
            run_batch((uint32_t) si, moe::Phase::Decode, &next, 1, pos);
            ++pos;
        }
        if (a.n_predict == 0 || res.hit_eog || (int) res.generated.size() == a.n_predict) {
            llama_synchronize(ctx);
            if (tr.active) incomplete_batches += tr.end();
        }
        res.prefill_ms = t_sampled.empty() ? 0 : (double) (t_sampled.front() - t0) / 1e3;
        if (t_sampled.size() > 3) {
            // skip the first two decode steps (graph rebuild + HIP graph capture, see PLAN.md)
            res.decode_tps = (double) (t_sampled.size() - 3) / ((double) (t_sampled.back() - t_sampled[2]) / 1e6);
        }
        std::fprintf(stderr, "[%2zu/%zu] %-34s prompt %5d  gen %4zu%s  prefill %7.1f ms  decode %5.1f tok/s\n", si + 1,
                     prompts.size(), res.prompt.name.c_str(), n_prompt, res.generated.size(), res.hit_eog ? " (eog)" : "",
                     res.prefill_ms, res.decode_tps);
        results.push_back(std::move(res));
    }

    if (writer) writer->close();

    // ---- sidecar JSON
    const std::string side = a.out + ".json";
    std::FILE * jf = std::fopen(side.c_str(), "w");
    if (!jf) throw std::runtime_error("cannot write " + side);
    using moe::json::escape;
    using moe::json::num;
    std::fprintf(jf, "{\n  \"trace\": %s,\n  \"traced\": %s,\n  \"model\": %s,\n", escape(a.out).c_str(),
                 a.no_trace ? "false" : "true", escape(a.model).c_str());
    std::fprintf(jf, "  \"sampling\": {\"temp\": %s, \"seed\": %u, \"n_predict\": %d},\n", num(a.temp).c_str(), a.seed,
                 a.n_predict);
    std::fprintf(jf, "  \"stats\": {\"records\": %llu, \"checked\": %llu, \"set_mismatch\": %llu, \"tie_mismatch\": %llu, "
                     "\"invalid\": %llu, \"unexpected\": %llu, \"incomplete_batches\": %llu},\n",
                 (unsigned long long) tr.n_records, (unsigned long long) tr.n_checked,
                 (unsigned long long) tr.n_set_mismatch, (unsigned long long) tr.n_tie_mismatch,
                 (unsigned long long) tr.n_invalid, (unsigned long long) tr.n_unexpected,
                 (unsigned long long) incomplete_batches);
    std::fprintf(jf, "  \"seqs\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & r = results[i];
        std::fprintf(jf, "    {\"seq\": %zu, \"name\": %s, \"category\": %s, \"n_prompt\": %d, \"n_gen\": %zu, \"eog\": %s, "
                         "\"prefill_ms\": %s, \"decode_tps\": %s,\n     \"tokens\": [",
                     i, escape(r.prompt.name).c_str(), escape(r.prompt.category).c_str(), r.n_prompt, r.generated.size(),
                     r.hit_eog ? "true" : "false", num(r.prefill_ms).c_str(), num(r.decode_tps).c_str());
        for (size_t j = 0; j < r.generated.size(); ++j) std::fprintf(jf, "%s%d", j ? "," : "", r.generated[j]);
        std::fprintf(jf, "],\n     \"text\": %s}%s\n", escape(r.text).c_str(), i + 1 < results.size() ? "," : "");
    }
    std::fprintf(jf, "  ]\n}\n");
    std::fclose(jf);

    // ---- summary
    uint64_t total_tokens = 0;
    for (const auto & r : results) total_tokens += (uint64_t) r.n_prompt + r.generated.size();
    std::fprintf(stderr, "moe-trace: %llu records for %llu tokens (expect %d per token)\n",
                 (unsigned long long) tr.n_records, (unsigned long long) total_tokens, info.layout.n_slots());
    if (!a.no_trace) {
        std::fprintf(stderr, "  CPU top-k check: %llu checked, %llu set mismatches (%llu at ties), %llu invalid, "
                             "%llu unexpected, %llu incomplete batches\n",
                     (unsigned long long) tr.n_checked, (unsigned long long) tr.n_set_mismatch,
                     (unsigned long long) tr.n_tie_mismatch, (unsigned long long) tr.n_invalid,
                     (unsigned long long) tr.n_unexpected, (unsigned long long) incomplete_batches);
    }
    std::fprintf(stderr, "  wrote %s%s\n", a.no_trace ? "" : (a.out + " and ").c_str(), side.c_str());

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    const bool ok = a.no_trace || (tr.n_records == total_tokens * (uint64_t) info.layout.n_slots() && tr.n_invalid == 0 &&
                                   tr.n_unexpected == 0 && incomplete_batches == 0);
    return ok ? 0 : 3;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::invalid_argument & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
