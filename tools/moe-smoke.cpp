// moe-smoke: proves that this project's headers match the prebuilt libllama, and that the model is the one we expect.
//
// Loads the model, checks its metadata, prints the loop layout, then greedily generates a few tokens.
// The output should match the shipped llama-completion at --temp 0 with the same prompt (PLAN.md, QA Q1).

#include "moe/model_info.hpp"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MOE_BACKEND_DIR
#define MOE_BACKEND_DIR ""
#endif
#ifndef MOE_DEFAULT_MODEL
#define MOE_DEFAULT_MODEL ""
#endif

namespace {

struct Args {
    std::string model = MOE_DEFAULT_MODEL;
    std::string prompt = "The capital of France is";
    int n_predict = 16;
    int n_ctx = 2048;
    int ngl = 99;
    bool info_only = false;
    bool step_times = false;
    bool verbose = false;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [-m model.gguf] [-p prompt] [-n n_predict] [-c n_ctx] [-ngl n] [--info-only] [--step-times] [-v]\n"
                 "default model: %s\n",
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
        else if (s == "-p") a.prompt = next();
        else if (s == "-n") a.n_predict = std::stoi(next());
        else if (s == "-c") a.n_ctx = std::stoi(next());
        else if (s == "-ngl") a.ngl = std::stoi(next());
        else if (s == "--info-only") a.info_only = true;
        else if (s == "--step-times") a.step_times = true;
        else if (s == "-v") a.verbose = true;
        else if (s == "-h" || s == "--help") { usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown argument: " + s);
    }
    if (a.model.empty()) throw std::invalid_argument("no model given (-m)");
    return a;
}

bool g_verbose = false;
void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) throw std::runtime_error("token_to_piece failed");
    return std::string(buf, n);
}

void print_info(const moe::ModelInfo & info) {
    const auto & L = info.layout;
    std::printf("arch            %s\n", info.arch.c_str());
    std::printf("experts         %d (top-%d)\n", info.n_expert, info.n_expert_used);
    std::printf("physical blocks %d, loop core [%d, %d) x %d passes\n", L.n_phys, L.core_start, L.core_end, L.n_passes);
    std::printf("graph slots     %d (llama_model_n_layer = %d)\n", L.n_slots(), info.n_slots_graph);
    std::printf("exec order     ");
    for (int il : L.exec_order()) std::printf(" %d", il);
    std::printf("\n");
}

int run(const Args & a) {
    g_verbose = a.verbose;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    // backends (HIP, CPU variants) are plugins next to the prebuilt libllama
    if (*MOE_BACKEND_DIR) ggml_backend_load_all_from_path(MOE_BACKEND_DIR);
    else ggml_backend_load_all();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = a.ngl;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mparams);
    if (!model) throw std::runtime_error("failed to load model: " + a.model);

    const auto info = moe::read_model_info(model);
    print_info(info);
    moe::check_model_info(info);
    std::printf("metadata check  OK\n");

    if (a.info_only) {
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto prompt_len = static_cast<int32_t>(a.prompt.size());
    const int n_prompt = -llama_tokenize(vocab, a.prompt.c_str(), prompt_len, nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, a.prompt.c_str(), prompt_len, tokens.data(), n_prompt, true, true) < 0) {
        throw std::runtime_error("tokenize failed");
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = a.n_ctx;
    cparams.n_batch = std::max(n_prompt, 512);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) throw std::runtime_error("failed to create context");

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    auto set_tokens = [&](const llama_token * t, int n, llama_pos pos0) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; ++i) {
            const int idx = llama_batch_ext_add_token(batch, 0, t[i]);
            if (idx < 0) throw std::runtime_error("batch add failed: " + std::to_string(idx));
            const llama_pos pos = pos0 + i;
            llama_batch_ext_set_pos(batch, idx, &pos);
        }
        llama_batch_ext_set_output_logits(batch, n - 1, true);
    };

    std::printf("prompt tokens   %d\n---\n%s", n_prompt, a.prompt.c_str());
    std::fflush(stdout);

    set_tokens(tokens.data(), n_prompt, 0);
    llama_pos pos = 0;
    int n_gen = 0;
    // llama_process() only queues GPU work; sampling reads the logits and therefore waits for it.
    // So time is taken after each sample: step i = the wall time from sample i-1 to sample i.
    const int64_t t_start = ggml_time_us();
    std::vector<int64_t> t_sampled;
    for (int n_tokens = n_prompt; n_gen < a.n_predict;) {
        if (const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch); rc != 0) {
            throw std::runtime_error("llama_process failed: " + std::to_string(rc));
        }
        pos += n_tokens;
        llama_token next = llama_sampler_sample(smpl, ctx, -1);
        t_sampled.push_back(ggml_time_us());
        if (llama_vocab_is_eog(vocab, next)) break;
        std::printf("%s", piece(vocab, next).c_str());
        std::fflush(stdout);
        ++n_gen;
        set_tokens(&next, 1, pos);
        n_tokens = 1;
    }
    std::printf("\n---\n");
    if (!t_sampled.empty()) {
        std::printf("prefill %d tokens: %.1f ms\n", n_prompt, static_cast<double>(t_sampled[0] - t_start) / 1e3);
    }
    if (t_sampled.size() > 1) {
        const double decode_s = static_cast<double>(t_sampled.back() - t_sampled.front()) / 1e6;
        const auto n_steps = static_cast<double>(t_sampled.size() - 1);
        std::printf("generated %d tokens, decode %.1f tok/s\n", n_gen, n_steps / decode_s);
    }
    if (a.step_times && t_sampled.size() > 1) {
        std::printf("decode step ms:");
        for (size_t i = 1; i < t_sampled.size(); ++i) std::printf(" %.1f", static_cast<double>(t_sampled[i] - t_sampled[i - 1]) / 1e3);
        std::printf("\n");
    }

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
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
