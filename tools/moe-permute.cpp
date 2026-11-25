// moe-permute: choose per-layer expert orderings and rewrite a GGUF with them.
//
//   moe-permute plan   -t trace.mtrace [-t more.mtrace] --method clustered|random|identity [--seed S] -o perm.txt
//   moe-permute apply  -i model.gguf -p perm.txt -o permuted.gguf
//   moe-permute verify -i model.gguf -p perm.txt -o permuted.gguf
//
// `plan` builds a co-selection matrix per physical layer from the decode records of the traces and orders the
// experts so that frequently co-selected experts are adjacent (see include/moe/ordering.hpp).
// `apply` never overwrites an existing file and writes nothing but the reordered expert tensors differently.

#include "moe/gguf_permute.hpp"
#include "moe/ordering.hpp"
#include "moe/permutation.hpp"
#include "moe/trace.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int cmd_plan(const std::vector<std::string> & traces, const std::string & method, uint64_t seed, const std::string & out) {
    if (traces.empty() || out.empty()) throw std::invalid_argument("plan needs -t trace and -o perm.txt");
    int n_phys = 0, n_expert = 0, k = 0;
    std::vector<moe::CoMatrix> co;
    std::vector<std::vector<std::vector<uint8_t>>> sel;
    for (const auto & path : traces) {
        const auto t = moe::read_trace(path);
        if (co.empty()) {
            n_phys = t.header.layout.n_phys;
            n_expert = (int) t.header.n_expert;
            k = (int) t.header.k;
            co.assign(n_phys, moe::CoMatrix(n_expert, std::vector<uint64_t>(n_expert, 0)));
            sel.assign(n_phys, {});
        } else if (t.header.layout.n_phys != n_phys || (int) t.header.n_expert != n_expert) {
            throw std::runtime_error(path + ": layout differs from the first trace");
        }
        for (const auto & r : t.records) {
            if (r.phase != moe::Phase::Decode) continue;
            const int b = t.header.layout.phys(r.slot);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < k; ++j)
                    if (i != j) ++co[b][r.ids[i]][r.ids[j]];
            sel[b].emplace_back(r.ids.begin(), r.ids.begin() + k);
        }
    }
    moe::Permutation perm = moe::Permutation::identity(n_phys, n_expert);
    std::mt19937_64 rng(seed);
    double runs_before = 0, runs_after = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < n_phys; ++b) {
        if (method == "clustered") perm.new_to_old[b] = moe::cluster_order(co[b]);
        else if (method == "random") perm.new_to_old[b] = moe::random_order(n_expert, rng);
        else if (method != "identity") throw std::invalid_argument("unknown method " + method);
        runs_before += moe::mean_runs(sel[b], moe::Permutation::identity(1, n_expert).new_to_old[0]);
        runs_after += moe::mean_runs(sel[b], perm.new_to_old[b]);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    moe::write_permutation(out, perm);
    std::printf("plan %s: %d layers x %d experts from %zu trace(s) in %.1f s\n", method.c_str(), n_phys, n_expert,
                traces.size(), secs);
    std::printf("  mean contiguous runs per top-%d selection on the training traces: original %.3f -> %.3f\n", k,
                runs_before / n_phys, runs_after / n_phys);
    std::printf("  wrote %s\n", out.c_str());
    return 0;
}

int cmd_apply(const std::string & in, const std::string & perm_path, const std::string & out) {
    if (in.empty() || perm_path.empty() || out.empty()) throw std::invalid_argument("apply needs -i, -p and -o");
    const auto perm = moe::read_permutation(perm_path);
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = moe::permute_gguf(in, out, perm);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("apply: %s -> %s in %.1f s\n", in.c_str(), out.c_str(), secs);
    std::printf("  permuted %d tensors (%d router weights, %d router biases, %d expert weights, %d expert scales), "
                "%.2f GB of %.2f GB; copied %d tensors unchanged\n",
                r.tensors_permuted, r.router_weights, r.router_biases, r.expert_weights, r.expert_scales,
                (double) r.bytes_permuted / 1e9, (double) r.bytes_total / 1e9, r.tensors_copied);
    return 0;
}

int cmd_verify(const std::string & in, const std::string & perm_path, const std::string & out) {
    if (in.empty() || perm_path.empty() || out.empty()) throw std::invalid_argument("verify needs -i, -p and -o");
    const auto perm = moe::read_permutation(perm_path);
    const auto v = moe::verify_permuted(in, out, perm);
    if (v.ok) {
        std::printf("verify: OK (%d tensors checked; all non-expert bytes identical, every expert chunk where the "
                    "permutation puts it)\n", v.tensors_checked);
        return 0;
    }
    std::printf("verify: FAILED: %s\n", v.first_error.c_str());
    return 4;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("usage: moe-permute plan|apply|verify ...");
        const std::string cmd = argv[1];
        std::vector<std::string> traces;
        std::string method = "clustered", in, perm, out;
        uint64_t seed = 1;
        for (int i = 2; i < argc; ++i) {
            const std::string s = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + s);
                return argv[++i];
            };
            if (s == "-t") traces.push_back(next());
            else if (s == "--method") method = next();
            else if (s == "--seed") seed = std::stoull(next());
            else if (s == "-i") in = next();
            else if (s == "-p") perm = next();
            else if (s == "-o") out = next();
            else throw std::invalid_argument("unknown argument: " + s);
        }
        if (cmd == "plan") return cmd_plan(traces, method, seed, out);
        if (cmd == "apply") return cmd_apply(in, perm, out);
        if (cmd == "verify") return cmd_verify(in, perm, out);
        throw std::invalid_argument("unknown command " + cmd);
    } catch (const std::invalid_argument & e) {
        std::fprintf(stderr, "error: %s\nusage:\n"
                             "  moe-permute plan   -t trace.mtrace [-t ...] --method clustered|random|identity [--seed S] -o perm.txt\n"
                             "  moe-permute apply  -i model.gguf -p perm.txt -o permuted.gguf\n"
                             "  moe-permute verify -i model.gguf -p perm.txt -o permuted.gguf\n",
                     e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
