// moe-dump: print a trace as CSV (one row per record) or just its header.
//
// usage: moe-dump trace.mtrace [--header] [--limit N] [--perm perm.txt]
//   --perm maps expert ids of a trace taken on a permuted model back to the original ids (see moe-permute).

#include "moe/permutation.hpp"
#include "moe/trace.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    try {
        std::string path, perm_path;
        bool header_only = false;
        long long limit = -1;
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            if (s == "--header") header_only = true;
            else if (s == "--limit" && i + 1 < argc) limit = std::stoll(argv[++i]);
            else if (s == "--perm" && i + 1 < argc) perm_path = argv[++i];
            else if (path.empty() && s[0] != '-') path = s;
            else throw std::invalid_argument("unknown argument: " + s);
        }
        if (path.empty()) throw std::invalid_argument("usage: moe-dump trace.mtrace [--header] [--limit N] [--perm perm.txt]");

        const auto t = moe::read_trace(path);
        const auto & h = t.header;
        std::printf("# version %u  n_expert %u  k %u  layout %d/[%d,%d)x%d  slots %d  weights %s  records %zu\n",
                    h.version, h.n_expert, h.k, h.layout.n_phys, h.layout.core_start, h.layout.core_end, h.layout.n_passes,
                    h.layout.n_slots(), h.has_weights() ? "yes" : "no", t.records.size());
        std::printf("# model size %llu  hash %016llx\n", (unsigned long long) h.model_size, (unsigned long long) h.model_hash);
        if (header_only) return 0;

        moe::Permutation perm;
        if (!perm_path.empty()) perm = moe::read_permutation(perm_path);

        std::printf("seq,pos,token,phase,slot,phys,pass");
        for (uint32_t e = 0; e < h.k; ++e) std::printf(",e%u", e);
        if (h.has_weights()) for (uint32_t e = 0; e < h.k; ++e) std::printf(",w%u", e);
        std::printf("\n");
        long long n = 0;
        for (const auto & r : t.records) {
            if (limit >= 0 && n++ >= limit) break;
            const int phys = h.layout.phys(r.slot);
            std::printf("%u,%u,%d,%s,%u,%d,%d", r.seq, r.pos, r.token, r.phase == moe::Phase::Decode ? "decode" : "prefill",
                        r.slot, phys, h.layout.pass(r.slot));
            for (uint32_t e = 0; e < h.k; ++e) {
                const int id = perm.empty() ? r.ids[e] : perm.to_original(phys, r.ids[e]);
                std::printf(",%d", id);
            }
            if (h.has_weights()) for (uint32_t e = 0; e < h.k; ++e) std::printf(",%.5f", r.weights[e]);
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
