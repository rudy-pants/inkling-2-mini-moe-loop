// moe-analyze: statistics and cache simulation over a moe-trace trace.
//
// usage: moe-analyze trace.mtrace [-m model.gguf] [-o results/dir] [--perm perm.txt] [--no-kv]
//
// Writes CSV/JSON files into the output directory and prints a summary. See PLAN.md, Phase 2.

#include "moe/cachesim.hpp"
#include "moe/gguf_layout.hpp"
#include "moe/json.hpp"
#include "moe/ordering.hpp"
#include "moe/permutation.hpp"
#include "moe/stats.hpp"
#include "moe/trace.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef MOE_DEFAULT_MODEL
#define MOE_DEFAULT_MODEL ""
#endif

namespace fs = std::filesystem;
using moe::json::escape;
using moe::json::num;

namespace {

struct Args {
    std::string trace;
    std::string model = MOE_DEFAULT_MODEL;
    std::string out;
    std::string perm;
    bool kv = true;
};

struct SeqInfo {
    std::string name = "?";
    std::string category = "?";
};

// Reads seq -> name/category from the sidecar JSON written by moe-trace (fixed layout, one seq object per line).
std::map<uint32_t, SeqInfo> read_sidecar(const std::string & path) {
    std::map<uint32_t, SeqInfo> out;
    std::ifstream in(path);
    if (!in) return out;
    static const std::regex re(R"re(^\s*\{"seq": (\d+), "name": "([^"]*)", "category": "([^"]*)")re");
    std::string line;
    std::smatch m;
    while (std::getline(in, line)) {
        if (std::regex_search(line, m, re)) out[(uint32_t) std::stoul(m[1])] = {m[2], m[3]};
    }
    return out;
}

// All records of one token, indexed by slot.
struct TokenView {
    uint32_t seq = 0, pos = 0;
    moe::Phase phase = moe::Phase::Prefill;
    std::vector<const moe::TraceRecord *> slot;
};

struct Csv {
    std::ofstream f;
    explicit Csv(const fs::path & p) : f(p) {
        if (!f) throw std::runtime_error("cannot write " + p.string());
    }
    template <typename... T> void row(const T &... v) {
        bool first = true;
        ((f << (first ? "" : ",") << v, first = false), ...);
        f << "\n";
    }
};

std::string mb(uint64_t b) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f MB", (double) b / (1 << 20));
    return buf;
}

int run(const Args & a) {
    const auto trace = moe::read_trace(a.trace);
    const auto & H = trace.header;
    const auto & L = H.layout;
    const int K = (int) H.k, NE = (int) H.n_expert, NS = L.n_slots(), NP = L.n_phys;
    const fs::path out = a.out.empty() ? fs::path("results") / fs::path(a.trace).stem() : fs::path(a.out);
    fs::create_directories(out);
    const auto seqinfo = read_sidecar(a.trace + ".json");

    std::printf("trace %s: %zu records, %d slots, %d experts, top-%d, weights %s\n", a.trace.c_str(), trace.records.size(),
                NS, NE, K, H.has_weights() ? "yes" : "no");

    // ------------------------------------------------------------------ index tokens
    std::vector<TokenView> toks;
    {
        std::unordered_map<uint64_t, size_t> idx;
        for (const auto & r : trace.records) {
            const uint64_t key = (uint64_t) r.seq << 32 | r.pos;
            auto [it, fresh] = idx.try_emplace(key, toks.size());
            if (fresh) {
                TokenView t;
                t.seq = r.seq;
                t.pos = r.pos;
                t.phase = r.phase;
                t.slot.assign(NS, nullptr);
                toks.push_back(std::move(t));
            }
            auto & t = toks[it->second];
            if (t.slot[r.slot]) throw std::runtime_error("duplicate record for one token and slot");
            t.slot[r.slot] = &r;
        }
        std::sort(toks.begin(), toks.end(), [](const TokenView & x, const TokenView & y) {
            return x.seq != y.seq ? x.seq < y.seq : x.pos < y.pos;
        });
    }
    size_t n_dec = 0, n_incomplete = 0;
    for (const auto & t : toks) {
        n_dec += t.phase == moe::Phase::Decode;
        for (auto * p : t.slot) n_incomplete += p == nullptr;
    }
    if (n_incomplete) throw std::runtime_error(std::to_string(n_incomplete) + " missing (token, slot) records");
    std::printf("tokens: %zu (%zu prefill, %zu decode) in %zu sequences\n", toks.size(), toks.size() - n_dec, n_dec,
                seqinfo.size());

    std::ostringstream summary;  // JSON body
    summary << "{\n  \"trace\": " << escape(a.trace) << ",\n  \"tokens\": " << toks.size() << ",\n  \"decode_tokens\": " << n_dec
            << ",\n  \"n_expert\": " << NE << ",\n  \"k\": " << K << ",\n";

    auto phase_name = [](int ph) { return ph == 0 ? "prefill" : ph == 1 ? "decode" : "all"; };
    auto phase_ok = [](const TokenView & t, int ph) { return ph == 2 || (int) t.phase == ph; };

    // ------------------------------------------------------------------ A. popularity
    // counts[phase][phys][pass+1][expert]; pass index 0 = prelude/coda (-1), 1..P = pass 0..P-1
    const int NPASS = L.n_passes + 1;
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> counts(
        2, std::vector<std::vector<std::vector<uint64_t>>>(NP, std::vector<std::vector<uint64_t>>(NPASS, std::vector<uint64_t>(NE, 0))));
    for (const auto & t : toks) {
        for (int s = 0; s < NS; ++s) {
            const auto * r = t.slot[s];
            auto & c = counts[(int) t.phase][L.phys(s)][L.pass(s) + 1];
            for (int e = 0; e < K; ++e) ++c[r->ids[e]];
        }
    }
    {
        Csv pop(out / "popularity.csv");
        pop.row("phase", "phys", "pass", "expert", "count");
        Csv ls(out / "layer_stats.csv");
        ls.row("phase", "phys", "experts_used", "entropy_bits", "effective_experts", "top8_share", "max_share", "gini");
        std::printf("\n[A] expert popularity per physical layer (decode, all passes pooled)\n");
        std::printf("    phys  used  eff.experts  top8-share  max-share  gini\n");
        double sum_eff = 0, min_eff = 1e9, max_top8 = 0;
        for (int ph = 0; ph < 3; ++ph) {
            for (int b = 0; b < NP; ++b) {
                std::vector<uint64_t> tot(NE, 0);
                for (int p = 0; p < NPASS; ++p) {
                    for (int e = 0; e < NE; ++e) {
                        const uint64_t v = ph == 2 ? counts[0][b][p][e] + counts[1][b][p][e] : counts[ph][b][p][e];
                        tot[e] += v;
                        if (ph < 2 && v) pop.row(phase_name(ph), b, p - 1, e, v);
                    }
                }
                const uint64_t total = std::accumulate(tot.begin(), tot.end(), (uint64_t) 0);
                if (!total) continue;
                auto sorted = tot;
                std::sort(sorted.rbegin(), sorted.rend());
                const uint64_t top8 = std::accumulate(sorted.begin(), sorted.begin() + std::min(8, NE), (uint64_t) 0);
                const int used = (int) std::count_if(tot.begin(), tot.end(), [](uint64_t v) { return v > 0; });
                const double h = moe::entropy_bits(tot), eff = std::exp2(h);
                const double top8s = (double) top8 / (double) total, maxs = (double) sorted[0] / (double) total;
                ls.row(phase_name(ph), b, used, num(h), num(eff), num(top8s), num(maxs), num(moe::gini(tot)));
                if (ph == 1) {
                    std::printf("    %4d  %4d  %11.1f  %9.1f%%  %8.1f%%  %.3f\n", b, used, eff, 100 * top8s, 100 * maxs,
                                moe::gini(tot));
                    sum_eff += eff;
                    min_eff = std::min(min_eff, eff);
                    max_top8 = std::max(max_top8, top8s);
                }
            }
        }
        std::printf("    (uniform routing would give eff.experts %d and top8-share %.1f%%; selection share of a\n"
                    "     single expert is at most 1/%d of selections = %.1f%% when it is always picked)\n",
                    NE, 100.0 * 8 * K / (NE * K), K, 100.0 / K);
        summary << "  \"popularity_decode\": {\"mean_effective_experts\": " << num(sum_eff / NP)
                << ", \"min_effective_experts\": " << num(min_eff) << ", \"max_top8_share\": " << num(max_top8) << "},\n";
    }

    // ------------------------------------------------------------------ B. loop overlap
    {
        Csv lo(out / "loop_overlap.csv");
        lo.row("phase", "phys", "pass_a", "pass_b", "tokens", "mean_overlap", "mean_weighted_overlap", "frac_identical_set",
               "random_baseline");
        Csv hist(out / "loop_overlap_hist.csv");
        hist.row("phase", "pass_a", "pass_b", "overlap", "count");
        const double base = moe::random_overlap(K, NE);
        std::vector<std::pair<int, int>> pairs;
        for (int x = 0; x < L.n_passes; ++x)
            for (int y = x + 1; y < L.n_passes; ++y) pairs.emplace_back(x, y);
        std::printf("\n[B] loop overlap: experts shared by the same token at the same physical block in two passes\n");
        std::printf("    (random routing baseline: %.2f of %d)\n", base, K);
        summary << "  \"loop_overlap\": {\"random_baseline\": " << num(base);
        for (int ph = 0; ph < 3; ++ph) {
            std::vector<double> pair_sum(pairs.size(), 0), pair_wsum(pairs.size(), 0);
            std::vector<uint64_t> pair_n(pairs.size(), 0);
            std::vector<std::vector<uint64_t>> h(pairs.size(), std::vector<uint64_t>(K + 1, 0));
            if (ph < 2) std::printf("    %s:\n      phys", phase_name(ph));
            if (ph < 2) for (auto [x, y] : pairs) std::printf("   p%d-p%d", x, y);
            if (ph < 2) std::printf("   (mean shared experts; weighted overlap in brackets)\n");
            for (int b = L.core_start; b < L.core_end; ++b) {
                std::vector<double> s(pairs.size(), 0), ws(pairs.size(), 0), ident(pairs.size(), 0);
                uint64_t n = 0;
                for (const auto & t : toks) {
                    if (!phase_ok(t, ph)) continue;
                    ++n;
                    for (size_t pi = 0; pi < pairs.size(); ++pi) {
                        const auto * ra = t.slot[L.slot(b, pairs[pi].first)];
                        const auto * rb = t.slot[L.slot(b, pairs[pi].second)];
                        const int o = moe::overlap(ra->ids.data(), rb->ids.data(), K);
                        s[pi] += o;
                        ident[pi] += o == K ? 1.0 : 0.0;
                        ++h[pi][o];
                        if (H.has_weights()) {
                            ws[pi] += moe::weighted_overlap(ra->ids.data(), ra->weights.data(), rb->ids.data(),
                                                            rb->weights.data(), K);
                        }
                    }
                }
                if (!n) continue;
                if (ph < 2) std::printf("      %4d", b);
                for (size_t pi = 0; pi < pairs.size(); ++pi) {
                    const double dn = (double) n;
                    lo.row(phase_name(ph), b, pairs[pi].first, pairs[pi].second, n, num(s[pi] / dn), num(ws[pi] / dn),
                           num(ident[pi] / dn), num(base));
                    if (ph < 2) std::printf("  %4.2f [%.2f]", s[pi] / dn, ws[pi] / dn);
                    pair_sum[pi] += s[pi];
                    pair_wsum[pi] += ws[pi];
                    pair_n[pi] += n;
                }
                if (ph < 2) std::printf("\n");
            }
            for (size_t pi = 0; pi < pairs.size(); ++pi) {
                for (int o = 0; o <= K; ++o) hist.row(phase_name(ph), pairs[pi].first, pairs[pi].second, o, h[pi][o]);
                if (pair_n[pi]) {
                    summary << ", \"" << phase_name(ph) << "_p" << pairs[pi].first << "_p" << pairs[pi].second
                            << "\": " << num(pair_sum[pi] / (double) pair_n[pi]);
                }
            }
            if (ph < 2) {
                std::printf("      all ");
                for (size_t pi = 0; pi < pairs.size(); ++pi) {
                    if (pair_n[pi]) std::printf("  %4.2f [%.2f]", pair_sum[pi] / (double) pair_n[pi], pair_wsum[pi] / (double) pair_n[pi]);
                }
                std::printf("\n");
            }
        }
        summary << "},\n";
    }

    // ------------------------------------------------------------------ C. token-to-token reuse (decode)
    {
        Csv tl(out / "token_lag.csv");
        tl.row("slot", "phys", "pass", "lag", "pairs", "mean_overlap");
        Csv un(out / "window_union.csv");
        un.row("slot", "phys", "pass", "window", "mean_distinct_experts");
        const int max_lag = 16;
        const std::vector<int> windows{1, 2, 3, 4, 8, 16, 32};
        std::vector<double> lag_sum(max_lag + 1, 0);
        std::vector<uint64_t> lag_n(max_lag + 1, 0);
        std::vector<double> win_sum(windows.size(), 0);
        std::vector<uint64_t> win_n(windows.size(), 0);
        for (int s = 0; s < NS; ++s) {
            for (int lag = 1; lag <= max_lag; ++lag) {
                double sum = 0;
                uint64_t n = 0;
                for (size_t i = lag; i < toks.size(); ++i) {
                    const auto & t = toks[i], & u = toks[i - lag];
                    if (t.phase != moe::Phase::Decode || u.phase != moe::Phase::Decode || t.seq != u.seq ||
                        t.pos != u.pos + (uint32_t) lag) continue;
                    sum += moe::overlap(t.slot[s]->ids.data(), u.slot[s]->ids.data(), K);
                    ++n;
                }
                if (n) {
                    tl.row(s, L.phys(s), L.pass(s), lag, n, num(sum / (double) n));
                    lag_sum[lag] += sum;
                    lag_n[lag] += n;
                }
            }
            for (size_t wi = 0; wi < windows.size(); ++wi) {
                const int w = windows[wi];
                double sum = 0;
                uint64_t n = 0;
                for (size_t i = 0; i + w <= toks.size(); i += w) {
                    if (toks[i].phase != moe::Phase::Decode || toks[i + w - 1].phase != moe::Phase::Decode ||
                        toks[i].seq != toks[i + w - 1].seq) continue;
                    std::array<char, 256> seen{};
                    int distinct = 0;
                    for (int j = 0; j < w; ++j)
                        for (int e = 0; e < K; ++e) distinct += seen[toks[i + j].slot[s]->ids[e]]++ ? 0 : 1;
                    sum += distinct;
                    ++n;
                }
                if (n) {
                    un.row(s, L.phys(s), L.pass(s), w, num(sum / (double) n));
                    win_sum[wi] += sum;
                    win_n[wi] += n;
                }
            }
        }
        std::printf("\n[C] decode: experts shared with the token `lag` steps earlier, same slot (random: %.2f)\n    lag ",
                    moe::random_overlap(K, NE));
        for (int lag : {1, 2, 3, 4, 8, 16}) std::printf("  %5d", lag);
        std::printf("\n        ");
        summary << "  \"token_lag_overlap\": {";
        for (int lag : {1, 2, 3, 4, 8, 16}) {
            const double v = lag_n[lag] ? lag_sum[lag] / (double) lag_n[lag] : 0;
            std::printf("  %5.2f", v);
            summary << (lag == 1 ? "" : ", ") << "\"" << lag << "\": " << num(v);
        }
        summary << "},\n  \"window_distinct_experts\": {";
        std::printf("\n    distinct experts needed per slot for a window of W consecutive decode tokens\n    W   ");
        for (int w : windows) std::printf("  %5d", w);
        std::printf("\n        ");
        for (size_t wi = 0; wi < windows.size(); ++wi) {
            const double v = win_n[wi] ? win_sum[wi] / (double) win_n[wi] : 0;
            std::printf("  %5.1f", v);
            summary << (wi ? ", " : "") << "\"" << windows[wi] << "\": " << num(v);
        }
        std::printf("   (max %d = W*k)\n", K);
        summary << "},\n";
    }

    // ------------------------------------------------------------------ D. specialization by prompt category
    {
        std::map<std::string, std::vector<std::vector<uint64_t>>> cat;  // category -> [phys][expert]
        std::vector<std::vector<uint64_t>> all(NP, std::vector<uint64_t>(NE, 0));
        for (const auto & t : toks) {
            const auto it = seqinfo.find(t.seq);
            const std::string c = it == seqinfo.end() ? "?" : it->second.category;
            auto & m = cat.try_emplace(c, NP, std::vector<uint64_t>(NE, 0)).first->second;
            for (int s = 0; s < NS; ++s)
                for (int e = 0; e < K; ++e) {
                    ++m[L.phys(s)][t.slot[s]->ids[e]];
                    ++all[L.phys(s)][t.slot[s]->ids[e]];
                }
        }
        Csv js(out / "category_js.csv");
        js.row("category", "phys", "js_divergence_bits", "experts_used", "top_experts");
        std::printf("\n[D] specialization: Jensen-Shannon divergence (bits) of each category's expert use vs. all prompts\n");
        std::printf("    category        mean JS   max JS (layer)   experts used per layer (mean)\n");
        summary << "  \"category_js_mean\": {";
        bool first = true;
        for (const auto & [c, m] : cat) {
            double sum = 0, mx = 0, used_sum = 0;
            int mx_l = 0;
            for (int b = 0; b < NP; ++b) {
                const double d = moe::js_divergence_bits(m[b], all[b]);
                const int used = (int) std::count_if(m[b].begin(), m[b].end(), [](uint64_t v) { return v > 0; });
                std::vector<int> order(NE);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int x, int y) { return m[b][x] > m[b][y]; });
                std::string top;
                for (int i = 0; i < 5; ++i) top += (i ? " " : "") + std::to_string(order[i]);
                js.row(c, b, num(d), used, top);
                sum += d;
                used_sum += used;
                if (d > mx) { mx = d; mx_l = b; }
            }
            std::printf("    %-14s  %7.3f   %6.3f (%2d)     %6.1f\n", c.c_str(), sum / NP, mx, mx_l, used_sum / NP);
            summary << (first ? "" : ", ") << escape(c) << ": " << num(sum / NP);
            first = false;
        }
        summary << "},\n";
    }

    // ------------------------------------------------------------------ E. cache simulation (decode tokens)
    {
        const auto model = moe::open_gguf(a.model);
        const auto fp = moe::model_fingerprint(a.model);
        if (fp.size != H.model_size || fp.hash != H.model_hash) {
            std::printf("\nWARNING: %s does not match the model the trace was recorded on (fingerprint differs)\n",
                        a.model.c_str());
        }
        const std::string arch = model.kv_str("general.architecture").value_or("");
        std::vector<uint64_t> dense(NP, 0), slab_gu(NP, 0), slab_dn(NP, 0);
        uint64_t head = 0, embd_row = 0;
        for (const auto & t : model.tensors) {
            const auto cls = moe::classify_tensor(t.name);
            if (t.name.rfind("blk.", 0) == 0) {
                const int b = std::stoi(t.name.substr(4));
                if (b >= NP) continue;
                if (cls.kind == moe::ExpertTensorKind::ExpertWeight) {
                    (cls.role == "down" ? slab_dn[b] : slab_gu[b]) += moe::expert_chunk_bytes(t, NE);
                } else {
                    dense[b] += t.size;
                }
            } else if (t.name == "token_embd.weight") {
                embd_row = t.size / (uint64_t) t.ne[1];
            } else {
                head += t.size;  // output.weight, output_norm, ... read once per token
            }
        }
        // KV model
        const int64_t kl = model.kv_int(arch + ".attention.key_length").value_or(0);
        const int64_t vl = model.kv_int(arch + ".attention.value_length").value_or(0);
        const int64_t window = model.kv_int(arch + ".attention.sliding_window").value_or(0);
        // head_count_kv is either a scalar or a per-layer array; the cache is F16
        std::vector<uint64_t> kv_per_pos_l(NP, 0);
        {
            const std::string key = arch + ".attention.head_count_kv";
            const auto it = model.kv_int_arrays.find(key);
            const int64_t scalar = model.kv_int(key).value_or(0);
            for (int b = 0; b < NP; ++b) {
                const int64_t h = it != model.kv_int_arrays.end() && b < (int) it->second.size() ? it->second[b] : scalar;
                kv_per_pos_l[b] = (uint64_t) (h * (kl + vl) * 2);
            }
        }
        const uint64_t kv_per_pos = *std::max_element(kv_per_pos_l.begin(), kv_per_pos_l.end());
        std::vector<char> is_swa(NP, 0);
        if (const auto it = model.kv_int_arrays.find(arch + ".attention.sliding_window_pattern"); it != model.kv_int_arrays.end()) {
            for (int b = 0; b < NP && b < (int) it->second.size(); ++b) is_swa[b] = it->second[b] != 0;
        }
        const uint32_t kv_chunk_pos = 256;

        enum Kind : uint8_t { DENSE = 0, EXPERT = 1, KV = 2, HEAD = 3, NKIND = 4 };
        const char * kind_name[NKIND] = {"dense layer weights", "expert slabs", "KV cache", "output head + embd"};
        std::vector<uint64_t> osize;
        // object ids: [0, NP) dense; then experts (b, e, half); then head; then KV chunks on demand
        for (int b = 0; b < NP; ++b) osize.push_back(dense[b]);
        const uint32_t exp_base = (uint32_t) osize.size();
        for (int b = 0; b < NP; ++b)
            for (int e = 0; e < NE; ++e) {
                osize.push_back(slab_gu[b]);
                osize.push_back(slab_dn[b]);
            }
        const uint32_t head_id = (uint32_t) osize.size();
        osize.push_back(head + embd_row);
        std::unordered_map<uint64_t, uint32_t> kv_ids;

        const auto order = L.exec_order();
        std::vector<moe::Access> stream;
        std::vector<int> stream_slot;  // slot of each access, -1 for head
        uint64_t n_tok = 0;
        for (const auto & t : toks) {
            if (t.phase != moe::Phase::Decode) continue;
            ++n_tok;
            for (int s : order) {
                const int b = L.phys(s);
                stream.push_back({(uint32_t) b, DENSE});
                stream_slot.push_back(s);
                if (a.kv && kv_per_pos) {
                    const uint32_t lo = is_swa[b] && window > 0 && t.pos + 1 > (uint32_t) window ? t.pos + 1 - (uint32_t) window : 0;
                    for (uint32_t c = lo / kv_chunk_pos; c <= t.pos / kv_chunk_pos; ++c) {
                        const uint64_t key = (uint64_t) t.seq << 40 | (uint64_t) s << 32 | c;
                        auto [it, fresh] = kv_ids.try_emplace(key, (uint32_t) osize.size());
                        if (fresh) osize.push_back(kv_per_pos_l[b] * kv_chunk_pos);
                        stream.push_back({it->second, KV});
                        stream_slot.push_back(s);
                    }
                }
                for (int e = 0; e < K; ++e) {
                    const uint32_t id = exp_base + (uint32_t) ((b * NE + t.slot[s]->ids[e]) * 2);
                    stream.push_back({id, EXPERT});
                    stream.push_back({id + 1, EXPERT});
                    stream_slot.push_back(s);
                    stream_slot.push_back(s);
                }
            }
            stream.push_back({head_id, HEAD});
            stream_slot.push_back(-1);
        }
        if (!n_tok) throw std::runtime_error("no decode tokens to simulate");

        uint64_t bytes_kind[NKIND] = {0, 0, 0, 0};
        for (const auto & x : stream) bytes_kind[x.kind] += osize[x.obj];
        uint64_t per_tok = 0;
        std::printf("\n[E] cache simulation over %llu decode tokens (fully associative, whole-object reads)\n",
                    (unsigned long long) n_tok);
        std::printf("    bytes read per decode token:\n");
        for (int k = 0; k < NKIND; ++k) {
            per_tok += bytes_kind[k] / n_tok;
            std::printf("      %-22s %10s\n", kind_name[k], mb(bytes_kind[k] / n_tok).c_str());
        }
        std::printf("      %-22s %10s   (one expert slab pair = %s at NVFP4 layers)\n", "total", mb(per_tok).c_str(),
                    mb(slab_gu[0] + slab_dn[0]).c_str());
        summary << "  \"bytes_per_decode_token\": {\"dense\": " << bytes_kind[DENSE] / n_tok << ", \"experts\": "
                << bytes_kind[EXPERT] / n_tok << ", \"kv\": " << bytes_kind[KV] / n_tok << ", \"head\": "
                << bytes_kind[HEAD] / n_tok << ", \"total\": " << per_tok << "},\n";

        const auto dist = moe::stack_distances(stream, osize);
        std::vector<uint64_t> caps;
        for (uint64_t c = 2; c <= 16384; c *= 2) caps.push_back(c << 20);
        const auto lru = moe::lru_hits(stream, osize, dist, caps, NKIND);
        Csv cs(out / "cachesim.csv");
        cs.row("policy", "capacity_mb", "hit_all", "hit_dense", "hit_experts", "hit_kv", "hit_head");
        std::printf("    LRU byte hit rate by capacity:\n      capacity    all   dense  experts     KV\n");
        for (const auto & h : lru) {
            cs.row("lru", h.capacity >> 20, num(h.hit_rate()), num(h.hit_rate(DENSE)), num(h.hit_rate(EXPERT)),
                   num(h.hit_rate(KV)), num(h.hit_rate(HEAD)));
            std::printf("      %6llu MB  %5.1f%%  %5.1f%%  %6.1f%%  %5.1f%%\n", (unsigned long long) (h.capacity >> 20),
                        100 * h.hit_rate(), 100 * h.hit_rate(DENSE), 100 * h.hit_rate(EXPERT), 100 * h.hit_rate(KV));
        }
        std::printf("    offline Belady (size-aware, upper-bound heuristic):\n");
        summary << "  \"cache\": {";
        for (uint64_t cap_mb : {2ull, 32ull, 256ull, 2048ull}) {
            const auto b = moe::belady_hits(stream, osize, cap_mb << 20, NKIND);
            cs.row("belady", cap_mb, num(b.hit_rate()), num(b.hit_rate(DENSE)), num(b.hit_rate(EXPERT)), num(b.hit_rate(KV)),
                   num(b.hit_rate(HEAD)));
            const auto & l = *std::find_if(lru.begin(), lru.end(), [&](const moe::HitStats & h) { return h.capacity == cap_mb << 20; });
            std::printf("      %6llu MB  %5.1f%% of all bytes, %5.1f%% of expert bytes   (LRU: %.1f%% / %.1f%%)\n",
                        (unsigned long long) cap_mb, 100 * b.hit_rate(), 100 * b.hit_rate(EXPERT), 100 * l.hit_rate(),
                        100 * l.hit_rate(EXPERT));
            summary << (cap_mb == 2 ? "" : ", ") << "\"" << cap_mb << "MB\": {\"lru_all\": " << num(l.hit_rate())
                    << ", \"lru_experts\": " << num(l.hit_rate(EXPERT)) << ", \"belady_all\": " << num(b.hit_rate())
                    << ", \"belady_experts\": " << num(b.hit_rate(EXPERT)) << "}";
        }
        summary << "},\n";

        // reuse-distance histogram for expert accesses, split by where the previous use was
        Csv rh(out / "reuse_hist.csv");
        rh.row("kind", "distance_bucket_mb_upper", "bytes");
        std::map<uint64_t, uint64_t> exp_hist, dense_hist;
        uint64_t exp_cold = 0;
        for (size_t i = 0; i < stream.size(); ++i) {
            if (stream[i].kind != EXPERT && stream[i].kind != DENSE) continue;
            auto & hm = stream[i].kind == EXPERT ? exp_hist : dense_hist;
            if (dist[i] == moe::kInfDistance) {
                if (stream[i].kind == EXPERT) exp_cold += osize[stream[i].obj];
                continue;
            }
            uint64_t bucket = 1;
            while ((bucket << 20) < dist[i] + osize[stream[i].obj]) bucket <<= 1;
            hm[bucket] += osize[stream[i].obj];
        }
        for (auto [bk, v] : exp_hist) rh.row("expert", bk, v);
        for (auto [bk, v] : dense_hist) rh.row("dense", bk, v);
        uint64_t median_bucket = 0, acc = 0, tot = 0;
        for (auto [bk, v] : exp_hist) tot += v;
        for (auto [bk, v] : exp_hist) {
            acc += v;
            if (acc * 2 >= tot) { median_bucket = bk; break; }
        }
        std::printf("    expert re-use distance (distinct bytes since last use + own size): median <= %llu MB; "
                    "%.1f%% of expert bytes are first-ever uses\n",
                    (unsigned long long) median_bucket, 100.0 * (double) exp_cold / (double) bytes_kind[EXPERT]);
        summary << "  \"expert_reuse_median_mb\": " << median_bucket << ",\n";
    }

    // ------------------------------------------------------------------ F. contiguity preview for moe-permute
    {
        std::vector<std::vector<std::vector<uint8_t>>> train(NP), test(NP);  // per phys layer: selections
        for (const auto & t : toks) {
            if (t.phase != moe::Phase::Decode) continue;
            auto & dst = (t.seq % 2 == 0) ? train : test;
            for (int s = 0; s < NS; ++s) dst[L.phys(s)].emplace_back(t.slot[s]->ids.begin(), t.slot[s]->ids.begin() + K);
        }
        moe::Permutation perm;
        if (!a.perm.empty()) perm = moe::read_permutation(a.perm);
        std::mt19937_64 rng(42);
        Csv ct(out / "contiguity.csv");
        ct.row("phys", "identity_runs", "random_runs", "clustered_runs_heldout", "perm_runs");
        double s_id = 0, s_rand = 0, s_cl = 0, s_perm = 0;
        for (int b = 0; b < NP; ++b) {
            moe::CoMatrix c(NE, std::vector<uint64_t>(NE, 0));
            for (const auto & sel : train[b])
                for (int i = 0; i < K; ++i)
                    for (int j = 0; j < K; ++j)
                        if (i != j) ++c[sel[i]][sel[j]];
            std::vector<int> ident(NE);
            std::iota(ident.begin(), ident.end(), 0);
            const double r_id = moe::mean_runs(test[b], ident);
            double r_rand = 0;
            for (int i = 0; i < 5; ++i) r_rand += moe::mean_runs(test[b], moe::random_order(NE, rng)) / 5;
            const double r_cl = moe::mean_runs(test[b], moe::cluster_order(c));
            const double r_perm = perm.empty() ? r_id : moe::mean_runs(test[b], perm.new_to_old.at(b));
            ct.row(b, num(r_id), num(r_rand), num(r_cl), num(r_perm));
            s_id += r_id; s_rand += r_rand; s_cl += r_cl; s_perm += r_perm;
        }
        std::printf("\n[F] contiguous runs per top-%d selection (held-out odd sequences; %d = no two experts adjacent)\n",
                    K, K);
        std::printf("    original order %.3f   random order %.3f   co-activation clustered %.3f", s_id / NP, s_rand / NP,
                    s_cl / NP);
        if (!perm.empty()) std::printf("   --perm %.3f", s_perm / NP);
        std::printf("\n");
        summary << "  \"contiguity_runs\": {\"identity\": " << num(s_id / NP) << ", \"random\": " << num(s_rand / NP)
                << ", \"clustered_heldout\": " << num(s_cl / NP);
        if (!perm.empty()) summary << ", \"perm\": " << num(s_perm / NP);
        summary << "}\n}\n";
    }

    std::ofstream(out / "summary.json") << summary.str();
    std::printf("\nwrote %s/{summary.json, *.csv}\n", out.string().c_str());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + s);
                return argv[++i];
            };
            if (s == "-m") a.model = next();
            else if (s == "-o") a.out = next();
            else if (s == "--perm") a.perm = next();
            else if (s == "--no-kv") a.kv = false;
            else if (a.trace.empty() && s[0] != '-') a.trace = s;
            else throw std::invalid_argument("unknown argument: " + s);
        }
        if (a.trace.empty()) throw std::invalid_argument("usage: moe-analyze trace.mtrace [-m model.gguf] [-o dir] [--perm p.txt] [--no-kv]");
        return run(a);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
