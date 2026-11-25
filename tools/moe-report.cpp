// moe-report: expert profiles ("what is each expert used for?") plus a self-contained HTML report.
//
// usage: moe-report trace.mtrace [-m model.gguf] [--analysis results/<trace>] [--phase3 bench/phase3.json]
//                   [--template viz/report.html.in] [-o results/<trace>/report.html] [--perms 5000]
//
// Per (physical layer, expert) it computes, against that layer's baseline:
//   * category lift: share of the expert's selections from each prompt category / that category's share of all
//     selections in the layer, with a sequence-level permutation test (category labels shuffled across prompts,
//     so a single unusual prompt cannot fake a category effect) and Benjamini-Hochberg q-values;
//   * token-type lift: same ratio over coarse token classes (CJK, digits, punctuation, ...);
//   * top tokens: tokens with the highest selection rate relative to the expert's overall rate;
//   * loop-pass and prefill/decode shares.
// Writes expert_profiles.csv and the HTML report (data embedded as JSON, no external resources).

#include "moe/gguf_layout.hpp"
#include "moe/json.hpp"
#include "moe/tokens.hpp"
#include "moe/trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef MOE_DEFAULT_MODEL
#define MOE_DEFAULT_MODEL ""
#endif
#ifndef MOE_SOURCE_DIR
#define MOE_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;
using moe::json::escape;
using moe::json::num;

namespace {

constexpr int kNClass = (int) moe::TokenClass::Count;
constexpr int kMinSeqsForTest = 3;   // categories with fewer prompts are reported but not tested
constexpr int kMinTokenRecords = 12; // a token must appear in at least this many records of a layer to be ranked
constexpr int kTopTokens = 16;

struct Args {
    std::string trace, model = MOE_DEFAULT_MODEL, analysis, phase3, out;
    std::string tmpl = std::string(MOE_SOURCE_DIR) + "/viz/report.html.in";
    int perms = 5000;
};

std::string slurp(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// CSV (no quoting; our analyzer never emits commas inside fields) -> JSON {"cols":[...],"rows":[[...],...]}
std::string csv_to_json(const fs::path & p) {
    std::ifstream f(p);
    if (!f) return "null";
    std::string line;
    std::ostringstream o;
    bool header = true, first_row = true;
    auto field = [](const std::string & v) {
        if (v.empty()) return std::string("null");
        char * end = nullptr;
        std::strtod(v.c_str(), &end);
        if (end && *end == '\0') return v;
        return escape(v);
    };
    o << "{";
    while (std::getline(f, line)) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) cells.push_back(c);
        if (header) {
            o << "\"cols\":[";
            for (size_t i = 0; i < cells.size(); ++i) o << (i ? "," : "") << escape(cells[i]);
            o << "],\"rows\":[";
            header = false;
            continue;
        }
        o << (first_row ? "" : ",") << "[";
        for (size_t i = 0; i < cells.size(); ++i) o << (i ? "," : "") << field(cells[i]);
        o << "]";
        first_row = false;
    }
    o << "]}";
    return o.str();
}

struct SeqInfo {
    std::string name = "?", category = "?";
};

std::map<uint32_t, SeqInfo> read_sidecar(const std::string & path) {
    std::map<uint32_t, SeqInfo> out;
    std::ifstream in(path);
    static const std::regex re(R"re(^\s*\{"seq": (\d+), "name": "([^"]*)", "category": "([^"]*)")re");
    std::string line;
    std::smatch m;
    while (std::getline(in, line)) {
        if (std::regex_search(line, m, re)) out[(uint32_t) std::stoul(m[1])] = {m[2], m[3]};
    }
    return out;
}

// Benjamini-Hochberg q-values for a list of p-values (NaN entries are skipped and stay NaN).
std::vector<double> bh_qvalues(const std::vector<double> & p) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < p.size(); ++i) if (!std::isnan(p[i])) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return p[a] < p[b]; });
    std::vector<double> q(p.size(), std::nan(""));
    const double m = (double) idx.size();
    double running = 1.0;
    for (size_t r = idx.size(); r-- > 0;) {
        running = std::min(running, p[idx[r]] * m / (double) (r + 1));
        q[idx[r]] = running;
    }
    return q;
}

std::string visible(const std::string & s) {
    std::string o;
    std::string in = s;
    for (size_t p = 0; (p = in.find("\xC2\xA0", p)) != std::string::npos;) in.replace(p, 2, "\xE2\x8D\xBD");  // NBSP -> ⍽
    for (char c : in) {
        if (c == ' ') o += "\xC2\xB7";           // middle dot
        else if (c == '\n') o += "\xE2\x8F\x8E"; // return symbol
        else if (c == '\t') o += "\xE2\x87\xA5";
        else o += c;
    }
    return o;
}

int run(const Args & a) {
    const auto trace = moe::read_trace(a.trace);
    const auto & H = trace.header;
    const auto & L = H.layout;
    const int NE = (int) H.n_expert, K = (int) H.k, NP = L.n_phys, NPASS = L.n_passes;
    const fs::path analysis = a.analysis.empty() ? fs::path("results") / fs::path(a.trace).stem() : fs::path(a.analysis);
    const fs::path out = a.out.empty() ? analysis / "report.html" : fs::path(a.out);
    fs::create_directories(out.parent_path().empty() ? fs::path(".") : out.parent_path());

    // ---------------------------------------------------------------- sequences and categories
    const auto seqinfo = read_sidecar(a.trace + ".json");
    uint32_t n_seq = 0;
    for (const auto & r : trace.records) n_seq = std::max(n_seq, r.seq + 1);
    std::vector<std::string> cats;
    std::vector<int> cat_of(n_seq, 0);
    for (uint32_t s = 0; s < n_seq; ++s) {
        const auto it = seqinfo.find(s);
        const std::string c = it == seqinfo.end() ? "?" : it->second.category;
        auto pos = std::find(cats.begin(), cats.end(), c);
        if (pos == cats.end()) { cats.push_back(c); pos = cats.end() - 1; }
        cat_of[s] = (int) (pos - cats.begin());
    }
    const int NC = (int) cats.size();
    std::vector<int> cat_nseq(NC, 0);
    for (uint32_t s = 0; s < n_seq; ++s) ++cat_nseq[cat_of[s]];

    // ---------------------------------------------------------------- vocabulary
    const auto vocab = moe::read_string_array(a.model, "tokenizer.ggml.tokens");
    const auto gm = moe::open_gguf(a.model);
    std::vector<int64_t> ttype;
    if (auto it = gm.kv_int_arrays.find("tokenizer.ggml.token_type"); it != gm.kv_int_arrays.end()) ttype = it->second;
    std::unordered_map<int32_t, std::pair<std::string, moe::TokenClass>> tokinfo;
    auto token = [&](int32_t id) -> const std::pair<std::string, moe::TokenClass> & {
        auto it = tokinfo.find(id);
        if (it != tokinfo.end()) return it->second;
        std::string bytes = id >= 0 && id < (int32_t) vocab.size() ? moe::bpe_decode(vocab[id]) : "?";
        const bool special = id >= 0 && id < (int32_t) ttype.size() && (ttype[id] == 3 || ttype[id] == 4);
        return tokinfo.emplace(id, std::make_pair(bytes, moe::classify_token(bytes, special))).first->second;
    };

    // ---------------------------------------------------------------- counting
    const size_t NLE = (size_t) NP * NE;
    std::vector<std::vector<uint32_t>> cnt_seq(n_seq, std::vector<uint32_t>(NLE, 0));  // selections
    std::vector<std::vector<uint64_t>> tot_seq(n_seq, std::vector<uint64_t>(NP, 0));
    std::vector<std::array<uint64_t, kNClass>> cls_cnt(NLE), cls_tot(NP);
    for (auto & x : cls_cnt) x.fill(0);
    for (auto & x : cls_tot) x.fill(0);
    std::vector<std::vector<uint64_t>> pass_cnt(NLE, std::vector<uint64_t>(NPASS, 0));
    std::vector<std::array<uint64_t, 2>> phase_cnt(NLE), phase_tot(NP);
    for (auto & x : phase_cnt) x.fill(0);
    for (auto & x : phase_tot) x.fill(0);
    std::vector<uint64_t> layer_records(NP, 0);
    std::vector<std::unordered_map<int32_t, std::vector<uint32_t>>> tok_sel(NP);  // [layer][token] -> per expert
    std::vector<std::unordered_map<int32_t, uint32_t>> tok_rec(NP);
    std::map<std::string, uint64_t> class_tokens;  // distinct token ids per class (vocabulary coverage of the trace)

    for (const auto & r : trace.records) {
        const int b = L.phys(r.slot), ps = L.pass(r.slot);
        const auto & ti = token(r.token);
        const int cl = (int) ti.second, ph = (int) r.phase;
        ++layer_records[b];
        ++tok_rec[b][r.token];
        auto & ts = tok_sel[b][r.token];
        if (ts.empty()) ts.assign(NE, 0);
        for (int k = 0; k < K; ++k) {
            const int e = r.ids[k];
            const size_t le = (size_t) b * NE + e;
            ++cnt_seq[r.seq][le];
            ++tot_seq[r.seq][b];
            ++cls_cnt[le][cl];
            ++cls_tot[b][cl];
            ++phase_cnt[le][ph];
            ++phase_tot[b][ph];
            if (ps >= 0) ++pass_cnt[le][ps];
            ++ts[e];
        }
    }

    // ---------------------------------------------------------------- category lift + permutation test
    auto lifts_for = [&](const std::vector<int> & labels, std::vector<double> & lift) {
        // lift[(le) * NC + c]
        std::vector<double> cat_tot((size_t) NP * NC, 0), all_tot(NP, 0);
        std::vector<double> c_cnt(NLE * NC, 0), a_cnt(NLE, 0);
        for (uint32_t s = 0; s < n_seq; ++s) {
            const int c = labels[s];
            for (int b = 0; b < NP; ++b) {
                cat_tot[(size_t) b * NC + c] += (double) tot_seq[s][b];
                all_tot[b] += (double) tot_seq[s][b];
            }
            const auto & cs = cnt_seq[s];
            for (size_t le = 0; le < NLE; ++le) {
                c_cnt[le * NC + c] += cs[le];
                a_cnt[le] += cs[le];
            }
        }
        lift.assign(NLE * NC, 0);
        for (size_t le = 0; le < NLE; ++le) {
            const int b = (int) (le / NE);
            for (int c = 0; c < NC; ++c) {
                const double base = cat_tot[(size_t) b * NC + c] / all_tot[b];
                lift[le * NC + c] = a_cnt[le] > 0 && base > 0 ? (c_cnt[le * NC + c] / a_cnt[le]) / base : 0;
            }
        }
    };
    std::vector<double> lift_obs;
    lifts_for(cat_of, lift_obs);
    std::vector<uint32_t> ge(NLE * NC, 0);
    {
        std::mt19937_64 rng(20261124);
        std::vector<int> perm = cat_of, lift_tmp_labels;
        std::vector<double> lp;
        std::fprintf(stderr, "moe-report: %d permutations over %u prompts...\n", a.perms, n_seq);
        for (int p = 0; p < a.perms; ++p) {
            std::shuffle(perm.begin(), perm.end(), rng);
            lifts_for(perm, lp);
            for (size_t i = 0; i < lp.size(); ++i) ge[i] += lp[i] >= lift_obs[i] - 1e-12;
        }
    }
    std::vector<double> pval(NLE * NC, std::nan(""));
    for (size_t i = 0; i < pval.size(); ++i) {
        const int c = (int) (i % NC);
        if (cat_nseq[c] >= kMinSeqsForTest && lift_obs[i] > 1.0) pval[i] = (1.0 + ge[i]) / (1.0 + a.perms);
    }
    const auto qval = bh_qvalues(pval);

    // ---------------------------------------------------------------- JSON assembly
    std::ostringstream J;
    J << "{\n\"meta\":{\"trace\":" << escape(a.trace) << ",\"model\":" << escape(a.model) << ",\"records\":"
      << trace.records.size() << ",\"n_expert\":" << NE << ",\"k\":" << K << ",\"n_phys\":" << NP << ",\"core\":["
      << L.core_start << "," << L.core_end << "],\"n_passes\":" << NPASS << ",\"perms\":" << a.perms
      << ",\"min_seqs_for_test\":" << kMinSeqsForTest << ",\"generated\":";
    {
        char buf[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", std::localtime(&now));
        J << escape(buf);
    }
    J << ",\"categories\":[";
    for (int c = 0; c < NC; ++c) {
        uint64_t toks = 0;
        for (uint32_t s = 0; s < n_seq; ++s) if (cat_of[s] == c) toks += tot_seq[s][0] / (uint64_t) K;
        J << (c ? "," : "") << "{\"name\":" << escape(cats[c]) << ",\"n_seqs\":" << cat_nseq[c] << ",\"tokens\":" << toks
          << ",\"tested\":" << (cat_nseq[c] >= kMinSeqsForTest ? "true" : "false") << "}";
    }
    J << "],\"classes\":[";
    for (int c = 0; c < kNClass; ++c) J << (c ? "," : "") << escape(moe::token_class_name((moe::TokenClass) c));
    J << "],\"prompts\":[";
    for (uint32_t s = 0; s < n_seq; ++s) {
        const auto it = seqinfo.find(s);
        J << (s ? "," : "") << "{\"name\":" << escape(it == seqinfo.end() ? "?" : it->second.name) << ",\"category\":"
          << escape(cats[cat_of[s]]) << ",\"tokens\":" << tot_seq[s][0] / (uint64_t) K << "}";
    }
    J << "]},\n";

    // class baseline per layer (share of selections)
    J << "\"class_base\":[";
    for (int b = 0; b < NP; ++b) {
        const double t = (double) std::accumulate(cls_tot[b].begin(), cls_tot[b].end(), (uint64_t) 0);
        J << (b ? "," : "") << "[";
        for (int c = 0; c < kNClass; ++c) J << (c ? "," : "") << num((double) cls_tot[b][c] / t, 5);
        J << "]";
    }
    J << "],\n\"experts\":[\n";

    std::ofstream csv(analysis / "expert_profiles.csv");
    csv << "phys,expert,selections,share,best_category,best_category_lift,best_category_q,best_class,best_class_lift,"
           "top_tokens\n";
    int n_sig_experts = 0, n_sig_cells = 0;
    std::vector<int> sig_per_cat(NC, 0);
    for (int b = 0; b < NP; ++b) {
        const double layer_sel = (double) layer_records[b] * K;
        const double cls_layer_tot = (double) std::accumulate(cls_tot[b].begin(), cls_tot[b].end(), (uint64_t) 0);
        for (int e = 0; e < NE; ++e) {
            const size_t le = (size_t) b * NE + e;
            uint64_t n = 0;
            for (uint32_t s = 0; s < n_seq; ++s) n += cnt_seq[s][le];
            const double share = (double) n / layer_sel;  // fraction of all selections in the layer
            J << (le ? ",\n" : "") << "{\"l\":" << b << ",\"e\":" << e << ",\"n\":" << n << ",\"share\":" << num(share, 5);
            // categories
            J << ",\"cl\":[";
            int best_c = -1;
            double best_l = 0, best_q = 1;
            bool any_sig = false;
            for (int c = 0; c < NC; ++c) {
                const double l = lift_obs[le * NC + c];
                J << (c ? "," : "") << num(l, 4);
                const double q = qval[le * NC + c];
                if (!std::isnan(q) && q < 0.05 && l >= 1.5) {
                    any_sig = true;
                    ++n_sig_cells;
                    ++sig_per_cat[c];
                }
                if (cat_nseq[c] >= kMinSeqsForTest && l > best_l) { best_l = l; best_c = c; best_q = std::isnan(q) ? 1 : q; }
            }
            n_sig_experts += any_sig;
            J << "],\"cq\":[";
            for (int c = 0; c < NC; ++c) {
                const double q = qval[le * NC + c];
                J << (c ? "," : "") << (std::isnan(q) ? "null" : num(q, 3));
            }
            // token classes
            J << "],\"tl\":[";
            int best_t = -1;
            double best_tl = 0;
            for (int c = 0; c < kNClass; ++c) {
                const double base = (double) cls_tot[b][c] / cls_layer_tot;
                const double expct = base * (double) n;
                const double l = expct > 0 ? (double) cls_cnt[le][c] / expct : 0;
                J << (c ? "," : "") << num(l, 4);
                if (expct >= 20 && l > best_tl) { best_tl = l; best_t = c; }
            }
            J << "],\"te\":[";  // expected counts (to judge support)
            for (int c = 0; c < kNClass; ++c) J << (c ? "," : "") << (int) std::lround((double) cls_tot[b][c] / cls_layer_tot * (double) n);
            // passes and phase
            J << "],\"ps\":";
            if (b >= L.core_start && b < L.core_end && n > 0) {
                J << "[";
                for (int p = 0; p < NPASS; ++p) J << (p ? "," : "") << num((double) pass_cnt[le][p] / (double) n, 4);
                J << "]";
            } else {
                J << "null";
            }
            const double pre_base = (double) phase_tot[b][0] / (double) (phase_tot[b][0] + phase_tot[b][1]);
            J << ",\"pre\":" << num(n ? ((double) phase_cnt[le][0] / (double) n) / pre_base : 0, 4);
            // top tokens
            const double rate_e = (double) n / (double) layer_records[b];
            std::vector<std::pair<double, int32_t>> cand;
            for (const auto & [tok, recs] : tok_rec[b]) {
                if (recs < (uint32_t) kMinTokenRecords) continue;
                const uint32_t hits = tok_sel[b].at(tok)[e];
                if (hits < 5) continue;
                cand.emplace_back(((double) hits / recs) / rate_e, tok);
            }
            std::partial_sort(cand.begin(), cand.begin() + std::min<size_t>(kTopTokens, cand.size()), cand.end(),
                              [](auto & x, auto & y) { return x.first > y.first; });
            J << ",\"top\":[";
            std::string top_csv;
            for (size_t i = 0; i < std::min<size_t>(kTopTokens, cand.size()); ++i) {
                const auto tok = cand[i].second;
                const auto & ti = token(tok);
                J << (i ? "," : "") << "[" << escape(visible(ti.first)) << "," << num(cand[i].first, 3) << ","
                  << tok_sel[b].at(tok)[e] << "," << tok_rec[b].at(tok) << "," << (int) ti.second << "]";
                if (i < 4) top_csv += (i ? " | " : "") + visible(ti.first);
            }
            J << "]}";
            std::string top_clean;
            for (char ch : top_csv) top_clean += ch == ',' ? ';' : ch;
            csv << b << "," << e << "," << n << "," << num(share, 5) << "," << (best_c >= 0 ? cats[best_c] : "") << ","
                << num(best_l, 4) << "," << num(best_q, 3) << "," << (best_t >= 0 ? moe::token_class_name((moe::TokenClass) best_t) : "")
                << "," << num(best_tl, 4) << "," << top_clean << "\n";
        }
    }
    J << "\n],\n";

    // analysis CSVs and other inputs, embedded verbatim
    const std::string summary = fs::exists(analysis / "summary.json") ? slurp(analysis / "summary.json") : "null";
    J << "\"summary\":" << summary << ",\n";
    for (const char * name : {"popularity", "layer_stats", "loop_overlap", "loop_overlap_hist", "token_lag", "window_union",
                              "category_js", "cachesim", "contiguity"}) {
        J << "\"" << name << "\":" << csv_to_json(analysis / (std::string(name) + ".csv")) << ",\n";
    }
    J << "\"phase3\":" << (!a.phase3.empty() && fs::exists(a.phase3) ? slurp(a.phase3) : "null") << "\n}";

    // ---------------------------------------------------------------- HTML
    std::string data = J.str();
    for (size_t pos = 0; (pos = data.find("</", pos)) != std::string::npos; pos += 3) data.replace(pos, 2, "<\\/");
    std::string html = slurp(a.tmpl);
    const std::string marker = "/*__MOE_DATA__*/null";
    const auto at = html.find(marker);
    if (at == std::string::npos) throw std::runtime_error("template has no " + marker + " marker");
    html.replace(at, marker.size(), data);
    std::ofstream(out, std::ios::binary) << html;
    {
        std::ofstream(analysis / "report-data.json") << J.str();
    }

    std::printf("expert profiles: %d of %zu layer-experts have a significant category enrichment "
                "(lift >= 1.5, BH q < 0.05; %d cells)\n", n_sig_experts, NLE, n_sig_cells);
    for (int c = 0; c < NC; ++c) {
        std::printf("  %-14s %2d prompts %s  significant layer-experts: %d\n", cats[c].c_str(), cat_nseq[c],
                    cat_nseq[c] >= kMinSeqsForTest ? "      " : "(untested)", sig_per_cat[c]);
    }
    std::printf("wrote %s, %s, %s\n", out.string().c_str(), (analysis / "expert_profiles.csv").string().c_str(),
                (analysis / "report-data.json").string().c_str());
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
            else if (s == "--analysis") a.analysis = next();
            else if (s == "--phase3") a.phase3 = next();
            else if (s == "--template") a.tmpl = next();
            else if (s == "-o") a.out = next();
            else if (s == "--perms") a.perms = std::stoi(next());
            else if (a.trace.empty() && s[0] != '-') a.trace = s;
            else throw std::invalid_argument("unknown argument: " + s);
        }
        if (a.trace.empty()) {
            throw std::invalid_argument("usage: moe-report trace.mtrace [-m model.gguf] [--analysis dir] [--phase3 f.json] "
                                        "[--template t.html.in] [-o report.html] [--perms N]");
        }
        return run(a);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
