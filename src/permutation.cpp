#include "moe/permutation.hpp"

#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace moe {

std::vector<int> Permutation::old_to_new(int layer) const {
    std::vector<int> inv(n_expert);
    if (empty()) {
        std::iota(inv.begin(), inv.end(), 0);
        return inv;
    }
    const auto & row = new_to_old.at(layer);
    for (int i = 0; i < n_expert; ++i) inv[row[i]] = i;
    return inv;
}

void Permutation::validate() const {
    for (size_t l = 0; l < new_to_old.size(); ++l) {
        const auto & row = new_to_old[l];
        if ((int) row.size() != n_expert) throw std::invalid_argument("permutation layer " + std::to_string(l) + " has wrong length");
        std::vector<char> seen(n_expert, 0);
        for (int v : row) {
            if (v < 0 || v >= n_expert || seen[v]) {
                throw std::invalid_argument("permutation layer " + std::to_string(l) + " is not a permutation");
            }
            seen[v] = 1;
        }
    }
}

Permutation Permutation::identity(int n_layer, int n_expert) {
    Permutation p;
    p.n_expert = n_expert;
    p.new_to_old.assign(n_layer, std::vector<int>(n_expert));
    for (auto & row : p.new_to_old) std::iota(row.begin(), row.end(), 0);
    return p;
}

void write_permutation(const std::string & path, const Permutation & p) {
    p.validate();
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "moe-loop-permutation v1 " << p.new_to_old.size() << " " << p.n_expert << "\n";
    for (size_t l = 0; l < p.new_to_old.size(); ++l) {
        out << l << ":";
        for (int v : p.new_to_old[l]) out << " " << v;
        out << "\n";
    }
    if (!out) throw std::runtime_error("write failed: " + path);
}

Permutation read_permutation(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string magic, ver;
    size_t n_layer = 0;
    Permutation p;
    in >> magic >> ver >> n_layer >> p.n_expert;
    if (magic != "moe-loop-permutation" || ver != "v1") throw std::runtime_error(path + ": not a v1 permutation file");
    p.new_to_old.assign(n_layer, std::vector<int>(p.n_expert));
    for (size_t l = 0; l < n_layer; ++l) {
        std::string tag;
        in >> tag;
        if (tag != std::to_string(l) + ":") throw std::runtime_error(path + ": expected layer " + std::to_string(l));
        for (int i = 0; i < p.n_expert; ++i) in >> p.new_to_old[l][i];
    }
    if (!in) throw std::runtime_error(path + ": truncated");
    p.validate();
    return p;
}

} // namespace moe
