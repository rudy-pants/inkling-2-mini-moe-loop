// Loop layout of an inkling2_loop model: maps the graph's layer slots to physical blocks and loop passes.
//
// The graph tags every layer visit with a slot index `il` (see llama.cpp-src/src/models/inkling2-loop.cpp):
//   [0, core_start)                 prelude              (physical = il)
//   [core_start, core_end)          core, pass 0         (physical = il)
//   [core_end, n_phys)              coda                 (physical = il)
//   [n_phys, n_slots)               core, passes 1..P-1  (physical = core_start + (il - n_phys) % core_len)
// Execution order: prelude, pass 0, pass 1, ..., pass P-1, coda.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace moe {

struct LoopLayout {
    int n_phys     = 0; // physical blocks in the GGUF (block_count)
    int core_start = 0; // first looped block (loop.start)
    int core_end   = 0; // one past the last looped block (loop.end)
    int n_passes   = 0; // times the core runs (loop.count)

    int core_len() const { return core_end - core_start; }
    int n_slots() const { return n_phys + (n_passes - 1) * core_len(); }

    // Throws std::invalid_argument when the values cannot describe a loop.
    void validate() const {
        if (n_phys <= 0 || core_start < 0 || core_start >= core_end || core_end > n_phys || n_passes < 1) {
            throw std::invalid_argument("invalid loop layout: n_phys=" + std::to_string(n_phys) +
                                        " core=[" + std::to_string(core_start) + "," + std::to_string(core_end) +
                                        ") passes=" + std::to_string(n_passes));
        }
    }

    bool valid_slot(int il) const { return il >= 0 && il < n_slots(); }

    bool is_core(int il) const { return il >= n_phys || (il >= core_start && il < core_end); }

    int phys(int il) const {
        check(il);
        return il < n_phys ? il : core_start + (il - n_phys) % core_len();
    }

    // Loop pass of a slot, or -1 for prelude and coda slots.
    int pass(int il) const {
        check(il);
        if (il >= n_phys) return 1 + (il - n_phys) / core_len();
        return (il >= core_start && il < core_end) ? 0 : -1;
    }

    // Inverse of (phys, pass) for core blocks; pass is ignored for prelude/coda blocks.
    int slot(int phys_block, int pass_idx) const {
        if (phys_block < 0 || phys_block >= n_phys) throw std::out_of_range("physical block out of range");
        const bool core = phys_block >= core_start && phys_block < core_end;
        if (!core || pass_idx == 0) return phys_block;
        if (pass_idx < 0 || pass_idx >= n_passes) throw std::out_of_range("pass out of range");
        return n_phys + (pass_idx - 1) * core_len() + (phys_block - core_start);
    }

    // Slots in the order the graph evaluates them.
    std::vector<int> exec_order() const {
        std::vector<int> order;
        order.reserve(n_slots());
        for (int il = 0; il < core_start; ++il) order.push_back(il);
        for (int p = 0; p < n_passes; ++p) {
            for (int b = core_start; b < core_end; ++b) order.push_back(slot(b, p));
        }
        for (int il = core_end; il < n_phys; ++il) order.push_back(il);
        return order;
    }

    // Position of each slot within exec_order(): exec_index()[il] = step at which slot il runs.
    std::vector<int> exec_index() const {
        const auto order = exec_order();
        std::vector<int> idx(order.size());
        for (int i = 0; i < (int) order.size(); ++i) idx[order[i]] = i;
        return idx;
    }

private:
    void check(int il) const {
        if (!valid_slot(il)) throw std::out_of_range("slot " + std::to_string(il) + " out of range");
    }
};

} // namespace moe
