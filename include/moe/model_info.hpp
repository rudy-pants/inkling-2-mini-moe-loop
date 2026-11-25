// Reads and checks the GGUF metadata this project depends on.
#pragma once

#include "moe/layout.hpp"

#include <optional>
#include <string>

struct llama_model;

namespace moe {

struct ModelInfo {
    std::string arch;
    int n_expert      = 0;
    int n_expert_used = 0;
    int n_slots_graph = 0; // llama_model_n_layer(): slot count as seen by the graph
    LoopLayout layout;
};

// Reads metadata from a loaded model. Throws std::runtime_error when a key is missing or unparsable.
ModelInfo read_model_info(const llama_model * model);

// Values the tools were written against (Inkling 2 Mini).
struct Expected {
    std::string arch = "inkling2_loop";
    int n_expert = 128;
    int n_expert_used = 8;
    LoopLayout layout{24, 6, 18, 3};
};

// Throws std::runtime_error describing every mismatch; returns normally when everything matches.
void check_model_info(const ModelInfo & info, const Expected & expected = {});

// Reads one metadata value as a string, or nullopt when the key is absent.
std::optional<std::string> meta_str(const llama_model * model, const std::string & key);

} // namespace moe
