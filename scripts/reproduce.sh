#!/usr/bin/env bash
# Reproduces the Phase 1-3 results in RESULTS.md. Stop other GPU users (llama-server, vLLM) first.
# Steps can be run individually: scripts/reproduce.sh trace|analyze|permute|qa|bench
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd .. && pwd)
MODEL=$ROOT/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf
BIN=$ROOT/llama-rocm
OUTM=$ROOT/models/permuted
step=${1:-all}

if [[ $step == all || $step == prompts ]]; then
    # needs a llama-server on :18089 for the chat template, e.g.
    #   $BIN/llama-server -m $MODEL -ngl 99 -c 4096 --jinja --port 18089 &
    python3 -I scripts/render_prompts.py
fi

if [[ $step == all || $step == trace ]]; then
    mkdir -p traces
    ./build/moe-trace -o traces/suite.mtrace --prompts-dir prompts/rendered -n 384 --temp 1.0 --seed 1234
fi

if [[ $step == all || $step == analyze ]]; then
    ./build/moe-analyze traces/suite.mtrace -o results/suite | tee results/suite-report.txt
    ./build/moe-report traces/suite.mtrace --analysis results/suite --phase3 bench/phase3.json   # -> results/suite/report.html
fi

if [[ $step == all || $step == permute ]]; then
    mkdir -p perms "$OUTM"
    for m in identity random clustered; do
        ./build/moe-permute plan -t traces/suite.mtrace --method $m --seed 7 -o perms/$m.txt
    done
    ./build/moe-permute apply -i "$MODEL" -p perms/identity.txt -o "$OUTM/identity.gguf"
    cmp "$MODEL" "$OUTM/identity.gguf" && echo "identity rewrite is byte-identical"
    rm "$OUTM/identity.gguf"
    for m in random clustered; do
        ./build/moe-permute apply -i "$MODEL" -p perms/$m.txt -o "$OUTM/$m.gguf"
        ./build/moe-permute verify -i "$MODEL" -p perms/$m.txt -o "$OUTM/$m.gguf"
    done
fi

if [[ $step == all || $step == qa ]]; then
    C=bench/ppl-corpus.txt
    # fixed English/technical corpus: the larger markdown docs of the llama.cpp source tree
    [[ -s $C ]] || (cd "$ROOT/llama.cpp-src" && find . -name "*.md" -size +8k | grep -v -E "vendor|third_party" | head -30 |
                    xargs cat | head -c 600000) > $C
    B=/tmp/kld-base.bin
    $BIN/llama-perplexity -m "$MODEL" -f $C -c 512 --chunks 20 -ngl 99 -fa on --kl-divergence-base $B | grep "Final estimate"
    kld() { $BIN/llama-perplexity -m "$1" -f $C -c 512 --chunks 20 -ngl 99 -fa on ${2:-} --kl-divergence-base $B --kl-divergence 2>&1 |
            grep -E "^Mean +KLD|^Same top p|^Mean +PPL\(Q\)/PPL\(base\)"; }
    echo "control (-ub 128)"; kld "$MODEL" "-ub 128"
    for m in clustered random; do echo "$m"; kld "$OUTM/$m.gguf"; done
    rm -f $B
fi

if [[ $step == all || $step == bench ]]; then
    O=$MODEL R=$OUTM/random.gguf Cl=$OUTM/clustered.gguf
    for order in "$O $R $Cl" "$Cl $R $O"; do
        args=(); for m in $order; do args+=(-m "$m"); done
        $BIN/llama-bench "${args[@]}" -ngl 99 -fa on -p 512 -n 128 -r 5 -o csv 2>/dev/null
    done > bench/bench-raw.csv
    echo "wrote bench/bench-raw.csv"
fi
