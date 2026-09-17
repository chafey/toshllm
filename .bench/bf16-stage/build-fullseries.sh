#!/bin/zsh
# Full-series engine build (0001-0070 + 0072 + 0073, as applied by scripts/build-engines.sh
# from this branch) to close the PR's build-engines.sh box. Same cmake flags as
# .bench/bf16-stage/setup.sh, which produced the certified 0073 binary.
set -u
WT=/Users/chafey/src/toshllm/.bench/apply-check-0073
LOG=/Users/chafey/src/toshllm/.bench/apply-check-0073/build.log
{
    echo "full-series build start $(date +%H:%M:%S)"
    cd "$WT" || exit 1
    git --no-pager diff --stat | tail -1
    cmake -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_METAL=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF \
        "-DCMAKE_CXX_FLAGS=-DTOSH_VERSION=0.87.5-fullseries-check" || { echo CONFIGURE_FAIL; exit 1; }
    echo CONFIGURED
    cmake --build build-static -j 16 -t llama-bench llama-completion llama-perplexity \
        && echo BUILD_OK || echo BUILD_FAIL
    echo "full-series build done $(date +%H:%M:%S)"
} >> "$LOG" 2>&1
