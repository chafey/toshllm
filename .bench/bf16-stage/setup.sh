#!/bin/zsh
# Scaffold the 0073 (bf16 staging) prototype worktree: apply 0001-0065+0072, configure, build.
set -u
WT=/Users/chafey/src/toshllm/.bench/bf16-stage/llama
P=/Users/chafey/src/toshllm/patches/llama
LOG=/tmp/bf16-setup.log
{
    echo "setup start $(date +%H:%M:%S)"
    cd $WT || exit 1
    git reset -q --hard 465e49b9c || exit 1
    for f in $P/00*.patch; do
        n=${${f##*/}:0:4}
        if (( 10#$n <= 65 )) || (( 10#$n == 72 )); then
            git apply "$f" || echo "APPLY_FAIL $f"
        fi
    done
    git --no-pager diff --stat | tail -1
    cmake -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_METAL=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF \
        "-DCMAKE_CXX_FLAGS=-DTOSH_VERSION=0.87.4-bf16proto" || { echo CONFIGURE_FAIL; exit 1; }
    echo CONFIGURED
    cmake --build build-static -j 16 -t llama-bench llama-completion && echo BUILD_OK || echo BUILD_FAIL
    echo "setup done $(date +%H:%M:%S)"
} >> $LOG 2>&1
