#!/bin/zsh
# Verify the bf16 allreduce transport (TOSH_MGPU_STAGE_BF16) with the worktree binary.
# Worktree == vendor series + patch 0073 (knob default OFF), so r?_off must reproduce the
# vendor baseline (~1560 pp512) and r?_on is the hypothesis under test.
set -u
BASE=/Users/chafey/src/toshllm/.bench/bf16-stage
WT=$BASE/llama
RES=$BASE/results.txt
B=$WT/build-static/bin/llama-bench
C=$WT/build-static/bin/llama-completion
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
CTX=/Users/chafey/src/toshllm/.bench/prefill-bisect/ctx.txt
SHORT=/Users/chafey/src/toshllm/.bench/prefill-bisect/short.txt
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

pp() { local tag=$1; shift
    local v=$(env "$@" $B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 1 -r 3 2>/dev/null | awk -F'|' '/pp512/{gsub(/ /,"",$11); print $11}')
    echo "$tag pp512=$v" >> $RES
}
tg() { local tag=$1; shift
    local v=$(env "$@" $B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 128 -n 128 -r 2 2>/dev/null | awk -F'|' '/tg128/{gsub(/ /,"",$11); print $11}')
    echo "$tag tg128=$v" >> $RES
}
needle() { local tag=$1; shift
    local txt=$(env "$@" $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX 2>/dev/null | tr -d '\n' | cut -c1-60)
    echo "$tag needle=[$txt]" >> $RES
}
short() { local tag=$1; shift
    local txt=$(env "$@" $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 8 --no-display-prompt -f $SHORT 2>/dev/null | tr -d '\n' | cut -c1-40)
    echo "$tag short=[$txt]" >> $RES
}

echo "=== bf16-stage verify start $(date +%H:%M:%S)" >> $RES
pp r1_off
pp r1_on  TOSH_MGPU_STAGE_BF16=1
pp r2_off
pp r2_on  TOSH_MGPU_STAGE_BF16=1
needle on_needle TOSH_MGPU_STAGE_BF16=1
short  on_short  TOSH_MGPU_STAGE_BF16=1
needle off_needle
tg off_tg
tg on_tg  TOSH_MGPU_STAGE_BF16=1
echo "=== bf16-stage verify done $(date +%H:%M:%S)" >> $RES
