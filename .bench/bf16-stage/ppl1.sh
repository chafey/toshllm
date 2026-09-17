#!/bin/zsh
# TP4 perplexity smoke for patch 0073: limited-token corpus, knob OFF then ON.
# Checks rc=0, s5=0 and how much the f16 transport moves the loss.
set -u
BASE=/Users/chafey/src/toshllm/.bench/bf16-stage
WT=$BASE/llama
RES=$BASE/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
P=$WT/build-static/bin/llama-perplexity
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
CTX=/Users/chafey/src/toshllm/.bench/bf16-stage/ppl-corpus.txt
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

ppl() {
    local tag=$1 on=$2
    local e=$LOGD/$tag.err o=$LOGD/$tag.out rc
    local args=($P -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -c 512 --chunks 4 -f $CTX)
    if (( on )); then TOSH_MGPU_STAGE_F16=1 "${args[@]}" < /dev/null >$o 2>$e
    else env -u TOSH_MGPU_STAGE_F16 "${args[@]}" < /dev/null >$o 2>$e; fi
    rc=$?
    local p=$(grep -E 'Final estimate: PPL' $e | tail -1)
    echo "$tag rc=$rc s5=$(grep -c 'status 5' $e) ppl=[$p]" >> $RES
}

echo "=== ppl1 (stripped build) start $(date +%H:%M:%S)" >> $RES
ppl ppl_off 0
ppl ppl_on  1
echo "=== ppl1 done $(date +%H:%M:%S)" >> $RES
