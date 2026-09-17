#!/bin/zsh
# ppl1 follow-up: interleaved off/on/off/on with wall timing, to check whether the
# apparent 2x slowness of the f16-transport pass is real or an ordering artifact.
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
    local e=$LOGD/$tag.err rc t0 t1
    local args=($P -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -c 512 --chunks 4 -f $CTX)
    t0=$SECONDS
    if (( on )); then TOSH_MGPU_STAGE_F16=1 "${args[@]}" < /dev/null >$LOGD/$tag.out 2>$e
    else env -u TOSH_MGPU_STAGE_F16 "${args[@]}" < /dev/null >$LOGD/$tag.out 2>$e; fi
    rc=$?; t1=$SECONDS
    echo "$tag rc=$rc s5=$(grep -c 'status 5' $e) wall=$((t1-t0))s ppl=[$(grep -E 'Final estimate: PPL' $e | tail -1 | sed 's/.*PPL/PPL/')]" >> $RES
}

echo "=== ppl2 interleaved start $(date +%H:%M:%S)" >> $RES
ppl p2_off1 0
ppl p2_on1  1
ppl p2_off2 0
ppl p2_on2  1
echo "=== ppl2 done $(date +%H:%M:%S)" >> $RES
