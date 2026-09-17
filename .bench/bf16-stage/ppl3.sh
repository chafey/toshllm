#!/bin/zsh
# PPL drift of the f16 transport (TOSH_MGPU_STAGE_F16) on the FULL current series
# (0001-0070+0072+0073): interleaved off/on on the same binary, same corpus as 0073 cert.
set -u
BASE=/Users/chafey/src/toshllm/.bench
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
P=$BASE/apply-check-0073/build-static/bin/llama-perplexity
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
CORPUS=$BASE/bf16-stage/ppl-corpus.txt
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1
ppl() {
    local tag=$1 on=$2
    local e=$LOGD/$tag.err
    local args=($P -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -c 512 --chunks 4 -f $CORPUS)
    if (( on )); then TOSH_MGPU_STAGE_F16=1 "${args[@]}" < /dev/null >$LOGD/$tag.out 2>$e
    else env -u TOSH_MGPU_STAGE_F16 "${args[@]}" < /dev/null >$LOGD/$tag.out 2>$e; fi
    echo "$tag rc=$? s5=$(grep -c 'status 5' $e) ppl=[$(grep -E 'Final estimate: PPL' $e | tail -1 | sed 's/.*PPL/PPL/')]" >> $RES
}
echo "=== ppl3 f16-drift on current series start $(date +%H:%M:%S)" >> $RES
ppl p3_off1 0; ppl p3_on1 1; ppl p3_off2 0; ppl p3_on2 1
echo "=== ppl3 done $(date +%H:%M:%S)" >> $RES
