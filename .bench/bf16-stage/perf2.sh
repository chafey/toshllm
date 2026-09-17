#!/bin/zsh
# Perf recheck of patch 0073 (f16 transport) on a (self-)recovered driver: interleaved
# pp512 A/B x2 + tg128 A/B, stderr per run, appended to results.txt.
set -u
RES=/Users/chafey/src/toshllm/.bench/bf16-stage/results.txt
LOGD=/Users/chafey/src/toshllm/.bench/bf16-stage/v3-logs
B=/Users/chafey/src/toshllm/.bench/bf16-stage/llama/build-static/bin/llama-bench
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1
pp() {
    local tag=$1 on=$2
    local e=$LOGD/$tag.err
    local v
    if (( on )); then
        v=$(TOSH_MGPU_STAGE_F16=1 $B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 1 -r 3 2>$e | awk -F'|' '/pp512/{gsub(/ /,"",$11); print $11}')
    else
        v=$(env -u TOSH_MGPU_STAGE_F16 $B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 1 -r 3 2>$e | awk -F'|' '/pp512/{gsub(/ /,"",$11); print $11}')
    fi
    echo "$tag pp512=$v s5=$(grep -c 'status 5' $e)" >> $RES
}
tg() {
    local tag=$1 on=$2
    local e=$LOGD/$tag.err
    local v
    if (( on )); then
        v=$(TOSH_MGPU_STAGE_F16=1 $B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 128 -n 128 -r 2 2>$e | awk -F'|' '/tg128/{gsub(/ /,"",$11); print $11}')
    else
        v=$(env -u TOSH_MGPU_STAGE_F16 $B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 128 -n 128 -r 2 2>$e | awk -F'|' '/tg128/{gsub(/ /,"",$11); print $11}')
    fi
    echo "$tag tg128=$v s5=$(grep -c 'status 5' $e)" >> $RES
}
echo "=== perf2 (healthy driver recheck) start $(date +%H:%M:%S)" >> $RES
pp p_off1 0; pp p_on1 1; pp p_off2 0; pp p_on2 1
tg t_off 0; tg t_on 1
echo "=== perf2 done $(date +%H:%M:%S)" >> $RES
