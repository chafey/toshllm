#!/bin/zsh
# Perf sanity for the STRIPPED 0073 build: interleaved pp512 A/B x2, tg128 A/B,
# then pp2048 A/B x2 (exercises the grow/realloc path of the tmp_f16 slots).
set -u
BASE=/Users/chafey/src/toshllm/.bench/bf16-stage
WT=$BASE/llama
RES=$BASE/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
B=$WT/build-static/bin/llama-bench
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

ppn() {
    local tag=$1 on=$2 p=$3 n=$4 r=$5
    local e=$LOGD/$tag.err v
    local args=($B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p $p -n $n -r $r)
    if (( on )); then v=$(TOSH_MGPU_STAGE_F16=1 "${args[@]}" 2>$e | awk -F'|' "/pp${p}|tg${n}/{gsub(/ /,\"\",\$11); print \$11}")
    else v=$(env -u TOSH_MGPU_STAGE_F16 "${args[@]}" 2>$e | awk -F'|' "/pp${p}|tg${n}/{gsub(/ /,\"\",\$11); print \$11}"); fi
    echo "$tag pp${p}/tg${n}=$v s5=$(grep -c 'status 5' $e)" >> $RES
}

echo "=== perf3 (stripped build) start $(date +%H:%M:%S)" >> $RES
ppn p512_off1 0 512 1 3
ppn p512_on1  1 512 1 3
ppn p512_off2 0 512 1 3
ppn p512_on2  1 512 1 3
ppn t128_off  0 128 128 2
ppn t128_on   1 128 128 2
ppn p2k_off1  0 2048 1 3
ppn p2k_on1   1 2048 1 3
ppn p2k_off2  0 2048 1 3
ppn p2k_on2   1 2048 1 3
echo "=== perf3 done $(date +%H:%M:%S)" >> $RES
