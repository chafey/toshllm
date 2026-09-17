#!/bin/zsh
# Round 2 of the 30B cross-model A/B (round 1 had a_no72/b_w72off + the traced c retry).
set -u
BASE=/Users/chafey/src/toshllm/.bench
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
BINW=$BASE/apply-check-0073/build-static/bin
BIN72=$BASE/ab0072-main/build-static/bin
M30=/Users/chafey/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1
gb() {
    local tag=$1 bin=$2 on=$3
    local out=$LOGD/$tag.out e=$LOGD/$tag.err
    if (( on )); then TOSH_MGPU_STAGE_F16=1 $bin/llama-bench -m $M30 -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 128 -r 2 >$out 2>$e &
    else env -u TOSH_MGPU_STAGE_F16 $bin/llama-bench -m $M30 -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 128 -r 2 >$out 2>$e & fi
    local p=$! w=0
    while kill -0 $p 2>/dev/null; do
        (( w >= 300 )) && { kill -9 $p 2>/dev/null; wait $p 2>/dev/null; echo "$tag TIMEOUT@300s" >> $RES; return 124; }
        sleep 5; (( w += 5 ))
    done
    wait $p; local rc=$?
    echo "$tag rc=$rc vals=[$(awk -F'|' '/pp512|tg128/{gsub(/ /,"",$11); printf "%s ", $11}' $out)] s5=$(grep -c 'status 5' $e)" >> $RES
}
echo "=== ab-models 30B round2 start $(date +%H:%M:%S)" >> $RES
gb m30_r2_a_no72   $BIN72 0
gb m30_r2_b_w72off $BINW  0
gb m30_r2_c_w72on  $BINW  1
echo "=== ab-models 30B round2 done $(date +%H:%M:%S)" >> $RES
