#!/bin/zsh
# Follow-up: 30B f16-ON hang isolation (retry once, traced) + Qwen3-VL-2B control.
# Every llama-bench call is guarded by a 300 s timeout (0% CPU hangs must not wedge the suite).
set -u
BASE=/Users/chafey/src/toshllm/.bench
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
BINW=$BASE/apply-check-0073/build-static/bin
BIN72=$BASE/ab0072-main/build-static/bin
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

guarded_bench() {
    local tag=$1 bin=$2 model=$3 on=$4; shift 4
    local out=$LOGD/$tag.out e=$LOGD/$tag.err
    local -a args
    args=($bin/llama-bench -m $model -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 128 -r 2 "$@")
    if (( on )); then TOSH_MGPU_STAGE_F16=1 "${args[@]}" >$out 2>$e &
    else env -u TOSH_MGPU_STAGE_F16 "${args[@]}" >$out 2>$e & fi
    local p=$! w=0
    while kill -0 $p 2>/dev/null; do
        (( w >= 300 )) && { kill -9 $p 2>/dev/null; wait $p 2>/dev/null; echo "$tag TIMEOUT@300s (0%CPU hang)" >> $RES; return 124; }
        sleep 5; (( w += 5 ))
    done
    wait $p; local rc=$?
    local v=$(awk -F'|' '/pp512|tg128/{gsub(/ /,"",$11); printf "%s ", $11}' $out)
    echo "$tag rc=$rc vals=[$v] s5=$(grep -c 'status 5' $e)" >> $RES
    return $rc
}

M30=/Users/chafey/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
M02=/Users/chafey/models/Qwen3-VL-2B-Instruct-Q8_0.gguf

echo "=== ab-models2 start $(date +%H:%M:%S) (30B f16 retry + 2B control, 300s guard)" >> $RES
TOSH_MGPU_TRACE=1 guarded_bench m30_r1c_retry $BINW $M30 1   # traced retry of the hung arm
for i in 1 2; do
    guarded_bench m02_r${i}_a_no72   $BIN72 $M02 0
    guarded_bench m02_r${i}_b_w72off $BINW  $M02 0
    guarded_bench m02_r${i}_c_w72on  $BINW  $M02 1
done
echo "=== ab-models2 done $(date +%H:%M:%S)" >> $RES
