#!/bin/zsh
# Round 2 for qwen38: r1 showed pp512 291 (no72) vs 640 (w72off) — 0072 must not touch pp,
# so reproduce both before saying anything about it. 3 arms again incl. ON.
set -u
BASE=/Users/chafey/src/toshllm/.bench
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
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
        (( w >= 1500 )) && { kill -9 $p 2>/dev/null; wait $p 2>/dev/null; echo "$tag TIMEOUT@1500s (hang)" >> $RES; return 124; }
        sleep 5; (( w += 5 ))
    done
    wait $p; local rc=$?
    local v=$(awk -F'|' '/pp512|tg128/{gsub(/ /,"",$11); printf "%s ", $11}' $out)
    echo "$tag rc=$rc vals=[$v] s5=$(grep -c 'status 5' $e)" >> $RES
    return $rc
}

M38=/Users/chafey/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

# wait for round 1 to finish first
while pgrep -x llama-bench >/dev/null; do sleep 15; done
echo "=== ab-qwen38 r2 start $(date +%H:%M:%S) (pp-anomaly repro)" >> $RES
guarded_bench m38_r2_b_w72off $BINW  $M38 0
guarded_bench m38_r2_a_no72   $BIN72 $M38 0
guarded_bench m38_r2_c_w72on  $BINW  $M38 1
echo "=== ab-qwen38 r2 done $(date +%H:%M:%S)" >> $RES
