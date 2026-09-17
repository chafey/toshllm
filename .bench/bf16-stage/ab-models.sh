#!/bin/zsh
# Cross-model confirmation of 0072+0073 on the current series. Three arms, one llama-bench
# call each (loads model once, does pp512 x2 and tg128 x2):
#   a_no72   = .bench/ab0072-main       (0001-0070)                      -> 0072 delta = b-a
#   b_w72off = .bench/apply-check-0073  (0001-0070+0072+0073, knob off)  ->
#   c_w72on  = same binary, TOSH_MGPU_STAGE_F16=1                        -> 0073 delta = c-b
# 2 interleaved rounds per model. NOTE: perf-only (needle gates are GLM-specific).
set -u
BASE=/Users/chafey/src/toshllm/.bench
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
BIN72=$BASE/ab0072-main/build-static/bin
BINW=$BASE/apply-check-0073/build-static/bin
CTX=$BASE/prefill-bisect/ctx.txt
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

arm() {
    local tag=$1 bin=$2 model=$3 on=$4
    local e=$LOGD/$tag.err v
    local args=($bin/llama-bench -m $model -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 128 -r 2)
    if (( on )); then v=$(TOSH_MGPU_STAGE_F16=1 "${args[@]}" 2>$e | awk -F'|' '/pp512|tg128/{gsub(/ /,"",$11); printf "%s ", $11}')
    else v=$(env -u TOSH_MGPU_STAGE_F16 "${args[@]}" 2>$e | awk -F'|' '/pp512|tg128/{gsub(/ /,"",$11); printf "%s ", $11}'); fi
    local rc=$?
    echo "$tag rc=$rc vals=[$v] s5=$(grep -c 'status 5' $e)" >> $RES
    (( rc == 0 ))
}

runmodel() {
    local mtag=$1 model=$2
    echo "--- $mtag $(basename $model) start $(date +%H:%M:%S)" >> $RES
    local ok=1
    for i in 1 2; do
        arm ${mtag}_r${i}_a_no72   $BIN72 $model 0 || ok=0
        arm ${mtag}_r${i}_b_w72off $BINW  $model 0 || ok=0
        arm ${mtag}_r${i}_c_w72on  $BINW  $model 1 || ok=0
    done
    # cheap correctness spot check on the f16 path: needle only exists for GLM; for other
    # models just require the arms to have completed (rc=0 s5=0)
    echo "--- $mtag done $(date +%H:%M:%S) ok=$ok (vals order: pp512 tg128)" >> $RES
}

echo "=== ab-models start $(date +%H:%M:%S) (0072=b-a, 0073=c-b, pp512 then tg128)" >> $RES
runmodel m30 /Users/chafey/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
runmodel m02 /Users/chafey/models/Qwen3-VL-2B-Instruct-Q8_0.gguf
echo "=== ab-models done $(date +%H:%M:%S)" >> $RES
