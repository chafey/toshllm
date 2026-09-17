#!/bin/zsh
# Correctness gate for the STRIPPED 0073 build (no diagnostic scaffolding).
# Same structure as verify3.sh but short runs with -n 16 (with -n 8 the f16-rounded greedy
# start truncates before the answer) and one extra OFF needle control.
set -u
BASE=/Users/chafey/src/toshllm/.bench/bf16-stage
WT=$BASE/llama
RES=$BASE/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
B=$WT/build-static/bin/llama-bench
C=$WT/build-static/bin/llama-completion
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
CTX=/Users/chafey/src/toshllm/.bench/prefill-bisect/ctx.txt
SHORT=/Users/chafey/src/toshllm/.bench/prefill-bisect/short.txt
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

run() {
    local tag=$1 on=$2; shift 2
    local e=$LOGD/$tag.err o=$LOGD/$tag.out rc
    if (( on )); then
        TOSH_MGPU_STAGE_F16=1 "$@" < /dev/null >$o 2>$e
    else
        env -u TOSH_MGPU_STAGE_F16 "$@" < /dev/null >$o 2>$e
    fi
    rc=$?
    local s5=$(grep -c 'status 5' $e)
    echo "$tag rc=$rc s5=$s5 needle=$(grep -qo 7352 $o && echo YES || echo no) out=[$(tr -d '\n' < $o | cut -c1-40)]" >> $RES
    (( rc == 0 && s5 == 0 ))
}

echo "=== v4 gate (stripped build) start $(date +%H:%M:%S)" >> $RES
fails=0
for i in 1 2 3; do
    run c${i}_off 0 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX || ((fails++))
done
if (( fails >= 2 )); then
    echo "ABORT: $fails/3 OFF controls failed - driver is poisoned, wait for self-heal (fault-hunt.md)" >> $RES
    exit 1
fi

for i in 1 2 3 4 5; do
    run n${i}_on 1 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX
done
run short_on  1 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 16 --no-display-prompt -f $SHORT
run short_off 0 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 16 --no-display-prompt -f $SHORT
echo "=== v4 gate done $(date +%H:%M:%S) — gates: every n*_on needle=YES rc=0 s5=0; short_on out has 108" >> $RES
