#!/bin/zsh
# Post-reboot certification gate for patch 0073 (f16 transport for the prefill allreduce).
# Knob: TOSH_MGPU_STAGE_F16=1 (default off). Run ONLY after a fresh boot: the AMD driver can
# sit in a post-GPURestart state where even the vendor path faults (see fault-hunt.md), and the
# first rows are pure OFF controls that detect exactly that and abort.
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

# run <tag> <on=1/off=0> <cmd...>: rc, status-5 tally and the output head land in results.txt;
# stdout is also kept in v3-logs/<tag>.out for content checks (needle grep).
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

pp() {
    local tag=$1 on=$2
    local e=$LOGD/$tag.err v
    local args=($B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 512 -n 1 -r 3)
    if (( on )); then v=$(TOSH_MGPU_STAGE_F16=1 "${args[@]}" 2>$e | awk -F'|' '/pp512/{gsub(/ /,"",$11); print $11}')
    else v=$(env -u TOSH_MGPU_STAGE_F16 "${args[@]}" 2>$e | awk -F'|' '/pp512/{gsub(/ /,"",$11); print $11}'); fi
    echo "$tag pp512=$v s5=$(grep -c 'status 5' $e)" >> $RES
}
tg() {
    local tag=$1 on=$2
    local e=$LOGD/$tag.err v
    local args=($B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p 128 -n 128 -r 2)
    if (( on )); then v=$(TOSH_MGPU_STAGE_F16=1 "${args[@]}" 2>$e | awk -F'|' '/tg128/{gsub(/ /,"",$11); print $11}')
    else v=$(env -u TOSH_MGPU_STAGE_F16 "${args[@]}" 2>$e | awk -F'|' '/tg128/{gsub(/ /,"",$11); print $11}'); fi
    echo "$tag tg128=$v s5=$(grep -c 'status 5' $e)" >> $RES
}

echo "=== v3 gate start $(date +%H:%M:%S)" >> $RES
fails=0
for i in 1 2 3; do
    run c${i}_off 0 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX || ((fails++))
done
if (( fails >= 2 )); then
    echo "ABORT: $fails/3 OFF controls failed - driver is poisoned, REBOOT and rerun (fault-hunt.md)" >> $RES
    exit 1
fi

for i in 1 2 3 4 5; do
    run n${i}_on 1 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX
done
run short_on  1 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 8 --no-display-prompt -f $SHORT
run short_off 0 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 8 --no-display-prompt -f $SHORT

pp off1 0;  pp on1 1;  pp off2 0;  pp on2 1
tg off_tg 0; tg on_tg 1
echo "=== v3 gate done $(date +%H:%M:%S) — gates: every n*_on line must show needle=YES rc=0 s5=0" >> $RES
