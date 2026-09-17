#!/bin/zsh
# Re-certify patch 0073 on the FULL current series (0001-0070 + 0072 + 0073, i.e. with
# upstream's 0067-0070 tp work included). Needle controls first (poison detector), then
# one interleaved pp512 A/B pair and one tg128 A/B pair. Results append to the committed
# .bench/bf16-stage/results.txt so the PR record and the raw rows stay together.
set -u
BASE=/Users/chafey/src/toshllm/.bench
WT=$BASE/apply-check-0073
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
B=$WT/build-static/bin/llama-bench
C=$WT/build-static/bin/llama-completion
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
CTX=/Users/chafey/src/toshllm/.bench/prefill-bisect/ctx.txt
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

ppn() {
    local tag=$1 on=$2 p=$3 n=$4 r=$5
    local e=$LOGD/$tag.err v
    local args=($B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p $p -n $n -r $r)
    if (( on )); then v=$(TOSH_MGPU_STAGE_F16=1 "${args[@]}" 2>$e | awk -F'|' "/pp${p}/{gsub(/ /,\"\",\$11); print \$11}")
    else v=$(env -u TOSH_MGPU_STAGE_F16 "${args[@]}" 2>$e | awk -F'|' "/pp${p}/{gsub(/ /,\"\",\$11); print \$11}"); fi
    echo "$tag pp${p}=$v s5=$(grep -c 'status 5' $e)" >> $RES
}
tgn() {
    local tag=$1 on=$2 p=$3 n=$4 r=$5
    local e=$LOGD/$tag.err v
    local args=($B -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -p $p -n $n -r $r)
    if (( on )); then v=$(TOSH_MGPU_STAGE_F16=1 "${args[@]}" 2>$e | awk -F'|' "/tg${n}/{gsub(/ /,\"\",\$11); print \$11}")
    else v=$(env -u TOSH_MGPU_STAGE_F16 "${args[@]}" 2>$e | awk -F'|' "/tg${n}/{gsub(/ /,\"\",\$11); print \$11}"); fi
    echo "$tag tg${n}=$v s5=$(grep -c 'status 5' $e)" >> $RES
}

echo "=== full-series recert (0001-0070+0072+0073) start $(date +%H:%M:%S)" >> $RES
fails=0
for i in 1 2; do
    run fs_c${i}_off 0 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX || ((fails++))
done
if (( fails >= 2 )); then
    echo "ABORT: $fails/2 OFF controls failed - driver poisoned, wait for self-heal (fault-hunt.md)" >> $RES
    exit 1
fi
run fs_n1_on 1 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX
run fs_n2_on 1 $C -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none -t 0 -n 12 --no-display-prompt -f $CTX
ppn fs_p512_off 0 512 1 3
ppn fs_p512_on  1 512 1 3
tgn fs_t128_off 0 128 128 2
tgn fs_t128_on  1 128 128 2
echo "=== full-series recert done $(date +%H:%M:%S) — gates: needle=YES rc=0 s5=0 everywhere; pp on>off; tg parity" >> $RES
