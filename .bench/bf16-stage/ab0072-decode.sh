#!/bin/zsh
# Real A/B: does patch 0072 still change performance on the current series?
# no72 = .bench/ab0072-main (0001-0070), w72 = .bench/apply-check-0073 (0001-0070+0072+0073).
# 0073 is default-off, so with no env knobs the binaries differ by exactly 0072.
set -u
BASE=/Users/chafey/src/toshllm/.bench
RES=$BASE/bf16-stage/results.txt
LOGD=$BASE/v3-logs
mkdir -p $LOGD
BIN72=$BASE/ab0072-main/build-static/bin
BINW=$BASE/apply-check-0073/build-static/bin
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
CTX=$BASE/prefill-bisect/ctx.txt
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1

bench() {
    local tag=$1 bin=$2 p=$3 n=$4 r=$5
    local e=$LOGD/$tag.err
    local v=$("$bin/llama-bench" -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none \
        -p $p -n $n -r $r 2>$e | awk -F'|' "/pp${p}|tg${n}/{gsub(/ /,\"\",\$11); printf \"%s \", \$11}")
    echo "$tag vals=[$v] s5=$(grep -c 'status 5' $e)" >> $RES
}
fb() {
    local tag=$1 bin=$2
    local e=$LOGD/$tag.err
    TOSH_MGPU_TRACE=1 "$bin/llama-completion" -m $MODEL -sm tensor -ngl 99 -fa 1 --load-mode none \
        -t 0 -n 12 --no-display-prompt -f $CTX < /dev/null >$LOGD/$tag.out 2>$e
    echo "$tag rc=$? fb=$(grep -c 'fell back to the generic butterfly' $e) s5=$(grep -c 'status 5' $e)" >> $RES
}

echo "=== ab-0072 real A/B start $(date +%H:%M:%S)" >> $RES
for i in 1 2 3; do
    bench ab_r${i}_tg_no72 $BIN72 0 128 2
    bench ab_r${i}_tg_w72  $BINW  0 128 2
    bench ab_r${i}_pp_no72 $BIN72 512 1 3
    bench ab_r${i}_pp_w72  $BINW  512 1 3
done
fb ab_fb_no72 $BIN72
fb ab_fb_w72  $BINW
echo "=== ab-0072 real A/B done $(date +%H:%M:%S) — first tg value is tg128; pp pair is pp512+tg1" >> $RES
