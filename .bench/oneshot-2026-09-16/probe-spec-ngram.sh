#!/bin/zsh
# Draft-less speculative-decode probe on TP4 (GLM-4-9B), via llama-server timings.
# llama-completion in this tree never wires up common_speculative (set_examples =
# server/speculative/cli only), so the server is the measurement vehicle.
#
# CAVEAT found in-run: warming the ngram cache with the SAME prompt at temp 0 makes each
# pass replay the previous generation ("draft acceptance = 1.00000, mean len = 65").
# The 77 t/s here is a repetition-bound upper bound, not a baseline; the honest rows are
# in scripts/tp-baseline.sh (SPEC=1), which warms with different filler text.
# Also NOTE: llama-server was a stale 05:44 build for the first run (6.7 t/s nonsense);
# rebuild it after any ggml edit.
set -u
ROOT=/Users/chafey/src/toshllm
BIN=$ROOT/vendor/llama.cpp/build-static/bin/llama-server
MODEL=/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf
OUT=$ROOT/.bench/oneshot-2026-09-16
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1
PROMPT="Explain in one paragraph why memory bandwidth dominates single-stream decode."

# probe <name> <port> [spec flags...]
probe() {
    local name=$1 port=$2; shift 2
    echo "=== $name ($*)"
    "$BIN" -m "$MODEL" -sm tensor -ngl 99 -fa 1 --load-mode none \
        -c 2048 -np 1 --port "$port" "$@" > "$OUT/srv-$name.log" 2>&1 &
    local pid=$! i=0
    while (( i++ < 90 )); do curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 2; done
    # warm-up pass (builds the ngram cache), then two timed passes, same prompt
    curl -s "http://127.0.0.1:$port/completion" \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":128,\"temperature\":0,\"ignore_eos\":true}" >/dev/null
    for r in 1 2; do
        curl -s "http://127.0.0.1:$port/completion" \
            -d "{\"prompt\":\"$PROMPT\",\"n_predict\":128,\"temperature\":0,\"ignore_eos\":true}" \
        | python3 -c '
import json, sys
d = json.load(sys.stdin)
t = d["timings"]
n = t.get("predicted_n")
ms = t.get("predicted_ms")
if n and ms:
    print(f"{n} tok / {ms:.0f} ms = {1000.0*n/ms:.2f} tok/s")
else:
    print("timings:", json.dumps(t))'
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    grep -iE 'spec.*accept|accept.*draft' "$OUT/srv-$name.log" | tail -4
    sleep 3
}

probe none   8181
probe ngram4 8182 --spec-type ngram-mod --spec-draft-n-max 4
probe ngram8 8183 --spec-type ngram-mod --spec-draft-n-max 8
echo "=== done"
