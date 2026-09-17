# One-shot all-reduce + zero-fill fix for TP4 decode — 2026-09-16

## Parallel comm encode — negative result (later)

Tried fanning the staged round's per-card exchange encodes out on GCD threads
(`TOSH_MGPU_COMM_PARALLEL=1`, with a serial `ggml_metal_xdev_prepare` pre-pass so the lazy
link creation cannot race). Measured: TP2 41.4 ± 9.1 vs 56.6 serial; TP4 36.5 vs 38.0.
Worse and unstable at both device counts — the encodes are too short to pay for thread
handoff and concurrent cross-device event encoding contends in the driver. Kept as an
opt-in; default stays serial. Note the correctness of the parallel path was verified.

Machine-state caveat: later "clean" runs drifted 35.7-38.0 because Firefox was playing
video (VTDecoderXPCService active). TP4 staged decode numbers within ±1.5 t/s of 38 should
be treated as the same value; kill all video/browser GPU work for tighter comparisons.

## FINAL clean-machine numbers (background llama-server killed, r=2-3, interleaved A/B)

All earlier figures today were polluted by background GPU load; treat those as void.
GLM-4-9B-Q4_K_M, `-ngl 99 -fa 1 --load-mode none`, tg rows with `-p 0 -n 128`:

| Config | tg128 | notes |
|---|---|---|
| solo (dev 1) | 60.0–60.4 | old baseline said 65.2 — that was measured under unknown background load too |
| TP2 (dies 1,2) staged | 57.6–57.9 | ≈ solo −4%; was 57.75 before AND after the zero-fill change |
| **TP4 staged (default now) + zero-fill** | **37.9–38.0** | was 28.3 before the zero-fill fix |
| TP4 staged NOWAIT | 38.2 | staged waits overlap ≈ free |
| TP4 one-shot (TOSH_MGPU_ONESHOT=1, double-slot) | 35.8 | 4-party rendezvous stalls ~10 ms/token |
| TP4 one-shot NOWAIT | 45.7 | the stall, isolated |
| TP4 allreduce NOOP floor | 93.5 | compute floor |
| TP4 pp512 | 1367 | unchanged by any of this |

Decision: **one-shot default OFF** (opt-in `TOSH_MGPU_ONESHOT=1`). At n=4 staged beats it
38.0 vs 35.8: the one-shot saves a round but makes every queue wait on all 3 partners in one
held command buffer (pure stall = 45.7−35.8 ≈ 10 ms/token), while the butterfly's pairwise
rounds overlap and its waits are invisible to NOWAIT. Double-slot parity (per-link x_slot,
now correctly tracked; staged stamps both slots) helped one-shot 35.0→35.8 but cannot fix the
rendezvous shape. One-shot code kept for the day the wait cost falls (e.g. flag polling).

Bottom line: **TP4 decode no longer beats solo decode, 38.0 vs 60.2, but TP2 nearly ties**
(57.7). The remaining TP4 cost is ~196 us/collective of host encode + submission + queue
latency (waits are already ~free in staged), not the fabric.

## UPDATE (earlier, under background load — numbers superseded): the real bug was the silent meta fallback — tg128 28 → ~34–42

The +406 submissions/token difference vs NOOP was not graph reuse: it was the **meta
fallback butterfly** running for ~40 of the 80 collectives per token. Half the collective
tensors (one of attn-out / FFN-out per layer) carry GGML_TENSOR_FLAG_COMPUTE on only two of
the four devices — the other two hold full-size tensors of stale scratch (the empty-split-
slice nodes). `comm_allreduce_tensor` bailed on that and handed the collective to the generic
butterfly (~10 extra cmd-buffer submissions each, zero-fills included).

Fix: `zero_mask` + `ggml_metal_zero_tensor()` — the metal collectives now zero the non-COMPUTE
tensors with a blit fill (folded into the one-shot's merged blit encoder) instead of bailing.
Fallback count at TP4 decode is now **0** (`TOSH_MGPU_TRACE=1` prints each silent fallback
once per subgraph point; grep "fell back to the generic butterfly").

The zero_mask / ggml_metal_zero_tensor fix survived the clean re-measurement: submissions
dropped 735→329/token (= NOOP), and TP4 decode went 28.3 → 38.0 (staged, the new default).
All the "before" numbers in this section were taken with a background llama-server holding
the GPU and are unreliable in absolute terms; the fallback bug and its fix are real.

Correctness: "Paris" / "108" coherent, exit 0 across pp512+tg128, fallbacks 0.

---

Model: GLM-4-9B-0414-Q4_K_M (5.73 GiB, 40 layers, hidden 4096 → 16 KiB decode payload,
40 collectives/token). All runs: llama-bench `-ngl 99 -fa 1 --load-mode none`,
`GGML_METAL_DEVICE_LIST` per row, TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1.

## Headline numbers (tg128 unless noted)

| Config | tg128 |
|---|---|
| solo (1 die) | 65.2 |
| TP4 staged (butterfly, ONESHOT=0) | 28.3 |
| TP4 one-shot (default, n=4) | 28.1–28.6 |
| TP4 one-shot NOWAIT | 29.8 |
| TP4 staged + CTIME timeline | 23.3 (ctime adds samples) |
| TP2 staged (dies 1,2) | 61.3–61.6 |
| TP2 one-shot (would-be) | 49.2 → **gated off, n≥3 only** |
| TP4 allreduce NOOP (compute floor) | 101.0 |

## The one-shot landed, works, and does not win

`ggml_metal_allreduce_oneshot()` (host-staged single round, one blit encoder + one
compute encoder per card, per-link back-pressure identical to staged) is correct
("Paris", exit 0 across pp512+tg128) and neutral on perf at 4 cards. Two latent bugs
were killed on the way (they had prevented the path from ever executing on GPU):

1. **Pre-pass reserve ordering**: the cap check read `in_l[i][j]->cap` (the directed
   link j→i) before iteration (j,i) reserved it. Fixed by reserving both directions
   in each iteration, like staged does.
2. **`done@seq-2` back-pressure was wrong**: seq is a group-wide counter and the
   staged butterfly only ever uses butterfly-partner links (offsets 2,1), so links
   like 0→3 never had a `done` signal at seq−2 → GPU stall → watchdog fault (`res=-3`).
   Fixed with staged's per-link protocol: capture `prev = link->seq_x` for every link
   *before* any card encodes, wait `ev_x_done @ prev`, re-stamp all `seq_x` *after*
   all cards encode (all 4 contexts share one address space; early stamping would
   deadlock). Parity double-slot dropped; single slot at offset 0 is safe under this
   protocol and matches staged's writes.

## Why one-shot cannot win (yet): it is not the link, it is the host

CTIME timeline on the staged exchange (n=20800):
**publish 4.3 µs · WAIT-WINDOW 183.7 µs · read 5.2 µs.**
The bytes cost ~10 µs; the collective command buffer then sits on a GPU event for
~184 µs while the *partner's* buffer waits to start (commit→GPU-start mean ≈ 116 µs).

But removing all waits (NOWAIT) changes decode only 28.3 → 29.8. The stall is
**shadowed by host serialisation**, not exposed by it:

- encode CPU ≈ 0.12–0.15 ms per TP4 collective (all 4 cards encoded serially on one
  host thread; ≈ 5 ms/token);
- 735 command-buffer submissions/token with real collectives vs 329 under NOOP;
- every collective buffer then pays ~116 µs queue-start latency because the host
  cannot stay ahead of the GPU (empty-queue wake-up).

TP2 proves this: 61.3 ≈ solo 65.2 → ≈23 µs/collective on the critical path, because
one host thread keeps both queues fed and there is only one round. TP4 needs
≤137 µs/collective to match solo decode; the host encode + submission + start-latency
chain costs ≈600 µs. Halving rounds (one-shot) cannot fix a per-submission tax that
exists per round- *set*.

## Decision (superseded in part — see UPDATE above)

- One-shot kept, default-on but **gated to n≥3** (n=2 is a strict loss: extra event
  ops on a single-round path).
- The productive next steps for TP4 decode are **not** on the sync path:
  1. cut submissions: keep the whole decode step in merged/held buffers (why do real
     collectives cost +406 submissions/token vs NOOP? graph-reuse interaction),
  2. parallelise the per-card encode of one collective across host threads,
  3. speculative decode to amortise 40 collectives over several accepted tokens,
  4. or serve decode with layer-split/TP2 and keep TP4 for prefill.

## Fixed this session in scripts/tp-baseline.sh

`run_spec` used `llama-cli -no-cnv` — this fork's llama-cli has no conversation-disable
flag and hangs printing `>` forever. Switched to `llama-completion` (build with
`cmake --build build-static -t llama-completion`).
