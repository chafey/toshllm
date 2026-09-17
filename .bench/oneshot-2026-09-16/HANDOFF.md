# HANDOFF — TP4 decode effort (2026-09-16, end of Zed-thread-1)

This file exists so a fresh agent thread (e.g. Zed on another machine SSHed into this Mac
Pro) can continue with zero archaeology. Read this + `.bench/oneshot-2026-09-16/RESULTS.md`
and you have full state. Companion: `.bench/tp2-jumper-2026-09-16/RESULTS.md`.

## THREAD-2 UPDATE (2026-09-16, later same day) — read this first

### POST-MERGE (latest state — supersedes parts of everything below)

Fork synced to `f86e471`: series loses 0066-0068 (TP2 wg32), 0069 (qwen4exp MTP head),
0070 (dual-shape graph cache), 0071 (clear-new-buffers). Vendor tree = `465e49b9c` + 0001-0065
+ 0072, merge verified byte-identical, binaries rebuilt. **Everything got faster** — TP2 67.4/68.1
vs solo 66.2 (decode lead now solid), solo itself 66.2/+4.5 % pp, layer_ref 59.5, TP4 decode holds
42.8, and Qwen3.6-35B TP4 decode 15.2→35.6 (+134 %): 0070/0071 were a GLOBAL tax, not TP features.
**The dflash "+161 %" is corrected to PARITY** (34.8 vs 35.6 post-merge) — it had been measuring
around a self-inflicted slowdown; mechanism verified, payoff needs a draft that beats its overhead
(`TOSH_DFLASH_*` knobs). Full numbers: `.bench/tp-baseline-2026-09-16-postmerge/RESULTS.md`.
Prefill regression is NOT explained by 0070/0071 (tp_prod pp512 1500→1560, still ≪ 2930) —
bisect range is now 0048-0065.
**Exact 0072 benefit vs the pulled series** (ablation A/B, `.bench/0072-ablate/`): TP4 decode
+31.5 % (32.35 → 42.56), TP4 prefill +3 % (noise-ish), solo/TP2/group2 identical within sigma
(zero cost outside TP4). The 23.1 → 42.6 arc since 09-11 = +31.5 % from 0072, rest from the
pull's patch drops.
**Incident 2026-09-16 11:16: gpuRestart storm** (Restart Channel 25 = VMPT) rebooted the
machine mid-benchmark after ~2h20 of continuous TP/MoE load; one A/B batch was voided, the
post-reboot re-run was clean. Suspect the driver + long heavy peer-view load, NOT specific
code; plausibly also explains yesterday's slow-machine day and the 16-min solo-Qwen stall.

### (earlier thread-2 notes, some numbers now superseded by the section above)

Next items 1 and 3 are DONE; item 2 is BLOCKED on a draft model.

- Exported `patches/llama/0072-metal-zero-filled-collectives-and-one-shot-allreduce.patch`
  (the 4-file today's-work delta vs a `465e49b9c`+≤0071 reference tree). Verified: a fresh
  `465e49b9c` worktree + `git apply` 0001–0072 (69 patches) reproduces `vendor/llama.cpp`
  byte-for-byte, and a standalone cmake build of llama-bench/llama-completion from that
  clean tree succeeds (`verify-0072-{configure,build}.log` here).
- Re-baselined sweep + fresh solo reference row: **`.bench/tp-baseline-2026-09-16/RESULTS.md`**.
  Headline: on a clean machine TP2 decode now MEASURED FASTER than solo (tg128 66.2 vs 65.2,
  interleaved A/B, σ≈0.05) — the original goal is met at n=2, and TP2 pp512 is +41 % over
  solo. Yesterday's "TP2 −4 %" was a slow-machine-state artifact (solo pp512 is the tell:
  987 now, 60.2-solo day was slow machine-wide). TP4 decode ≈42 (−35 % vs solo); knob-
  independent across prod/events/none rows now. Prefill regression 09-11 pp512 2930 → ≈1500
  CONFIRMED, predates the zero-fill work, needs its own bisect.
- Spec decode: no GLM-compatible draft GGUF on disk (`ls /Users/chafey/models`: only a
  Qwen3.6 `.dflash` draft; GLM-4-9B needs same-vocab draft, e.g. a GLM MTP head or tiny GLM).
- `scripts/tp-baseline.sh` gained a `solo` row (`GGML_METAL_DEVICE_LIST=1 -sm none`) — decode
  rows are meaningless without the in-run single-die reference.

## Goal

Make **multi-GPU decode beat single-GPU decode** on this Mac Pro 2019: 4× W6800X Duo dies
in ONE bridged Metal peer group `5545434699043849590` (GGML indices 1-4; {1,2}=Slot-1
module, {3,4}=Slot-3; dev 0 = RX 6900 XT display GPU, always excluded via
`GGML_METAL_DEVICE_LIST=1,2,3,4`). Model: `/Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf`
(5.73 GiB, 40 layers, hidden 4096 → 16 KiB decode payload, **80 allreduce points/token**).

## Clean-machine baseline table (trust these; older notes' numbers were polluted by background GPU load)

| Config | tg128 |
|---|---|
| solo (dev 1) | 60.0–60.4 |
| TP2 (dies 1,2) staged | 57.6–57.9 (≈ solo −4%) |
| **TP4 staged (current default) + zero-fill** | **~38 (35.7–38.0 depending on machine noise)** |
| TP4 staged NOWAIT | 38.2 — staged waits overlap, ≈ free |
| TP4 one-shot (`TOSH_MGPU_ONESHOT=1`) | 35.8 |
| TP4 one-shot NOWAIT | 45.7 — one-shot's 4-party rendezvous stalls ~10 ms/token |
| TP4 allreduce NOOP (compute floor) | 93.5 |
| **TP2 NOOP (2-die compute floor)** | **90.0** ← if collectives were free, TP2 = 90 ≫ solo. Headroom is real. |
| TP4 pp512 | ~1370 (prefill regression to ~1400 predates today's work; separate issue) |

Current answer to the original question: multi-GPU decode does NOT yet beat solo. TP2 is
the near-miss; the entire TP2 gap (60.2 → 57.7 ≈ 78 µs/collective × 80/token) is host
encode + submission + queue-start latency, NOT fabric waits and NOT the bridge (jumper vs
bridge measured identical; see tp2-jumper RESULTS.md).

## Code state — ALL UNCOMMITTED on top of upstream `465e49b9c` in `vendor/llama.cpp`

Patches ≤0071 are applied-but-uncommitted in that tree (that's the normal workflow here;
`scripts/build-engines.sh` applies `patches/llama/*.patch` via `git apply`). Today's work
is on top and NOT yet exported as `patches/llama/0072-*.patch`. Files changed today:

- `ggml/src/ggml-metal/ggml-metal-context.h`
  - declares `ggml_metal_allreduce_oneshot(ctxs, tensors, n, seq, zero_mask)`
  - declares `ggml_metal_zero_tensor(ctx, t)`
  - declares `ggml_metal_xdev_prepare(ctx_self, ctx_peer, size)`
- `ggml/src/ggml-metal/ggml-metal-context.m`
  - link struct: `uint64_t x_slot[2]` (per-slot back-pressure)
  - `ggml_metal_oneshot_enabled()` — **default OFF**, `TOSH_MGPU_ONESHOT=1` opts in
  - `ggml_metal_zero_tensor()` — blit fillBuffer, used for empty-split-slice tensors
  - `ggml_metal_xdev_prepare()` — serial link create+reserve both directions (pre-pass for parallel encode)
  - `ggml_metal_allreduce_oneshot()` — single-round host-staged collective: one merged blit
    encoder (incl. zero fills), one merged compute encoder (n−1 dispatches, rb4-or-bin_one
    pipeline resolved in pre-pass), per-link `prev = x_slot[seq&1]` back-pressure, parity
    double-slot, stamps `seq_x`+`x_slot[sp]` after ALL cards encode (contexts share one
    address space — stamping early deadlocks). Gated n≥3 (n=2 is a strict loss).
  - staged `exchange_reduce` now stamps `x_slot[0]=x_slot[1]=seq` (it writes offset 0).
- `ggml/src/ggml-metal/ggml-metal.cpp`
  - **THE BIG FIX**: `comm_allreduce_tensor` no longer bails when a tensor lacks
    `GGML_TENSOR_FLAG_COMPUTE` (empty split slice = stale scratch). It records `zero_mask`
    and the metal collectives zero those tensors (one-shot folds fills into its blit
    encoder; staged uses `ggml_metal_zero_tensor` after the failed-one-shot point).
    Before: ~40 of 80 collectives/token silently fell into the generic meta butterfly
    (~10 extra cmd-buffer submissions each; 735→329 submissions/token after the fix).
    **This is what took TP4 decode 28→38.**
  - trace-gated (`TOSH_MGPU_TRACE=1`) bail-reason prints in comm_allreduce_tensor
  - opt-in parallel comm encode: `TOSH_MGPU_COMM_PARALLEL=1` → dispatch_apply over the
    round's exchange_reduce calls after a serial `xdev_prepare` pre-pass. **NEGATIVE
    RESULT by default** (TP2 41.4±9.1 vs 56.6 serial; TP4 36.5 vs 38.0) — driver
    contention on concurrent peer-group event encoding. Correctness verified; kept as knob.
- `ggml/src/ggml-backend-meta.cpp` — trace-gated once-per-subgraph print:
  `"allreduce subgraph %zu fell back to the generic butterfly"` (permanent diagnostic).
- `scripts/tp-baseline.sh` (repo root, itself untracked) — `run_spec` fixed to use
  `llama-completion` (llama-cli in this fork has NO conversation-disable flag and hangs
  printing `>` forever); `ONLY=` filter; tp2_jumper/tp2_bridge rows.
- Safety snapshot: `.bench/oneshot-2026-09-16/wip-llama-cpp.diff` (git diff of today's edits
  only, taken 2026-09-16; regenerate with
  `cd vendor/llama.cpp && git diff > ../.bench/oneshot-2026-09-16/wip-llama-cpp.diff`).
  LABEL WRONG — it is the FULL tree diff vs upstream (76 files: the whole ≤0071 series +
  today's work). 0072 was instead generated against a clean `465e49b9c`+≤0071 worktree.

## Build & bench (never run two GPU benchmarks concurrently)

```sh
cd vendor/llama.cpp
cmake --build build-static -j$(sysctl -n hw.ncpu) -t llama-bench llama-completion
export GGML_METAL_DEVICE_LIST=1,2,3,4 GGML_METAL_SHARED_BUFFERS_DISABLE=1 \
       GGML_METAL_CONCURRENCY_DISABLE=1 TOSH_FA_AMD=1 TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1
./build-static/bin/llama-bench -m /Users/chafey/models/GLM-4-9B-0414-Q4_K_M.gguf \
  -sm tensor -ngl 99 -fa 1 --load-mode none -p 0 -n 128 -r 3
```
Correctness sanity: `llama-completion -sm tensor ... -p "Q: What is 12*9? A: 12*9 is"` →
expect `108`; capital-of-France → `Paris`. **Kill video playback / llama-server before
benchmarks** — Firefox VTDecoderXPCService alone moves tg128 by ±2.

Env knobs: `TOSH_MGPU_ONESHOT=1` (one-shot), `TOSH_MGPU_COMM_PARALLEL=1`,
`TOSH_MGPU_NOWAIT=1` (bench-only, no waits, garbage output), `TOSH_MGPU_ALLREDUCE_NOOP=1`
(compute floor), `TOSH_MGPU_TRACE=1` (counters incl. fallback print + oneshot bail reasons),
`TOSH_MGPU_CTIME=1` (GPU timestamps inside staged exchange), `TOSH_MGPU_PEER_MIN_BATCH`
(batch gate, default 32), `TOSH_MGPU_SERIAL_ENCODE=1` (meta-layer serial encode).

## Do not re-derive (established the hard way)

1. Compute-kernel access to remote buffer views = ~600 µs/16 KiB on this stack. Peer-direct
   collectives via views are dead; host-block staging is the medium. Blits peer-reading are fine.
2. The two silent perf killers found today: meta fallback for empty-slice collectives
   (FIXED via zero_mask), and one-shot's 4-party in-buffer rendezvous (why it's default-off).
3. seq protocol across collectives: one global counter (`g_metal_comm_seq`), links may skip
   values but never go backwards; all 4 metal contexts live in ONE process — stamp per-link
   counters only after every card encoded. `done@seq-2` style fixed-offset back-pressure is
   WRONG (links aren't used every round); wait on the per-link/per-slot captured prev.
4. `-3` from llama_decode = GGML_STATUS_FAILED, usually a GPU-side event/stall watchdog fault;
   check event-value monotonicity per link.
5. Peer-fused/push paths (0061/0062/0065) are dead code in production (try_peer vs
   prefer_events mutually exclusive).
6. xlinks are directed, stored in ctx_dst->xlinks keyed by src_dev, 4 slots max per ctx.
7. llama-cli prints `>` forever in this fork (no --no-conversation); use llama-completion.
   And for spec: llama-COMPLETION cannot do spec at all (spec args are server/speculative/cli
   only, common_speculative never called) — spec rows must drive llama-server.
8. `llama-server` is not in the HANDOFF build one-liner and silently goes stale (05:44 binary
   vs 08:01 sources measured TP4 decode at 6.7 t/s). Rebuild `-t llama-server` before server rows.
9. Ngram spec + temp-0 + warming with the measured prompt = degenerate acceptance=1.0 replays
   (looked like 77 t/s on TP4). Warm with different filler; real fresh-prose accept ≈ 0.
10. Never run solo (dev-1) rows for the 20.9 GiB Qwen3.6-35B: CPU-offloaded decode at ~2.2 t/s
    AND the server stalls ~16 min in early init before `load_model`. Solo MoE rows are out of
    the spec matrix on purpose. (Recheck after fresh boot — the stall may be driver-state, not code.)
11. The AMD driver can gpuRestart-panic after hours of continuous TP2/TP4/MoE benchmarking
    (vmpt page-table restart). Reboot before serious measurement runs; interleave A/Bs in
    short batches with per-row result logging (see .bench/0072-ablate/ab.sh).

## Next steps (priority order) — thread-2 status inline

1. [DONE thread 2] **Export today's work as `patches/llama/0072-*.patch`** — exported as
   `0072-metal-zero-filled-collectives-and-one-shot-allreduce.patch`; fresh-tree apply of
   0001–0072 verified byte-identical to the vendor tree + standalone rebuild OK. Vendor
   tree is now REPRODUCIBLE from the patch series; ≤0071 remain applied-but-uncommitted by
   design.
2. [MECHANISM VERIFIED; payoff pending better draft] **Speculative decode** — rows run
   llama-server with `--spec-type` via `SPEC=1 ./scripts/tp-baseline.sh` (llama-completion
   can't do spec: args are server/spec/cli-only). Draft-less ngram-mod = no-op on fresh prose
   at every device count (pre- and post-merge). dflash head on Qwen3.6-35B TP4: acceptance
   ~0.6, mean verified len 2.8, coherent output — post-merge tg 34.8 vs 35.6 none = PARITY
   (the pre-merge +161 % was against the 0070/0071 tax; see POST-MERGE block). Follow-ups:
   `TOSH_DFLASH_ACC_MIN/PROBE/PROBE_CAP/EWMA` sweep; GLM-4-9B still lacks any same-vocab
   draft head (GLM-4.5-Air shares vocab 151552 but is 38 GiB target-class — cost-disqualified).
3. [DONE thread 2; regression CLOSED thread 3] Full `scripts/tp-baseline.sh` sweep — see
   `.bench/tp-baseline-2026-09-16/RESULTS.md`; prefill regression flagged separately (09-11
   tp_prod pp512 2930 → ≈1500; tp_events pp512 1247 → 886; decode unaffected; bisect needed).
   Thread-3 bisect: cliff is patch **0059** and 2930 was an INVALID measurement (pre-0059
   exchange waits were silently bypassed → corrupted long-context output; needle test proves it).
   1560 is the first honest TP4 prefill number. See "PREFILL BISECT RESOLVED" in
   `.bench/tp-baseline-2026-09-16-postmerge/RESULTS.md`.
4. [CLOSED thread 3b — do NOT implement flag-polling] Optional experiments: flag-polling (poll a host u64)
   instead of MTLSharedEvent waits to test whether event wake latency is worth attacking; `TOSH_MGPU_CTIME=1`
   timeline on the zero-fill'd staged path to see the new publish/wait/read split.
   ANSWER (measured): collective waits already cost ~0 at pp512 (see Thread 3b block below). The
   event-wake theory is dead; the remaining prefill cost is f32 fabric BYTES — see the bf16 design there.
5. [RESOLVED thread 2's question — answer was "nowhere to bisect": 0059, cost of correctness] Prefill bisect: TP4 pp512 fell 2930→≈1500 somewhere before 2026-09-16;
   solo/layer/TP2 prefill never changed much (987/917/1390) — look inside the TP4 tensor
   path (0063/0064 held-buffer/graph-cache interactions vs 09-11).

## Thread 3 (2026-09-16 pm): prefill bisect EXECUTED and RESOLVED

- Method: `git worktree` of vendor at `465e49b9c` + patch-prefix states, rebuilt per state
  (build-static, incremental ~1–4 min), same-machine pp512 interleaved. Scripts + per-line
  results kept in `.bench/prefill-bisect/` (run1–6.sh, results.txt, ctx.txt, short.txt);
  worktree itself removed after use (`git -C vendor/llama.cpp worktree remove --force
  .bench/prefill-bisect/llama && git -C vendor/llama.cpp worktree prune`).
- Bisect: S_A(≤0057)=2979–3142; S_A+0059=1554–1567; every state ≥{0059..0062}=1533–1580
  regardless of 0058/0060–0065. 0064/0065 (tg8 26.6→32.5 decode win) are prefill-innocent.
- Correctness evidence: 452-token needle prompt (code at start, question at end), greedy TP4.
  S_A: 4 runs → 4 DIFFERENT completions, needle not cleanly retrieved (nondeterministic race).
  H1(+0059): 2 runs → both exactly correct. Short prompt (12*9): both answer 108 — small
  decode payloads never trip the race, which is why 09-11 correctness checks passed.
- Conclusion: the "regression" = cost of correctness (real event waits on 80 prefill
  collectives, ~2 ms wake each). Do NOT roll back 0059. Only remaining prefill upside is
  wake-latency reduction → optional item 4 (flag-polling) is now the top open experiment,
  with the needle test (`run4.sh` pattern) as its correctness gate.
- New bench artifacts: `.bench/prefill-bisect/{run*.sh,results.txt,ctx.txt,short.txt,build.log}`.
  ctx.txt = 452-token needle prompt; short.txt = 12*9 probe. Reusable as regression gates.
- Nothing committed (repo convention). Vendor tree still 0001–0065+0072 applied-uncommitted.

## Thread 3b (2026-09-17 am): prefill upside root-caused — knob space EXHAUSTED, next move is bf16 transport

All artifacts in `.bench/wait-ablate/` (results.txt, paths.txt, trace_*/b2_* logs; scripts ab.sh,
paths2.sh, bytes.sh, bytes2.sh).

1. WAITS ARE FREE: pp512 TP4 with TOSH_MGPU_NOWAIT / BP_SATISFIED / WAIT_SATISFIED ≈ base
   (round-2 clean: 1548/1566/1563 vs base 1573; round-1 base 1392 was cold-start). Event-wake
   latency ≈ 0 on today's tree -> flag-polling prototype is pointless (item 4 closed).
2. PATH KNOBS INERT: TOSH_MGPU_PEER_FUSED=2/3 and ONESHOT=1 all land at base ±2% (paths.txt).
   One-shot halves exchange count on -n 1 yet pp512 unchanged; FUSED_EXCHANGE_DISABLE=1 run
   died mid-bench (cause ambiguous: right after a 35B driver crash; rerun post-reboot).
3. WHERE THE TIME IS (TOSH_MGPU_TRACE census, pp512 -n 0): the COPY BUTTERFLY — ~12 peer-direct
   f32 copies of the 8 MiB partial per allreduce point ≈ 7.9 GB per pp512 pass, 1896 copies /
   15.9 GB per (warm+2 reps), sustained 16–21 GB/s across the bridge; collective GPU 554 ms vs
   ~664 ms wall for 2 reps. The fused/staged exchange paths carry only the small payloads
   (24–32 exchanges ≤1 MB) — which is why every collective-side knob (incl. 0059's waits) is
   invisible at prefill… and why pre-0059 could still read 2978: its damage rode the same
   copy/exchange boundary; irrelevant now, the number was invalid either way.
4. ONLY LEVER LEFT = HALVE THE FABRIC BYTES. bf16 staging of the allreduce partials (0073
   candidate), comm-layer design (ggml-metal.cpp push/reduce + helpers in ggml-metal-context.m):
   (a) local cpy-pipeline dispatch f32->bf16 into a per-device bf16 scratch (exists already);
   (b) same peer-direct blit moves bf16 = half the bytes; (c) mixed-type add f32 += bf16 via a
   UP-cast variant of the add pipeline (bin kernels already upcast; add_inplace_* need a
   src-type param). Double-buffer scratches by seq parity, reuse x_slot back-pressure.
   Correctness gates: needle test + 12*9 (`.bench/prefill-bisect/ctx.txt`, `short.txt`), then
   pp/tg A/B, then at least a perplexity spot check. Develop in a FRESH git worktree, never in
   the vendor tree; export as patches/llama/0073-*.patch when proven.
5. dflash knob sweep (.bench/dflash-sweep/sw.sh, ready to run, 9 rows): BLOCKED on machine
   stability — Qwen3.6-35B TP4 load crashed the driver ~1 min into load (cmd buffer status 5,
   SubmissionsIgnored, 4× gpuRestart/VMPT in 6 min). Two hard resets overnight (no panic
   reports = VMPT-storm signature). RUN sw.sh FIRST AFTER A REBOOT, before other GPU work.
6. Hygiene: long terminal sleeps kill the SSH session (idle timeout — likely also the earlier
   mystery drops); poll fast instead. A hard reset ate recently-written, unsynced files
   (paths2.sh vanished) — re-verify scripts on disk after any panic.

## Thread 3c (2026-09-17 pm-session): dflash sweep blocked by REPRODUCIBLE 35B-TP4 driver crash; bf16 prototype scaffolded

1. 35B CRASH IS NOT DRIVER WEAR — it reproduces on a FRESH BOOT, twice, at the same spot:
   `llama-server -m Qwen3.6-35B... -sm tensor -ngl 99` dies ~66 s into load, right after
   `ggml_backend_metal_comm_init: reducing across 4 devices in 2 steps` -> storm of
   `ggml_metal_synchronize: command buffer failed status 5` + `SubmissionsIgnored` (logs:
   `.bench/dflash-sweep/logs/none_a.log`). The same model+config ran FINE twice on 09-16 PM and
   9B TP4 ran 10+ loads clean on both today's boots, so this is 35B+TP4-specific and NEW since
   09-16 — suspect hardware/driver state degradation (two hard resets overnight with NO load,
   no panic reports = VMPT signature). USER ACTION: note for driver reset / hardware check;
   do not keep re-firing it (each attempt risks a gpuRestart storm).
2. dflash knob sweep (sw.sh, 9 rows, ready) stays BLOCKED pending 1. Recorded in
   `.bench/dflash-sweep/results.txt`.
3. fusx_off loose end CLOSED: TOSH_MGPU_FUSED_EXCHANGE_DISABLE=1 at pp512 = 1521.6 ± 51.6
   (base ~1540–1570), census 0 exchanges / 1920 copies / same 15.9 GB / collective GPU 528 ms
   (~= base 554). Prefill is 100% copy butterfly regardless of knob. `.bench/wait-ablate/paths.txt`.
4. PATCH 0073 PREP — worktree at `.bench/bf16-stage/llama` (vendor repo, base `465e49b9c`,
   series 0001–0065+0072 applied (69 files, +18258/-954), built OK, SMOKE-TESTED 2026-09-17
   07:41: solo pp512=1010 tg8=65.5 (on-baseline). bins at
   `.bench/bf16-stage/llama/build-static/bin/{llama-bench,llama-completion}`; setup script
   `.bench/bf16-stage/setup.sh` (re-runnable, resets + reapplies + rebuilds). NOTE: `git worktree
   add` with a RELATIVE path resolves against the repo dir — always pass the ABSOLUTE path.
   Implementation anchors (all verified against current tree):
   - comm loop: `ggml-metal.cpp` ~1199–1330: `push(j_src,j_dst,step,hold)` lambda does
     `ggml_backend_metal_cpy_tensor_async_ex(tensors[j_src], &tmp[j_dst])`; `reduce(j_dst)`
     does `ggml_metal_add_inplace_async(ctx, tensors[j_dst], &tmp[j_dst])`.
   - add: `ggml-metal-context.m:1971` `add_inplace_supported` hardcodes BOTH types F32;
     `:1991` `add_inplace_into` grabs `get_pipeline_bin_one(lib, GGML_OP_ADD)` (the f32-only
     variant); `:2105` `add_inplace_buffer` = raw-buffer add used by the fused path.
   - cpy: `get_pipeline_cpy(lib, F32, F32, true)` usage ~`:3160` — same factory with
     (F32, BF16) should yield the downconvert kernel; verify c4 flag semantics for mixed types.
   - Plan (new knob TOSH_MGPU_STAGE_BF16, default OFF = tree behavior byte-identical):
     (a) per-device bf16 scratch registry (double-buffered by seq parity; precedent:
     `xdev_shadow_reserve`); (b) in comm fallback: local cpy-pipeline dispatch F32->BF16 into
     scratch_src, then peer-transfer HALF bytes (reuse cpy_xdev machinery), then new
     `add_inplace` variant with src-type BF16 (needs the (dst F32, src BF16) bin pipeline +
     kargs with bf16 strides, nb10=2); (c) on pipeline miss, fall back to the f32 path.
   - Accuracy: partials round to bf16 once, sum in f32 — expected tiny loss, but GATES ARE:
     needle + 12*9 (`.bench/prefill-bisect/`), pp512/tg128 A/B interleaved, then llama-perplexity
     spot check before ANY default flip. Keep knob off in exported 0073.
   - When done: `git -C .bench/bf16-stage/llama diff > patches/llama/0073-...patch` style export
     from the applied worktree state (mirror the 0072 workflow), vendor tree untouched throughout.

## Thread 3d (2026-09-17): patch 0073 IMPLEMENTED in worktree — pp512 +9 %, decode cost 0, needle gate blocked on driver (reboot needed)

1. 0073 = TOSH_MGPU_STAGE_BF16 (default OFF): bf16 transport for the prefill copy-butterfly.
   Knob-off is byte-identical behavior (proved: worktree r1_off pp512 1555.8 = vendor band).
   Lives ONLY in worktree .bench/bf16-stage/llama (vendor tree untouched); snapshot diff:
   .bench/bf16-stage/snapshot-0001-0065+0072+0073wip.patch (combined 1-65+72+73 — the worktree
   itself is the source of truth; export a clean 0073 to patches/llama/ only after gates pass).
2. Design (all 6 edits compile first try): binbcast.metal += kernel_bin_fuse f32_bf16_f32 +
   f32_bf16e_f32 instantiations; device.cpp get_pipeline_bin_one_src1 (tname_pair -> bf16e on
   no-bfloat cards); context.m ggml_metal_cvt_f32_bf16 (existing kernel_cpy_contig_f32_bf16e,
   elementwise, RNE), add_inplace_bf16_src, bf16_stage_supported; ggml-metal.cpp comm ctx gets
   tmp_bf16 slots (same per-(dev,step) slotting as f32), push = cvt->half-size peer copy,
   reduce = f32 += bf16 upcast add; gated on !prefer_events (decode stays f32 by design).
3. Measured (verify.sh, results.txt): pp512 off 1555.8 / on 1706.0 + 1686.2 = **+8.5..9.7 %**;
   tg128 on 42.45 = vendor parity (decode untouched as designed); short 12*9 = 108.
   THEORY CHECK: expected more than +9 % if purely wire-bound (8 GB/pass halves) -> either
   convert/add dispatch overhead eats most of it or the copy cost is not wire-byte-dominated.
   Follow-up if user wants more: trace census split (convert vs transfer vs add time per point),
   vec4 kernels (bf16e4 exists in common.h), merge cvt into the held cmd buffer.
4. BLOCKED GATE: needle (and r2_off pp + off_tg) died mid-verify with cmd-buffer status 5 +
   SubmissionsIgnored — on the KNOB-OFF path. Driver degraded again (the 35B crash storm at
   07:20 seeded it; VMPT again). AFTER REBOOT: `nohup zsh .bench/bf16-stage/verify.sh &` — it
   appends; needle-on must retrieve 7352 coherently (numbers will differ slightly from f32:
   bf16 rounding is a real numeric change; gate = coherent + correct code, NOT text-equality).
   Then decide: export clean 0073 + consider a llama-perplexity spot check as final gate.
5. 35B TP4 status unchanged: reproducible crash at load (2/2 today). Do not run sw.sh on this
   machine until it can survive the 35B load; verify.sh (9B) is the canary — if 9B runs start
   status-5-ing, reboot rather than debug.

## Thread 4 (2026-09-17): patch 0073 correctness bug FOUND+FIXED; transport bf16->f16; driver CONTAMINATED (reboot needed before any number)

1. **Correctness bug in 0073 (fixed)**: `view_tmp_bf16` shared one slot per (dev, step) for
   BOTH directions. A bf16 push stages the outgoing conversion in local VRAM (the f32 path
   never stages outgoing bytes), so push(2,0)'s cvt overwrote the partial push(0,2) had just
   landed in slot(2,0) -> wrong sums (that's the on_needle garbage prose in verify3), plus a
   cross-device WAR (partner remote-reading an outbox while local cvt rewrote it). Fix: the
   buffer registry AND the view vector are now 2*n_backends slots: [0,n) inboxes,
   [n,2n) outboxes. Post-fix EVERY completed ON run returned the correct needle (7352):
   peer path 2/2, events path 2/2 (TOSH_MGPU_PEER=0), f16 mode2 2/2. No wrong answers since.
2. **Transport switched bf16 -> f16** (native half; scalar bf16e kernels had never executed on
   these no-bfloat cards and were the prime suspect for the GPU-restart poisoning; fork even
   has a TOSH_BF16_MANUAL_DISABLE kill-switch for the similar manual-bf16 mul_mm path).
   kernel_cpy_contig_f32_f16 (existing) + new binbcast.metal instantiation
   kernel_bin_fuse_f32_f16_f32 <float, half, float>. Knob renamed TOSH_MGPU_STAGE_F16.
   f16 = 10-bit mantissa, better than bf16 here; overflow needs |partial|>65504 - gates
   (needle + 12*9 + ppl) would catch it loudly.
3. **Driver state**: after several with-collision bf16 runs (cross-bridge racing on the same
   pages), kernel log shows repeated `IOAccelCommandQueue: Deny Submissions app[...] with 2
   GPURestarts in N submissions`. From then on EVERY path faults, including pure vendor:
   A/B (off vs mode3, /tmp/ab_results.txt) mode0 crashed 2/4 while mode3 passed 2/4.
   New processes fail ~0.14 s in with status 5 + SubmissionsIgnored. => numbers are
   meaningless until a REBOOT (HANDOFF rule: 9B status-5s, reboot don't debug).
4. Debug scaffolding still in the worktree (strip before final export): knob values 2/3
   (cvt dry-run / local loopback), TOSH_MGPU_STAGE_DEBUG=1 command-buffer labels+fault
   printing incl. on the graph-pool buffers. fault-hunt.md documents the story + decisions.
5. Files: worktree .bench/bf16-stage/llama (source of truth), draft patch
   .bench/bf16-stage/0073-draft-f16-transport.patch (git-apply-able on top of
   ref-0072 = 465e49b9c+0001-0065+0072 worktree, verified with git apply --check),
   hardened gate script .bench/bf16-stage/verify3.sh (OFF-control poison detector first,
   stderr per run, needle YES/no per run; results append to results.txt).
6. **NEXT (in order)**: ~~user reboots -> verify3~~ DONE — see Thread 5 below (certified on
   the fresh boot, patch exported, knob stays off).
   5x needle YES on, short 108, pp512 on > off (old bf16 gave +8.5..9.7 % on the WRONG
   sums, so the real perf number must be re-measured!), tg128 parity. If clean: strip
   diagnostics, rename tmp_f16 comment remnants if any, regenerate 0073-draft -> final
   patches/llama/0073-metal-f16-allreduce-transport.patch (export procedure + apply-check as
   today), llama-perplexity spot check, then flip nothing by default (knob stays off).
   If mode1 faults while OFF controls are clean on a fresh driver -> try vec4 transport
   (kernel_bin_fuse_f32_f16_f32_4 <float4,half4,float4> + ne0/4 kargs) before giving up.

## Thread 5 (2026-09-17): 0073 STRIPPED, RE-GATED, CERTIFIED, EXPORTED — done, knob stays OFF

1. Strip fixed + rebuilt (4 broken leftovers in ggml-metal-context.m: cvt's cmd_buf_end line
   restored, 3 stray tosh_stage_label call sites removed). All diagnostics gone; bf16->f16
   renames complete; knob = TOSH_MGPU_STAGE_F16 (default off).
2. Correctness re-proven on the STRIPPED build (verify4.sh): OFF controls 3/3 = 7352,
   needle ON 5/5 = 7352, short gate answers 108 BOTH modes (**use -n 16**; at -n 8 the
   f16-rounded greedy start truncates before the answer — sampling artifact, not a wrong
   sum). All rc=0, s5=0. Driver healthy throughout (verify4 aborts itself if >=2 OFF
   controls fail, so clean gates prove healthy driver).
3. Perf (perf3.sh, stripped, interleaved): pp512 off 1495/1500 vs on 1643/1670 = +10..11 %;
   pp2048 off 1408/1399 vs on 1539/1527 = +9 % (exercises tmp_f16 grow/realloc — fine);
   tg128 42.93 vs 43.08 = parity. Same as the pre-strip diagnostic build: strip changed
   nothing.
4. llama-perplexity smoke (ppl1.sh/ppl2.sh, -c 512 --chunks 4, ppl-corpus.txt = ctx.txt x7;
   NOTE the tool needs >= 1024 tokens, LOG_* goes to stderr): PPL off 1.4937+/-0.0712 vs on
   1.4940+/-0.0712 = +0.02 % — nothing. rc=0 s5=0. Its internal "s/pass" line is bogus
   (CPU-bound wall, off 63/72 s vs on 73/73 s — noise); don't judge perf there.
5. EXPORTED: patches/llama/0073-metal-f16-allreduce-transport.patch (6 files +301/-9),
   `git apply --check` verified on ref-0072. Final gate story + numbers: fault-hunt.md
   CLOSE-OUT section. Raw rows: .bench/bf16-stage/results.txt (blocks v4 gate / perf3 /
   ppl1 / ppl2, all 09:37-09:52).
6. NOTHING committed, vendor tree untouched, default behavior unchanged (knob off).
   Remaining OPTIONAL upsides (all unblocked, none started): vec4 f16 transport (~1 % more
   fabric savings + fewer dispatches); default-on flip after wider-model validation;
   35B-TP4 dflash sweep (thread 3c) still blocked on its own driver crash.

## Suggested first message to the new thread

"Read .bench/oneshot-2026-09-16/HANDOFF.md (Thread 5 = latest). Patch 0073 (f16 allreduce
transport, TOSH_MGPU_STAGE_F16, default OFF) is CERTIFIED and exported to
patches/llama/0073-metal-f16-allreduce-transport.patch: pp512 +10 %, pp2048 +9 %, decode
parity, needle 5/5, PPL drift +0.02 %. Pick the next upside from Thread 5 item 6 (vec4
transport, default-on validation sweep, or the 35B-TP4 dflash blocker). Machine: Mac Pro,
4 W6800X dies, GGML_METAL_DEVICE_LIST=1,2,3,4; run long suites under nohup and poll;
OFF controls first, abort if the driver looks poisoned (everything ~50x slow, no errors)."
