# 0073 fault hunt (2026-09-17, thread 4)

## Bugs found & fixed (in worktree .bench/bf16-stage/llama only)
1. **Inbox/outbox slot collision** (the real correctness bug): `view_tmp_bf16` used one slot
   per (dev, step), but push() stages BOTH sides in bf16 mode - cvt(j_src) writes the slot the
   partner's blit lands in for j_dst. Within a TP4 round: push(0,2) lands dev0's partial in
   slot(2,0); push(2,0)'s cvt then overwrites it -> reduce(2) adds cvt(dev2) instead of
   dev0's partial. Also cross-device WAR (partner remote-reading an outbox while local cvt
   rewrites it). Fix: doubled slot space, [0,n) inboxes / [n,2n) outboxes, in the buffer
   registry AND the view vector (the tmp_bf16[i] view was also aliased by src-role pushes).
   Post-fix every completed run returns the correct needle (7352) with knob ON.
2. Diagnostics left in-tree (env TOSH_MGPU_STAGE_BF16=2/3 = cvt dry-run / local loopback;
   TOSH_MGPU_STAGE_DEBUG=1 labels command buffers + prints fault descriptions). Remove
   before exporting 0073, keep until the driver story is settled.

## Driver fault story (the messy part)
- Post-reboot verify (08:18): pp512 knob-ON runs status-5'd mid-run (2/4), needle-on
  returned *hallucinated prose* pre-fix (that was the slot bug, plus possibly the race),
  off_needle = 7352 (driver healthy at that point). off_tg died SIGABRT at 08:30 (GGML_ABORT
  downstream of a GPU fault; /Library/Logs/DiagnosticReports/llama-bench-2026-09-17-083032.ips).
- After ~6 knob-ON pp512 runs WITH THE COLLISION BUG, kernel log shows repeated
  "IOAccelCommandQueue: Deny Submissions app[llama-completion] with 2 GPURestarts in 432
  submissions". From then on, faults happen on EVERY path.
- Decisive A/B (09:02-09:07, /tmp/ab_results.txt): mode0 (knob OFF, pure vendor path)
  crashed 2/4 while mode3 stayed clean. => driver session is poisoned; fault rates no longer
  discriminate code paths. HANDOFF rule: reboot, don't debug.
- Conclusion so far: collision bug (fixed) plausibly seeded the restart storm (peer remote
  reads racing cvt writes on the same pages). All numerically-completed post-fix runs were
  CORRECT. Need a fresh driver session to certify 0073-f16.

## Kernel-type decision: bf16 -> f16 transport
- The scalar bf16e path (no native bfloat on RDNA2, manual ushort struct in kernels/common.h)
  had NEVER executed on this machine (all models Q4/Q8). It was the prime suspect for the VMPT
  faults. To remove that risk entirely, the transport switched to **f16** (native half on
  these cards): kernel_cpy_contig_f32_f16 (existing) + new instantiation
  kernel_bin_fuse_f32_f16_f32 <float, half, float> in binbcast.metal. tname_pair(F16) = "f16".
- f16 accuracy for allreduce transport: 10-bit mantissa (4.9e-4 rel) - better than bf16 here;
  range |x| < 65504 (watch outliers in the needle/ppl gates; overflow would show as inf/NaN
  and break the gates loudly).
- binbcast.metal no longer instantiates kernel_bin_fuse_f32_bf16{,e}_f32.
- Knob names still say BF16 (TOSH_MGPU_STAGE_BF16, tmp_bf16, cvt_f32_bf16, ...) - mechanical
  rename to F16 pending after the next reboot validates the f16 path.

## What to do next (fresh driver session first!)
1. REBOOT (user action; each command hung/aborted runs otherwise).
2. Run the gate suite, interleaved, .bench/bf16-stage/verify.sh pattern with
   TOSH_MGPU_STAGE_DEBUG=1: needle on x3, off x1, short on, pp512 A/B x2, tg128 A/B.
3. If clean: rename bf16->f16 everywhere (knob -> TOSH_MGPU_STAGE_F16), strip debug
   labels + modes 2/3, export patches/llama/0073-metal-f16-allreduce-transport.patch
   (ref worktree: .bench/bf16-stage/ref-0072 = 465e49b9c + 0001-0065 + 0072, already built),
   then llama-perplexity spot check.
4. If mode1 faults again on a FRESH driver while OFF is clean -> the cvt/add pattern itself
   is toxic on this driver; try vec4 variants (bf16e4/half4 bin kernel <float4,half4,float4>
   with ne/4 kargs) before giving up.

## CLOSE-OUT (2026-09-17, fresh boot ~08:04): 0073 CERTIFIED + EXPORTED
- Reboot + self-heal restored the driver; every run below had ZERO status-5s and the OFF
  controls (3/3 needle) were clean, so the numbers are trustworthy per the HANDOFF rule.
- Stripped all debug scaffolding (modes 2/3, TOSH_MGPU_STAGE_DEBUG labels, tosh_stage_*;
  fixed 4 broken strip leftovers in ggml-metal-context.m: cvt's cmd_buf_end line + 3 stray
  tosh_stage_label call sites). Renames bf16->f16 all done (knob TOSH_MGPU_STAGE_F16).
- CORRECTNESS on the stripped build (verify4.sh, 09:37:56-09:38:47): needle 5/5 ON = 7352,
  3/3 OFF controls = 7352, short with -n 16 answers 108 in BOTH modes, all rc=0 s5=0.
  (short needs -n 16: at -n 8 the f16-rounded greedy start echoes "12 * 9 =" and truncates
  before the answer - a text-sampling artifact, not a wrong sum.)
- PERF on the stripped build (perf3.sh, interleaved A/B):
  pp512  off 1494.8/1500.2 vs on 1642.8/1669.5  = +9.9/+11.3 %
  pp2048 off 1407.7/1399.1 vs on 1539.1/1526.7  = +9.3/+9.1 %  (tmp_f16 grow/realloc path OK)
  tg128  off 42.93 vs on 43.08                  = parity (f32 decode path untouched)
- NUMERICS (llama-perplexity, -c 512 --chunks 4, ppl-corpus.txt = ctx.txt x7, ~3.1k tok,
  ppl1.sh + interleaved ppl2.sh): PPL off 1.4937 +/- 0.0712 vs on 1.4940 +/- 0.0712 =
  +0.02 % drift, i.e. nothing. rc=0 s5=0 everywhere. NOTE the tool's "seconds per pass"
  line is misleading (said 66.7 while wall was 73 incl. load); wall is CPU-bound and
  off itself varied 63/72 s - ppl is a SMOKE test here, the pp benches are the perf gate.
- EXPORTED: patches/llama/0073-metal-f16-allreduce-transport.patch (6 files, +301/-9),
  verified `git apply --check` clean on ref-0072. Knob default OFF. Vendor tree untouched,
  nothing committed.
- The bf16e-scalar-kernel poisoning theory stands as the explanation for the 09:02-09:30
  storm (those runs never executed f16); no fault has appeared since the reboot + fix.
- Still-open ideas (not needed for 0073 as shipped): vec4 transport for another ~1 % of
  fabric savings; default-on flip needs wider-model validation; 35B-TP4 dflash sweep
  (thread 3c) still blocked on its own driver crash.
