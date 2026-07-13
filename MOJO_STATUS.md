# Mojo Port Status

**Phase:** THE PURE-MOJO TRACK (ADR 0007) IS COMPLETE. WP2–WP10 + WP9
part 3 + WP12 (pure-Mojo loading) + WP13 (pure-Mojo DINOv3) + WP14
(pure-Mojo image IO/preprocess) are all done: the whole image->3D
pipeline runs in Mojo from PAM file to OBJ/npz, parity-verified
component by component, and Mojo beats torch on every benchmark case
against both default-thread and single-thread torch. The only Python
left in the runner is torch.randn (kept deliberately for noise-stream
compatibility with the original); Python/torch otherwise remain only in
tests/parity and benchmarks. Remaining tracks are optional/external:
WP0 golden outputs (needs a CUDA box), WP11 fp16/GPU (MAPPED 2026-07-09:
Metal GPU verified working from Mojo — see the WP11 entry below and
06_MASTER_PLAN; implementation remains), the perf queue in
docs/benchmarks/RESULTS.md, and the texturing pipeline (phase 2, out of
scope v1). Perf passes 7-8 (2026-07-09) took the steps-2 e2e smoke from
627 to 252 s.

**Copy:** Done to /Users/myrvoll/Documents/testfiler/trellis-mojo/ (21M)

**Docs:** 00_INDEX + inventory/mapping/architecture/conversion/decisions/o_voxel.

**Planner:** docs/06_MASTER_PLAN.md (work packages WP0–WP10 with dependencies and
acceptance criteria) + docs/07_PORT_TRACKER.md (per-file status — update as you go).

**Done:**
- WP1 (partial): pixi env with Mojo 1.0.0b2 + pytorch 2.12 (`pixi run mojo`);
  Mojo→Python interop verified; parity harness in tests/parity/.
- WP3 (core): `trellis2_mojo/sparse/{tensor,basic}.mojo` — VarLenTensor,
  SparseTensor, cat/unbind, elementwise + batch broadcast, getitem, to_dense,
  full, scale-keyed shared spatial cache (ArcPointer), segment reductions.
  Verified: `pixi run test-sparse` (17 unit tests), `pixi run test-parity`
  (20-seed fuzz vs the torch original — exact match).
- WP2: `trellis2_mojo/samplers/flow_euler.mojo` — FlowEuler + CFG +
  guidance-interval collapsed into one struct with a `VelocityModel` trait;
  `py_model.mojo` lets the Mojo loop drive a Python/torch model (hybrid
  direction), `trellis2_mojo/interop.mojo` is the tensor bridge. Parity:
  full trajectories, 3 seeds x 7 configs (`pixi run test-flow-euler`).
- WP4: `trellis2_mojo/modules/nn.mojo` (SparseLinear/linear, LayerNorm32,
  GroupNorm32, ChannelLayerNorm32, relu/silu/gelu, modulate) +
  `trellis2_mojo/sparse/attention/rope.mojo` (RoPE with phase caching).
  Parity: 5 seeds x all layers (`pixi run test-wp4`).
- WP5: one block-diagonal varlen-SDPA kernel serves every attention variant
  (`trellis2_mojo/sparse/attention/full_attn.mojo` sparse+dense,
  `windowed_attn.mojo` window partition, `modules.mojo`
  SparseMultiHeadAttention, `trellis2_mojo/modules/attention.mojo` dense
  MHA). Parity with real weights across all configs (`pixi run test-wp5`).
- WP6: `spatial/basic.mojo` + `spatial/spatial2channel.mojo` (down/up,
  s2c/c2s incl. cache handshake and scale bookkeeping) and `sparse/conv.mojo`
  (conv_none submanifold conv). Parity: `pixi run test-wp6`.
  Findings: `serialize.py` doesn't exist (phantom exports), nor do
  SparseSubdivide/interpolates; SparseInverseConv3d unused.

- WP7: all 8 transformer block variants (dense/sparse x plain/modulated,
  self+cross) + AbsolutePositionEmbedder + GELU-tanh.
  `trellis2_mojo/{sparse,modules}/transformer/{blocks,modulated}.mojo`,
  weights loaded from torch state_dicts via `trellis2_mojo/loaders.mojo`
  (the WP8 weight-loading seed). Parity: `pixi run test-wp7`.

- WP8.1: `trellis2_mojo/models/sparse_structure_flow.mojo` — first full
  network (dense DiT): TimestepEmbedder, APE/RoPE grid phases precomputed
  at build, block stack, final layer_norm; built from a state_dict via
  `sparse_structure_flow_from`. Dense RoPE landed with it
  (`modules/rope.mojo` + phases overloads on dense MHA/cross block).
  Parity: `pixi run test-wp8` (APE / rope+share_mod+qk_rms like the real
  checkpoint / rope with phase padding). Findings: `patchify`/`unpatchify`
  in modules/spatial.py are dead code (only `pixel_shuffle_3d` is used, by
  the SS VAE); the model's `rope_freq` arg never affects the phases buffer
  (always default freqs).

- WP8.2: `trellis2_mojo/models/sparse_structure_vae.mojo` — SS-VAE decoder
  (ResBlock3d, UpsampleBlock3d + pixel_shuffle_3d in `modules/spatial.mojo`,
  dense `Conv3d` in `modules/conv.mojo`). Encoder is training-only, not
  ported. Parity in `test-wp8`: decoder with layer and group norm plus
  standalone ResBlock3d (1x1 skip, both norms). Finding: the decoder never
  forwards norm_type to its res blocks — they always use the default
  "layer"; norm_type only affects out_layer (loader mirrors this).

- WP8.3: `trellis2_mojo/models/structured_latent_flow.mojo` — SLatFlowModel
  (sparse DiT): APE on coords or block-level rope (phases computed inside
  sparse attention, cached), share_mod, concat_cond feature-concat (the
  texture-SLat path). Loader `slat_flow_from` works for Elastic checkpoints
  (same keys). Not ported: cond as List[Tensor] (pipelines pass dense
  tensors), elastic mixin. Parity in `test-wp8` (APE / rope+share+qk_rms
  like the real checkpoint / concat_cond).

- WP8.4: `trellis2_mojo/models/sc_vaes/{sparse_unet_vae,fdg_vae}.mojo` —
  sparse UNet VAE decoder (SparseConvNeXtBlock3d + SparseResBlockC2S3d,
  the only block types the configs use) with pred_subdiv and guided modes
  + upsample_coords, and the FlexiDualGrid head (mesh extraction is now
  pure Mojo — see WP9 part 3 step 4). Encoder side and unreferenced block types not
  ported. Parity in `test-wp8` incl. coords, predicted subs, and the
  shape->tex guided handoff.

**Upstream bugs found so far:**
1. `to_dense()` on the 'none' conv backend crashes (basic.py:687).
2. `SparseLayerNorm` would crash if called — unused, not ported.
3. **Sparse attention naive/sdpa fallback attends to zero-padding when
   batch lengths differ (diff ~1.8 vs flash/xformers semantics)** — CPU
   results from the original are silently wrong; Mojo port matches the
   correct production backends (see tests/parity/torch_ref_wp5.py).
4. Windowed attention has no CPU backend at all (xformers/flash only).

**Test entry point:** `pixi run test-all` (unit tests + sparse fuzz +
sampler trajectories + layer/attention/spatial/conv parity).

- WP9 parts 1+2: `trellis2_mojo/pipelines/image_to_3d.mojo` — ALL
  model-facing pipeline stages: SSFlowVelocity/SlatFlowVelocity adapters
  plug the WP8 models into the WP2 sampler (sparse sampling reuses the
  dense loop; feats are the state, coords/cond/concat ride in the adapter),
  sample_sparse_structure (decode -> occupancy>0 -> max-pool -> argwhere
  coords), sample_slat (mean/std de-norm), decode_shape (unet + fdg_head),
  cascade_coords (quantize/unique/token-budget loop for the 1024 path),
  normalize_slat + texture sampling via set_concat, decode_tex (guided +
  *0.5+0.5). Integration parity: `pixi run test-wp9` — same weights/noise
  through shape -> cascade -> texture vs the original torch samplers +
  pipeline glue; threshold flips handled borderline-aware.

- WP10 (benchmarks + perf passes 1-6): `pixi run bench` runs the Mojo hot
  paths vs the torch originals (attention full/windowed, conv, modulated
  block, full FlowEuler sampling loop) — `benchmarks/bench_wp10.mojo` +
  `benchmarks/bench_torch_ref.py`, results in `docs/benchmarks/RESULTS.md`.
  Pass 1: SIMD dot/axpy + loop reorder in `varlen_sdpa`, SIMD dot in
  `linear`. Pass 2: `parallelize` over (segment, head) / rows /
  out-channel chunks in attention/linear/conv — disjoint write regions,
  bit-identical to the serial path, flops-proxy thresholds so small ops
  stay serial. Pass 3: q-tiling in `varlen_sdpa` (key loop outermost, k/v
  rows reused from L1 across 8 q rows; bit-identical) + SIMD in
  LayerNorm32/activation/modulate. Pass 4: register blocking — qk runs
  2 keys x 4 q rows per block (8 independent FMA chains sharing loads),
  av accumulates each out row in registers across all keys with the
  denominator division fused into the store, long q segments split into
  64-row chunks per work item for load balance (attn-L: 16 -> 256 items
  on 14 cores), and `linear` got 4 rows x 2 out features per block with
  row-block parallelization — per-pair math and per-lane accumulation
  order unchanged, so everything stays bit-identical. Pass 5: a
  packed-GEMM path in `linear` for large inputs (weight packed into
  [k][16] panels, x block into [k][4], the 4x16 output tile held in 8
  SIMD registers through the whole k loop; 270-775 GF/s — this path
  changes per-output accumulation order, within parity tolerance and
  deterministic), plus SIMD span helpers for ALL tensor primitives
  (reshape/unbind/binop/cat/slice/stack — profiling with
  benchmarks/microbench_block.mojo showed the block time was dominated
  by scalar glue ops, not matmuls) and chunk-parallelized activation.
  Pass 6: SIMD + chunk parallelization in the norm/rope layers that real
  checkpoints hit (MultiHeadRMSNorm for qk_rms 6.3 -> 1.1 ms, rope
  rotate via lane deinterleave 5.1 -> 2.3 ms, GroupNorm32 -> 0.6-1.1 ms,
  ChannelLayerNorm32 — used by every SS-VAE decoder res block, was a
  16 KB-stride scalar walk — 20-33 -> 1.7 ms). Full parity suite green
  after each pass. Net vs naive v1: attn-L 1429 -> 15.6 ms, conv
  27 -> 1.4 ms, mod-block 208 -> 3.1 ms, sampling loop 323 -> 20.6 ms,
  windowed 21 -> 1.7 ms. Mojo now beats torch on EVERY case: 0.08-0.99x
  vs default 10-thread torch, 0.1-0.6x vs single-thread. Remaining
  optional queue (RESULTS.md): uninitialized Tensor alloc, bigger GEMM
  kernels for real model shapes, rope phase-copy elimination.

**Next per plan:**
- WP9 part 3 is now UNBLOCKED (2026-07-08): `~/Documents/testfiler/
  trellis-mac/` (a working MPS port that has generated real meshes on
  this machine) provides everything that was missing — TRELLIS.2-4B
  checkpoints (14 GB in the HF cache, exact ckpt names the loaders
  expect), DINOv3 + RMBG-2.0 weights, a Python 3.11 venv with
  safetensors/transformers/trimesh/utils3d, and the pure-Python
  `stubs/o_voxel_override_convert.py` mesh extractor. See the asset map
  and suggested order in docs/08_HANDOVER.md. Step 1 is DONE (2026-07-08):
  `guidance_rescale` (used by the real pipeline at 0.7/0.5, originally
  skipped) is now ported in flow_euler.mojo with both upstream std
  semantics — dense torch .std (unbiased, per row) and VarLenTensor.std
  (biased sqrt(E[x2]-E[x]2), per segment, selected via seg_offsets) —
  wired through sample_sparse_structure/sample_slat, and parity-verified
  (2 new flow-euler cases + the whole WP9 integration now samples with
  rescale 0.7/0.5). Step 2 is DONE (2026-07-09): real-checkpoint loading —
  `trellis2_mojo/ckpt_io.py` resolves the local HF cache and casts the
  safetensors weights (bf16/fp16) to f32, `trellis2_mojo/checkpoints.mojo`
  interprets the config JSONs and builds every pipeline model through the
  existing `*_from` loaders. `interop.mojo` was rewritten to raw-pointer
  copies (torch `data_ptr()` -> `UnsafePointer(unsafe_from_address=...)`)
  — per-element conversion does not scale to 1.3B params; values are
  bit-identical and the whole parity suite got faster. Verified by
  `pixi run test-real` (own task, NOT in test-all: reads ~10 GB): all six
  models — ss_dec, shape/tex unet decoders incl. the guided handoff, and
  the three 512 DiTs at real shapes (ss_flow at 16^3 = 4096 tokens) —
  match the torch originals loaded independently from the same files,
  max|diff| 1e-5 to 3.5e-4 vs per-model atol 5e-4 to 2e-3. The 1024 slat
  variants share architecture and keys with the 512 ones and are skipped.
  Step 3 is DONE (2026-07-09): image conditioning behind a swappable
  interface — `pipelines/conditioning.mojo` (`ImageConditioner.get_cond
  (path, res)` -> [1, L, 1024]) delegating to `cond_io.py` (DINOv3 via
  transformers, offline HF cache; mirrors the upstream extractor without
  the torchvision dependency). rembg/BiRefNet is intentionally dropped
  (ADR 0007): input must be RGBA with a real alpha channel, which makes
  upstream preprocess skip rembg. Parity `pixi run test-cond`:
  BIT-IDENTICAL (0.0) to the original DinoV3FeatureExtractor at 128/512,
  preprocess pixel-identical, RGB input rejected. (WP13 has since swapped
  the interop backend for the pure-Mojo DINOv3 behind the same
  signature — see the WP13 entry below.)
  Step 4 is DONE (2026-07-09): mesh extraction in PURE Mojo —
  `trellis2_mojo/meshing/fdg_mesh.mojo` ports the trellis-mac o_voxel
  stub ("identical output to the CUDA version for inference"):
  Dict-based coord lookup (packed 21-bit keys), edge-neighbor table,
  quads in (i, axis) order, both split modes (split_weight = pipeline
  path, normal alignment for completeness), stub's empty early-out
  (0 quads -> 0 vertices), plus `write_obj`. Parity `pixi run test-mesh`
  (INCLUDED in test-all — fast, cache-independent) vs a vendored stub
  copy (tests/parity/o_voxel_stub.py, with a documented VENDORED-FIX:
  torch.cross needs explicit dim=1): triangles exact, vertices within
  1.2e-7, degenerates, OBJ round-trip re-read by Python.
  Step 5 is DONE (2026-07-09): the end-to-end runner —
  `run_image_to_3d.mojo` at the repo root (`pixi run e2e -- image.png
  [--seed N] [--steps N] [--out prefix] [--no-tex]`), Mojo host for the
  full 512 pipeline: conditioning -> ss flow -> shape slat -> tex slat
  -> decode -> pure-Mojo mesh extraction -> OBJ + texture-voxel npz
  (pbr layout base_color/metallic/roughness/alpha + origin/voxel_size).
  Sampler params / normalization come from pipeline.json; models load
  one at a time (~6 GB peak RSS); noise is drawn from the torch stream
  in the exact upstream order (seed AFTER conditioning — model
  construction, not inference, consumes RNG — so the ss noise is
  bit-identical to a same-seed trellis-mac run). Verified structurally
  against trellis-mac (MPS/bf16) on shoe_3q.png seed 42; numbers in
  docs/08_HANDOVER.md. Remaining Python in the path (per ADR 0007,
  removed by WP12/WP13): ckpt_io, cond_io, torch.randn.
  WP9 del 3 is COMPLETE.
- WP12 is DONE (2026-07-09): the whole loading path is pure Mojo —
  `trellis2_mojo/io/` with an arena-based mini JSON parser (json.mojo),
  a sequential per-tensor safetensors reader (safetensors.mojo; macOS
  caps read() at 2 GiB so >2 GB DiT files cannot be slurped whole;
  bf16 = u16<<16 via SIMD(from_bits=), f16 via the hardware cast,
  alignment=1 loads), HF-cache resolution (hf_cache.mojo) and a
  StateDict facade (state_dict.mojo) that every loader now takes — an
  @implicit PythonObject constructor keeps the parity tests' torch
  dicts working unchanged. Parity `pixi run test-io` (not in test-all,
  reads ~14 GB): all 8 checkpoints BIT-identical to ckpt_io.py
  (7.48e9 values), paths equal, pipeline.json floats exact vs Python's
  json; test-real green with the Mojo reader in the load path; full
  test-all green; runner smoke re-verified. ckpt_io.py remains only as
  the torch-side reference for parity tests.
- WP13 is DONE (2026-07-09): DINOv3 ViT-L/16 in pure Mojo —
  `trellis2_mojo/models/dinov3.mojo` mirrors the transformers model as
  the upstream extractor drives it (manual layer loop + final
  NON-affine layer_norm; the checkpoint's own `norm.*` is never used):
  patch conv 16x16 as im2col + linear, [cls; 4 registers; patches],
  2D rope theta=100 (inv_freq over head_dim/4, angle row [y-part,
  x-part] + tile(2), rotate_half split — NOT the per-pair interleave
  the flow models use; patch tokens only; pos_embed_rescale is a
  train-time augment and skipped in eval), separate q/k/v projections
  (k has NO bias), LayerScale, exact-erf gelu MLP (non-gated).
  Weights (1.2 GB f32) load through the WP12 reader:
  `checkpoints.mojo::load_dinov3()`. `ImageConditioner.get_cond` keeps
  its signature; cond_io.py shrinks to preprocess + `pixels()`
  (transformers is out of the runner path). Parity: `pixi run
  test-wp13` (IN test-all — small random config vs transformers,
  max|diff| ~1.7e-6, incl. a non-square grid) and `pixi run test-cond`
  (real ViT-L weights, max|diff| 3.4e-5 at 128/512, atol 5e-4).
- WP14 is DONE (2026-07-09): image IO + preprocess in pure Mojo — the
  last Python out of the runner. `io/image.mojo` reads PAM P7 / PPM P6
  (PNG stays a documented PIL one-liner conversion, README_MOJO.md);
  `imaging/resize.mojo` mirrors Pillow's fixed-point Lanczos BIT-exactly
  (PRECISION_BITS=22, clip8, u8 quantization BETWEEN the separable
  passes — a float implementation cannot match; plus the RGBa
  premultiply round-trip PIL applies to RGBA resize: MULDIV255 in,
  TRUNCATING division out, alpha 0/255 pass-through, found empirically;
  plus the copy() short-circuit for same-size BEFORE that round-trip);
  `imaging/preprocess.mojo` ports the alpha branch (bbox > 204, square
  crop with Python round-half-even + zero padding, premultiply with
  numpy's u8 truncation, ImageNet normalize). Parity `pixi run
  test-wp14` (IN test-all): everything bit-exact vs PIL (rasters, 5
  resize cases, preprocess at 640 and 1500 incl. the >1024 downscale,
  normalized pixels f32 max|diff| 0.0, RGBA-requirement rejections).
  ImageConditioner is now 100% Mojo; cond_io.py is reference-only (like
  ckpt_io.py). End-to-end proof: the steps-2 smoke re-run from a PAM of
  the same shoe image produced a BYTE-IDENTICAL OBJ (cmp) to the WP13
  smoke. README_MOJO.md rewritten with a full run example.
- Perf pass 7 (2026-07-09, after WP13/14): e2e profiling
  (microbench_dit_block/load/conv3d.mojo) showed the 519 s ss stage was
  NOT the DiT (linear runs 700–900 GF/s at real shapes, a whole
  cross-block ~550 ms) but dense Conv3d — the SS-VAE decoder's entire
  cost, still naive at a MEASURED 0.86 GF/s (67.6 s per 512-channel res
  conv at 16^3). SIMD over the innermost dim (W8+W4 interior rungs),
  OU=4 out-channel register blocks sharing x loads, parallelized items;
  edge/tail/strided keep the original scalar order -> bit-identical
  (byte-identical smoke OBJ). 750 ms on the hot shape (90x); e2e smoke
  (steps 2): ss stage 519->162 s, total 627->271 s. Updated queue in
  docs/benchmarks/RESULTS.md.
- Perf pass 8 (2026-07-09): flash path in varlen_sdpa for segments with
  kv_len >= 1024 (online softmax over 128-key blocks, scores/acc in L1,
  the 4x4 qk register blocks kept, 2-row av). NOT bit-identical
  (pass-5 precedent) — but every parity-test shape stays on the exact
  kernel (cond(128) unchanged proves it); real-weight verification:
  test-cond 4.1e-5, test-real ss_flow 1.1e-4. Attribution first
  (microbench_sdpa.mojo, degenerate ci/co dims): the kernel was bound
  by scores materialization + v re-streaming, not FMA — three cheaper
  hypotheses (SIMD softmax, KU=4, uninit scratch) measured ~nothing.
  self-sdpa 284->197 ms; e2e smoke 271->252 s, structurally unchanged.
  NOTE: OBJs are no longer byte-comparable across flash changes —
  compare structurally.
- WP11 MAPPED (2026-07-09) + STEP 2 DONE (2026-07-10): the tiled Metal
  GEMM (2.2-2.7 TF/s standalone) is wired behind `linear` —
  `trellis2_mojo/gpu/linear.mojo`, enabled by TRELLIS2_GPU=1 (CPU
  fallback otherwise). One shared GpuContext rides on the StateDict
  into the loaders; W^T is uploaded once per model load and
  SparseLinear.forward dispatches on shape (co%64, ci%16, weight and
  rows*co*ci thresholds — measured break-even). Bias is added on the
  CPU in a chunked-parallel readback pass. Parity: `pixi run
  test-wp11` (in test-all), max|diff| <= 4.3e-6 vs the CPU paths.
  Forward-level speedups incl. transfers: 1.36-1.70x on the hot DiT
  shapes at 4096 tokens; small-output shapes stay on CPU. FOUR new
  b2-Metal API traps found and documented in the gpu/linear.mojo
  header: >4 total kernel-arg bindings hands the kernel garbage
  (all scalars together count as one binding); NOTHING commits until
  a map_to_host of a host-written buffer (ctx.synchronize() does not
  commit — a 1-element fence buffer is the commit+wait primitive);
  mapped memory is write-combined (single-thread reads 2.2 GB/s,
  parallel reads scale 4.2x); enqueue_copy ignores Span lengths
  (always full buffer).
- WP11 STEP 3 DONE (2026-07-10): dense SDPA on the GPU as a GEMM
  composition with device-resident scores (gpu/attention.mojo) — qk
  batched over heads via grid z, masked row softmax + row sums, av,
  1/sum fused into the CPU readback; scale pre-baked into q. Wired
  into dense MultiHeadAttention (self/rope/cross) through
  dense_mha_from; shared GpuContext refactored to gpu/context.mojo.
  Parity `pixi run test-wp11-attn` (in test-all -> 14 files),
  max|diff| <= 2.3e-7. Measured: self-4096 H16 207.7->57.4 ms
  (3.62x), cross 4096x1029 58.3->20.1 ms (2.91x). FIFTH Metal trap:
  the first full write->kernel->read cycle in a process delivers
  corrupt reads at 256-byte boundaries (independent of fences and
  warm-up launches) — GpuContext burns a sacrificial cycle and runs
  a VERIFIED self-test at creation, falling back to CPU on failure.
- WP11 STEP 4 DONE (2026-07-10): varlen/sparse SDPA via q-padding —
  _sdpa_core pads BOTH sides to 64 (zero q rows are dropped in the
  readback; the composition is q-row-independent), so arbitrary
  lengths work. gpu_varlen_sdpa_single takes varlen_sdpa's [T, H, D]
  layout for the B=1 single-segment case; SparseMultiHeadAttention
  (full self + cross) dispatches through sparse_mha_from (windowed /
  multi-segment stay on CPU). Parity: test-wp11-attn extended to 7
  shapes + both MHA dispatches, max|diff| <= 2.3e-7. Measured:
  slat-self @2369 H16 67.2->23.0 ms (2.92x), slat-cross 2.10x.
  e2e smoke: 171 -> 156 s (slat stage 55->40 s), structurally
  identical (1 borderline voxel of ~950k).
- WP11 STEP 5 DONE (2026-07-10): device-resident mlp chaining —
  gpu_mlp_forward runs lin2(gelu_tanh(lin0(x))) with the [rows, hidden]
  intermediate on the GPU (the 134 MB/block WC round-trip is gone).
  lin0's bias lands on the GPU before the gelu (bias_dev); the
  gelu-tanh kernel computes tanh THROUGH EXP (the GPU library tanh is
  a fast ~2e-3 approximation; exp is precise). Wired into both FFN
  structs. Parity in test-wp11 (mlp chain + FFN dispatch, 1.6e-7).
  Measured: ss-mlp 4096x1536->8192 chain 84.2 ms vs 131.8 unchained
  vs 279.1 CPU (3.31x). e2e smoke: 156 -> 136 s = 1.85x vs pure CPU
  (ss sampling 66->59 s, slat 36->28 s), structurally identical
  (12 borderline flips of ~950k). PROBE LESSON: always build GPU
  probes on GpuContext (verified self-test) — hand-rolled sacrificial
  cycles are not reliably sufficient.
- WP11 STEP 6 DONE (2026-07-10): sparse conv on the GPU — decode
  instrumentation showed conv dominance (23 s ConvNeXt + most of the
  17 s upsample blocks of the 43 s decode). gpu/conv.mojo: the CPU
  edge lists are stable-counting-sorted to CSR by target on the host;
  one gather kernel walks each output row's edge range (thread =
  row x 8 co-lanes sharing the broadcast x scalar). Weight uploaded
  once per model as [K, Ci, Co]; ALL dims ride in the edges-buffer
  header (4 pointers leave no scalar binding). Parity:
  test-wp11-conv (in test-all -> 15 files), max|diff| <= 4.1e-5.
  Measured 3.0-3.5x on decode shapes; e2e smoke: decode 43->19 s,
  total 136->116 s = 2.17x vs pure CPU, structurally identical.
- WP11 STEP 7 DONE (2026-07-11): device-resident attention chaining —
  gpu_attn_self_chain runs the WHOLE self-attention (qkv-GEMM ->
  fused bias+qk-rms+rope kernel -> head-major pack -> the step-3 SDPA
  composition -> unpack with 1/sum fused -> out-GEMM) with only x
  uploaded and only [T, C] read back. Per-MHA constants (qkv bias +
  both rms gammas) ride in ONE device buffer (GpuAttnChain, built by
  the mha_from loaders AFTER gamma assignment) to stay inside the
  4-binding marshalling law; the sdpa scale is recomputed in-kernel
  from the Int head-dim (no float scalar args). Rope phases upload
  per call (dense: passed in; sparse: from coords via the same
  spatial cache as the CPU path). Gate: sdpa gate + the qkv linear's
  flops threshold (the out linear rides for free). Parity:
  test-wp11-attn extended with 3 whole-MHA chained cases (dense
  plain, dense rms+rope, sparse rms+rope) + gate checks, max|diff|
  <= 4.8e-7 — green on the FIRST run. Measured (whole MHA forward):
  ss_flow geometry (4096x1536 H12 D128) 171.8->90.3 ms (1.90x vs
  unchained GPU, 5.98x vs CPU); slat (2369x1024 H16 D64) 66.4->28.2
  ms (2.35x, 4.19x vs CPU). e2e smoke: total 116->94 s = 2.68x vs
  pure CPU (ss 61->48 s, slat 30->23 s); structurally identical
  (exact same 948 578 voxels, 2 triangle flips of ~2M).
- WP11 STEP 8 DONE (2026-07-11): cross-attention chaining — block
  profiling with the GPU on (microbench_gpu_block.mojo, real ss_flow
  geometry) showed cross-attention had become the LARGEST post
  (92.6 ms vs self 87 / mlp 81). gpu_attn_cross_chain runs
  q-GEMM -> fused bias+q-rms -> pack -> sdpa -> unpack -> out-GEMM
  device-resident; kv is computed and k-rms-normalized on the CPU
  (the kv linear at ~1k context rows is below the GPU break-even)
  and host-packed BEFORE the enqueues (mapping a host-written buffer
  commits pending work — ordering is law). pack_q_z gained a stride
  parameter; GpuAttnChain.try_build_cross builds [HD q-bias]
  [HD gamma_q] consts. Gate wants_cross = the sdpa gate alone — the
  q/out GEMMs ride on the upload sdpa needs anyway (verified: the
  slat cross geometry wins 1.56x despite its q linear being under
  the solo proxy threshold). Parity: 3 cross cases + build declines,
  max|diff| <= 4.2e-7 — green on the first run. Measured (whole
  cross-MHA): ss 4096x1029 C1536 86.3->51.4 ms (1.68x, 3.02x vs
  CPU); slat-geometry 44.0->28.1 ms (1.56x). Block total
  289.6->244.7 ms. e2e smoke: 94->88 s = 2.86x vs pure CPU (ss
  48->45 s, slat 23->20 s); structurally identical (1 borderline
  voxel flip of ~950k).
- WP11 STEP 9 DONE (2026-07-11): conv-kernel register blocking —
  sparse_conv_gather now processes TWO target rows x 8 co-lanes per
  thread, merge-walking the rows' edge lists on kidx (ascending
  within a row: the kidx-major build order survives the stable
  counting sort), so kernel offsets present in both rows load each
  weight line ONCE for two rows (the weight-plane slices are the
  dominant traffic). Per-row edge order and accumulator math are
  unchanged -> BIT-identical to the single-row kernel (parity diffs
  unchanged). Measured (incl. CSR+transfers): 1.42-1.57x on top of
  step 6 — 512ch@12k 216.7->152.4 ms (4.31x vs CPU), 256@55k 5.43x,
  128@216k 5.07x, up-conv1 512->2048 3.91x.
  e2e smoke: 88->86 s = 2.93x vs pure CPU (decode 19->17 s); the OBJ
  is BYTE-identical to the step-8 smoke (cmp) — the kernel is
  bit-identical per row as designed.
  QUEUE NOTE: "windowed attention batching" is DROPPED — the slat DiT
  (structured_latent_flow.py:71) uses attn_mode='full' for every
  block, so windowed attention is not in the 512 runner path at all.
- WP11 STEP 10 DONE (2026-07-11): whole-block residency —
  gpu/block.mojo runs the ENTIRE cross-block (dense and sparse share
  the orchestrator on flat feats) device-resident: ln+modulate ->
  self chain -> gate_add -> affine ln -> cross chain -> add ->
  ln+modulate -> mlp chain -> gate_add, with ONE x upload, ONE
  barrier and ONE readback per block (vs six transfers + ~28 ms CPU
  glue). The chains were refactored into enqueue-only parts with
  device-buffer in/out and NO out-bias (host wrappers fuse it in the
  readback as before; the block path folds it into the gate_add
  kernel's consts). ALL host uploads happen BEFORE any enqueue
  (mapping a host-written buffer commits the queue — law 2); the glue
  consts ride in ONE buffer indexed by Int offset scalars (offset
  POINTERS are untested against the marshalling laws); cross-kv got
  DEDICATED ckt/cvh buffers (the self chain's device pack owns
  kt/vh in the fused queue). The LN kernel mirrors LayerNorm32
  (biased var, eps comptime 1e-6 — the dispatch gate checks eps and
  the norm layout). Parity: 2 whole-block cases (dense rms+rope,
  sparse rope-from-coords), max|diff| <= 3.7e-6 (atol 1e-3). Measured
  (ss geometry): 249.2->211.4 ms/block (1.18x). e2e smoke: 86->77 s =
  3.27x vs pure CPU (ss 45->40 s, slat 20->16 s); structurally
  identical (12 borderline flips of ~950k — the LN numerics moved in
  every block, same class as the gelu pass; bbox equal).
- MEASURED NEGATIVE RESULTS (2026-07-11, microbench_gpu_gemm on real
  DiT shapes) — two queue items CLOSED without wiring: (1) GEMM
  register blocking: a 64x128/4x8 variant gained only +6%, 128x128/8x8
  +3% — the kernel is not register-bound. (2) bf16-stored B weights
  (u16<<16 on the shared fill; EXACT for the bf16 DiT checkpoints):
  FLAT within noise — the B tiles are L2-cached, so fp16/bf16 weight
  storage buys load time/memory only, not kernel time. The kernel
  sits at ~2.9 TF/s of ~9 theoretical (shared-memory/issue-bound);
  without simdgroup_matrix access in 1.0.0b2, further GEMM tuning is
  low-ROI.
- WP11 STEP 11 (2026-07-11): CSR caching — the conv CSR sort depends
  only on the edges, so SparseConv3d.forward now spatial-caches the
  sorted (row_start, src, kidx) next to the neighbor map; before,
  every conv on the same coords re-ran the counting sort per call.
  GpuSparseConv.forward takes the pre-sorted lists and fills the
  int32 pack chunk-parallel. Parity unchanged (bit-identical).
- WP11 STEP 12 DONE (2026-07-11): model-level residency —
  gpu_cross_block_forward split into upload/enqueue/readback
  primitives; the blocks gained _gpu_enqueue_resident (CPU prep +
  enqueue against the resident xs); BOTH DiT forwards (ss_flow +
  slat) keep x device-resident across ALL 30 blocks when every block
  passes the block gate: ONE x upload + ONE readback per FORWARD
  (was per block), rope phases uploaded once per forward. The
  per-block consts/kv uploads act as the inter-block syncs (mapping
  commits + waits). BIT-identical to the per-block path — verified
  two ways: a 2-block resident driver vs sequential in the parity
  test (diff == 0.0 EXACT) and the smoke OBJ byte-identical to step
  11. e2e smoke: 77->72 s = 3.50x vs pure CPU (ss 40->37 s, slat
  16->14 s).
  Next: bf16 weight storage for load time/memory only; otherwise the
  WP11 GPU queue is harvested (breakdown now: ss 37 s, decode 17 s,
  slat 14 s, cond 2 s).
- WP11 STEP 13 DONE (2026-07-11): sdpa gate floor 2048 -> 1024 +
  GOLDEN GPU VERIFICATION — the full-quality golden run (12 steps +
  texture, seed 42) landed at 1857 slat tokens, BELOW the old 2048
  floor, so the whole slat DiT fell back to CPU (177 s). Measured
  (H12 D128): 1857 self 3.35x, 1280 2.14x, 1024 still 1.81x on the
  GPU -> GPU_SDPA_MIN_Q lowered to 1024 (test gate checks moved to
  512; below 1024 is unmeasured). Golden GPU run: 244 s = 4.1 min
  vs the 27.4-min CPU golden = 6.74x (old floor: 449 s; slat
  177->48 s, tex-slat 103->30 s). Structural vs the CPU golden
  (cKDTree/trimesh, same method as before): exact same 1857 @32^3,
  514 604 vs 514 603 @512^3, V-dev 0.000%, bbox diff 0.0, NN mean
  2.9e-08 / max 8.8e-04 (< half a voxel; the CPU-vs-mac reference
  was mean 2.1e-3), 0 degenerate faces, healthy tex npz. Artifacts:
  outputs/shoe_3q_mojo_gpu_seed42.obj + _texvoxels.npz.
- WP11 STEP 14 DONE (2026-07-11): 16-bit device weight storage —
  GpuLinear stores W^T as u16 bits (bf16 expanded u16<<16, f16 by
  hardware cast, both on the shared-memory fill) whenever EVERY
  weight is bit-exactly representable (parallel SIMD or-scan at
  try_build): the bf16 DiT checkpoints classify bf16, the fp16 unet
  decoders f16, everything else (random test weights, DINOv3) stays
  f32. All weight-GEMM call sites (forward, mlp chain, attention
  chains, block queue) dispatch through GpuLinear.enqueue_gemm;
  allow_16bit=False forces f32 (tests/debug). The expansion is
  bit-exact, so this is NOT a numerics variant: smoke OBJ
  BYTE-identical, all existing parity numbers unchanged, new
  test-wp11 cases (bf16/f16/mixed + bit-identity vs f32 storage)
  green on first run; real-ckpt classification verified with
  tests/probe_wfmt_real.mojo. Device W^T per 1.3B DiT ~4.8 -> 2.4 GB
  (unified memory); process max RSS unchanged (~11-12 GB — the peak
  sits in the decode-stage activations, not the DiT weights); e2e
  71-72 s unchanged.
- WP11 STEP 15 DONE (2026-07-11): f16-stored sparse-conv weight —
  the step-14 pattern applied to GpuSparseConv ([K, Ci, Co] as f16
  bits, hardware cast per weight-line load; shared wfmt_scan; bf16
  deliberately unsupported — the 4-pointer marshalling limit leaves
  no scalar for a format flag, and no conv checkpoint is bf16).
  UNLIKE the GEMM (B tiles L2-cached, bf16 measured flat), the
  gather kernel STREAMS weight lines, so halving the weight traffic
  is a real speedup on weight-heavy shapes: 512ch@12k 1.14x,
  up-conv1 1.13x (256/128ch flat); e2e decode 17 -> 14-15 s, total
  71 s. OBJ BYTE-identical (exact expansion); test-wp11-conv
  extended (f16/mixed + bit-identity vs f32 storage), real decoder
  conv weight classifies f16 (probe extended). GOLDEN re-verified
  after steps 14+15 (12 steps + texture): BOTH the OBJ and the tex
  npz are BYTE-identical to the step-13 golden artifacts — this also
  covers the texture path (tex-slat DiT + tex decoder) that the
  smokes never exercise; 247 s total (step 13: 244 s — noise; decode
  10+11 -> 9+10 s). The WP11 GPU queue is now COMPLETELY harvested —
  nothing optional remains locally.
- PHASE 2 / WP15 DONE (2026-07-11, ADR 0008): textured GLB export via
  vertex attributes — upstream to_glb is hard-CUDA-bound
  (cumesh decimation/UV-unwrap, nvdiffrast baking, cv2 inpainting) and
  without decimation the FDG vertices sit AT voxel resolution, so
  per-vertex sampling carries the same information as a 2048^2 bake.
  New: meshing/vertex_attrs.mojo (pure-Mojo grid_sample_3d trilinear
  with the EXACT flex_gemm semantics — truncated p±0.5 neighbors incl.
  the p<0.5 duplicate quirk, weights prod(1-|n+0.5-p|), misses weigh 0,
  renorm clamp 1e-12 — plus area-weighted vertex normals), io/glb.mojo
  (pure-Mojo GLB 2.0 writer: POSITION/NORMAL/COLOR_0/u32 indices +
  pbrMetallicRoughness with global factors; upstream axis swap), and a
  runner stage sampling the [T,6] PBR volume at the mesh vertices
  (base_color+alpha -> COLOR_0; metallic/roughness -> factor means;
  full per-voxel PBR stays in the npz). test-wp15 green on first run
  (trilinear <= 6e-8 vs a plain-torch reimplementation, GLB round-trip
  BIT-identical via a dependency-free reader); test-all = 16 files.
  Golden (12 steps + texture): OBJ/npz BYTE-identical to the golden
  artifacts, GLB stage < 1 s @ 514k vertices, COLOR_0 matches an
  independent vectorized numpy reference at 2.4e-7, material factors
  at 1e-11, trimesh loads cleanly (514 604 V / 1 055 568 F, 33 MB).
  Deliberately out of scope (ADR 0008): decimation/remesh + UV baking +
  inpainting (only meaningful together; all CUDA-bound upstream).
- PHASE 2 / WP16 DONE (2026-07-11, REVISED same day): micro-hole
  filling in the GLB export — the FDG extraction is non-watertight by
  construction (the user-visible micro holes exist in trellis-mac's
  output too); upstream fills them with
  cumesh.fill_holes(max_hole_perimeter=3e-2). V1 assumed "hole ==
  clean cycle" and walked boundary chains (figure-eight split, full
  dead-end revert) — filled 524/~526 such cycles, but the user STILL
  saw holes: every BRAIDED cluster at the FDG's non-manifold junction
  vertices stayed open. The CuMesh source turned out to be PUBLISHED
  (github.com/JeffreyXiang/CuMesh, src/{connectivity,clean_up}.cu):
  its fill_holes is COMPONENT-based, not cycle-based — union-find
  connected components of boundary edges, reject only components
  containing a degree-1 (dead-end) vertex (junctions with degree > 2
  are allowed), fill the WHOLE component with ONE centroid (mean of
  edge midpoints) and a (b, a, centroid) triangle per boundary edge.
  meshing/postprocess.mojo rewritten to that formulation (simpler AND
  correct); test-wp16 updated (shared-vertex holes fill as ONE
  component). Measured on the 1024 golden: the cycle walk left 2 204
  braided components (13 654 boundary edges) open — the component
  fill takes them all. Remaining boundary edges are dead-end seam
  paths that cumesh skips too (the degree-1 criterion). GLB only —
  OBJ/npz stay raw. LESSON: read the actual source BEFORE designing
  an "intent port" — the clean-cycle assumption cost a revision round.
  V3 (same day): the user STILL saw holes — measured: 100% of the
  dead-end vertices sat on NON-MANIFOLD edges, i.e. hole RINGS where
  one edge is shared by 3+ faces read as open paths (22 087 such
  components @1024). Upstream interleaves repair_non_manifold_edges
  between the fill calls (and GitHub issue #105 confirms holes persist
  upstream too without remeshing). Ported cumesh's repair: CORNER-based
  union-find (every face corner its own vertex; merge only across
  manifold edges, matched by vertex id) -> non-manifold fans split
  into manifold sheets where ALL boundary is closed loops -> a second
  fill closes the rings. Runner sequence fill -> repair -> fill.
  Golden numbers: 512: 929 + 19 184 components closed (648 280 V),
  1024: 4 103 + 53 909 (2 380 344 V); final state: 0 non-manifold
  edges (was 124k @1024), 0 dead ends (was 46k), 0 fillable
  components; remaining boundary edges are exclusively CLOSED seam
  rings above the 3e-2 threshold (zero-width coincident sheet
  borders). COLOR_0 checks green, bounds unchanged, postprocess
  5 s @ 2.3M V. Holes larger than the threshold are model-generated
  missing geometry — upstream's answer to those is the remesh branch
  (out of scope, ADR 0008).
  V4 (same day): the VISIBLE symptom's root cause — FDG winding is a
  ~50/50 coin flip (995 684 same-direction manifold pairs; culled
  backfaces + cancelled vertex normals render as micro holes). Ported
  cumesh unify_face_orientations (parity union-find over faces, flip
  odd parity): 995 684 -> 6 845 (99.3%). A GLOBAL per-sheet in/out
  vote was tried in TWO variants (direct occupancy probing — useless,
  the pipeline voxels are a surface SHELL; flood-fill-derived
  inside/outside at 256^3 — ALSO 51%) and REMOVED per the user's
  remove-what-didn't-help instruction: the FDG surface is folded/
  double-layered in places, so any consistent orientation shows both
  signs to local probes (zero-sum vote). cumesh has no global
  decision either; rendering is guaranteed by doubleSided=true.
  Sequence: fill -> repair -> fill -> unify(parity). Final artifacts:
  outputs/shoe_{512,1024}_final.*; all wp16b/c/d/e intermediates and
  step smokes deleted (outputs/ 3.5 GB -> ~600 MB). NO CUDA anywhere:
  CuMesh was only ever READ as an algorithm reference — everything
  that runs is pure Mojo.
- PHASE 2 / WP17 DONE (2026-07-11): the 1024 cascade + head-grouped
  sdpa — `--pipeline 1024` runs upstream's default '1024_cascade'
  (cond@512+1024, ss@32^3, LR slat with the 512 DiT, the shape
  decoder's 4-level subdivision upsample quantized to 64^3, HR slat
  with the 1024 DiT, tex with the 1024 DiT, everything downstream at
  1024^3; stage functions take the model key; the token-budget loop is
  a no-op for the 1024 target). SIXTH b2-Metal law (probed): kernel
  writes past a 4 GiB byte offset within one buffer binding are
  SILENTLY lost — a full-H scores buffer for the HR slat (T~12k ->
  9 GB) can never work. Solution: _enqueue_sdpa_groups runs the
  qk -> softmax -> av composition in HEAD GROUPS against one scores
  scratch <= 2^28 floats (unchanged cap); the gate became per-head.
  One group == the old path exactly (all existing parity numbers
  unchanged); new 15+1-group case <= 2e-7, green on first run.
  HR-slat sampling 698 s (CPU attention) -> 120 s (5.8x). Golden
  @1024 (12 steps + texture): 836 s = 13.9 min (512: 244 s), 7545
  tokens @64^3 -> 2 058 563 voxels @1024^3 = EXACTLY 4.00x the 512
  golden -> 2.06M V / 4.16M F, 2080 micro holes filled, GLB check
  green (COLOR_0 3e-7 vs a sparse numpy reference — a dense 1024^3
  volume is 26 GB, use searchsorted), peak RSS 16.6 GB. Note: the
  1024 tex model predicts metallic mean 0.73 for the shoe (512:
  ~0.0001) — a model property, our npz/GLB match its output exactly.
- PHASE 2 / WP16 v6 DONE (2026-07-12): ported cumesh
  remove_small_connected_components(1e-5). KEY FIND: the CuMesh source
  lives LOCALLY in trellis-mac/deps/mtlmesh/src/ (mtlmesh is a Metal
  port of CuMesh with python bindings) — no GitHub fetch needed.
  Semantics read from clean_up.cu/connectivity.cu: face components
  union ONLY across manifold edges (exactly 2 incident faces — a
  shared vertex or non-manifold edge does NOT connect), component
  area = sum of 0.5*|cross|, drop components < min_area, then compact
  unreferenced vertices in original order. Runner sequence now mirrors
  upstream to_glb minus the CUDA-bound simplify steps:
  fill -> repair -> remove_small(1e-5) -> fill -> unify. Goldens
  regenerated 2026-07-12: @512 drops 13 897 fragment sheets / 60 472
  faces (GLB 648 280 -> 551 771 V, 902 sheets), @1024 drops 34 597 /
  167 997 (GLB -> 2 126 855 V, 81 sheets); OBJ/npz BYTE-identical
  (GLB-only path); check_glb_512/1024 green, unit normals, trimesh
  clean; independent numpy union-find: 0 components < 1e-5, 0
  non-manifold edges, 0 dead ends left in both GLBs.
- OPEN (2026-07-12, revised after v6): small dark specks on the knit
  texture (user screenshot; concentrated on the bumpy upper — smooth
  areas clean). NOT a regression; every mesh-level hypothesis fixed or
  measured out (closable rings, braided rings, non-manifold-broken
  rings, mixed winding, zero normals, and now fragment sheets — the
  visual effect of v6 is NOT yet user-verified). DIAGNOSTIC A/B DONE
  (tests/checks/ab_mac_vs_mojo.py): the same-seed trellis-mac raw mesh
  has the SAME structure class (11 040 fragment components < 1e-5, 42k
  non-manifold edges, 15k dead ends vs our 13 909/51k/17k) — the raw
  pathology is MODEL NATURE (issue #105), not a porting artifact.
  Remaining explanations: cracks between self-touching folds (the
  closed seam rings above the fill threshold live exactly there) and
  genuine model pits. VISUAL A/B DECIDED (2026-07-12): a textured
  upstream GLB turned out to be possible on this machine after all
  (trellis-mac postprocess_cpu.py::to_glb =
  fast_simplification/xatlas/MPS baking);
  outputs/shoe_3q_mac_seed42_tex.glb (same seed, their fill/cleanup +
  simplify + UV baking) shows THE SAME SPECKS — user-confirmed. The
  mesh-cleanup track is EXHAUSTED; the specks are model nature all
  the way through upstream's own export. The user chose SEAM SEWING
  over the remesh branch.
- PHASE 2 / WP16 v7 DONE (2026-07-12): sew_boundary_seams — seam
  sewing, OWN semantics (not an upstream port). The leftover closed
  seam rings are zero-width sheet borders with BIT-IDENTICAL duplicate
  positions (the corner split duplicates exactly; the FDG's
  self-touching folds emit exact duplicates too). Weld every group of
  position-coincident BOUNDARY vertices into one vertex (exact f32-bit
  equality, NO epsilon — near-coincident geometry is real geometry;
  63-bit spatial hash of the low-21 mantissa bits + exact-equality
  chain per bucket, first-seen representative = deterministic), drop
  faces that degenerate, compact. The crack disappears topologically,
  vertex normals average across the former seam (killing the dark
  shading line), a fill pass after the weld closes rings that only
  became closable by it, and unify propagates winding across it.
  Runner sequence: fill -> repair -> remove_small -> fill -> SEW ->
  fill -> unify. NEVER run repair after sew (it would sever the
  welds). Goldens: @512 welds 24 798 boundary verts (0 degenerate),
  541 components closed post-weld, final GLB 527 514 V / 1 046 916 F,
  boundary edges 59 794 -> 16 456 (-72%); @1024 welds 30 230, 498
  closed, GLB 2 097 123 V / 4 188 429 F, boundary 72 243 -> 19 543
  (-73%). OBJ/npz byte-identical; GLB checks green; test-all (17
  files) green. Known leftovers (deliberate): non-manifold edges
  recreated at weld junctions (6 502 / 8 608 — harmless, unify skips
  them), and 145 sub-threshold loops @512 living in MIXED boundary
  components with attached dead-end paths (the cumesh fill criterion
  rejects those; loop-level filling is the next candidate IF specks
  persist). Visual verification of the v7 GLBs by the user is the
  open item; the remesh branch (reference LOCAL in
  trellis-mac/deps/mtlmesh/src/remesh/, 223-line dual contour +
  Metal port) remains the big fallback.
  Verification scripts live in tests/checks/.
- PHASE 2 / WP18 DONE (2026-07-12, @512 only per the user's
  aggressive-first plan): the remesh branch — meshing/remesh.mojo
  ports cumesh.remeshing.remesh_narrow_band_dc + the
  simple_dual_contour.cu kernel (both read from the LOCAL mtlmesh
  source). Instead of patching the FDG topology, extract the OFFSET
  surface UDF(p) - eps = 0 (eps = 1 voxel) of the unsigned distance
  field around the raw mesh with narrow-band dual contouring: cracks
  and holes narrower than ~2 eps are swallowed BY CONSTRUCTION, the
  result is closed with globally consistent winding (quad orientation
  follows the field-crossing signs), and project_back=0.9 snaps
  vertices 90% back to the original surface. Deviations (documented
  in the module header): uniform triangle-grid CSR (R/4 cells, 3^3
  neighborhood, r_max 4 voxels) instead of cuBVH; direct
  triangle-AABB stamping + exact UDF filter instead of the
  coarse-to-fine octree (same voxel set); f64 internally; NINTH
  upstream bug found (remeshing.py:220-231 — the split-alignment
  test indexes columns 1,2,3 instead of triangle 2 at columns 3,4,5,
  comparing triangle 1 with itself / a zero normal, so split 1 always
  wins; we port the INTENT). Runner flag `--remesh` REPLACES the
  whole cleanup chain in the GLB path; color sampling and normals are
  position-based and work unchanged. test-wp18 in test-all (-> 18
  files): closed cube -> watertight DOUBLE shell with positive signed
  volume (orientation), PUNCTURED cube -> STILL watertight (the
  crack-swallowing property), unprojected shell sits at ~eps,
  projection pulls to < 0.16 eps, determinism, empty input. Golden
  @512 (outputs/shoe_512_remesh.glb — separate artifact, the v7 sew
  final is kept for visual A/B): 975 708 V / 1 952 036 F, ZERO
  boundary edges, 305 non-manifold DC-junction edges (known
  ambiguous-cell artifact, invisible), positive signed volume, unit
  normals, COLOR_0 2.4e-7, trimesh clean, 256 s total (remesh ~10 s);
  OBJ/npz byte-identical. Tuning knobs for the next iteration: band,
  project_back, DC resolution, post-remesh decimation (still out of
  scope). The user now has TWO candidate GLBs to compare visually.
- WP0 (bit-exact golden outputs) still needs a CUDA box, but a
  same-seed trellis-mac MPS run works as a structural reference.
- Also feasible locally: the optional leftover perf queue in
  docs/benchmarks/RESULTS.md — the WP10 performance target itself is met.

**Handover:** docs/08_HANDOVER.md has the full state + how-to-continue.

All per approved plan. Original untouched.
