# TRELLIS.2 → Mojo: native image→3D inference on Apple Silicon

A [Mojo](https://www.modular.com/mojo) inference port of Microsoft's
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image→3D pipeline.
The production path is Mojo from end to end: PAM decoding and preprocessing,
DINOv3 conditioning, random-noise generation, flow-matching DiT sampling,
VAE decoding, FDG mesh extraction, mesh post-processing/remeshing, NPZ
sidecar writing and textured GLB export.

There is no Python tensor bridge or PyTorch dependency. Model weights load
directly from safetensors into native `StateDict` values; a project-owned
xorshift64*/Box–Muller generator provides deterministic standard-normal
noise; and the NPZ writer emits ZIP/NumPy payloads directly from Mojo.

Highlights (M4 Pro, macOS arm64):

- **Metal GPU offload** (`TRELLIS2_GPU=1`, CPU fallback default): a
  hand-built tiled-GEMM/SDPA/sparse-conv stack with model-level device
  residency — the full 12-step 512 run went from 27.4 min (CPU) to
  **~4 min**, geometrically the same mesh. `TRELLIS2_GPU_F16=1` adds
  f16-shared-tile kernels for another ~20 % (13 borderline voxels of
  514k vs the f32 golden).
- **Watertight remeshing** (`--remesh`): narrow-band dual contouring on
  the offset surface UDF−ε with projection back to the original mesh —
  a port of cumesh's remesh branch. Zero boundary edges by
  construction; `--remesh-res` trades triangle count (~N²) against
  detail (512 → ~2M triangles, 256 → ~480k, 128 → ~115k).
- **Native regression suite**: runtime/RNG/NPZ, sparse tensor, mesh and
  Metal CPU-vs-GPU checks run with `pixi run test-all`.
- Six empirically probed Metal/Mojo pitfalls (binding limits,
  commit semantics, write-combined memory, 4 GiB binding offsets, …)
  are documented in the `trellis2_mojo/gpu/` file headers, and eight
  upstream bugs found during porting are listed in
  `docs/08_HANDOVER.md`.

Project journal: `docs/08_HANDOVER.md` (state + history),
`docs/07_PORT_TRACKER.md` (file-by-file), `docs/06_MASTER_PLAN.md`
(work packages), `docs/MOJO_NIGHTLY_MIGRATION.md` (Mojo 1.0 migration),
`docs/PURE_MOJO_RUNTIME.md` (framework-removal details), and
`MOJO_STATUS.md` (English summary). Older planning and comparison documents
are retained as an implementation journal; their removed commands and
dependencies are not part of the current runtime.

## Setup

The environment pins stable Mojo 1.0.0 plus the minimal MAX Core 26.5.0 module
package required by `max.algorithm` and `max.gpu`. It does not use the broad
`modular` meta-package. Install it and run the native suite from the repository
root:

```
pixi install
pixi run mojo --version     # Mojo 1.0.0 (ed45d567)
pixi run test-all           # eight native Mojo tasks
```

Individual tasks are `test-runtime`, `test-sparse`, `test-wp16`, `test-wp18`,
`test-simplify`, `test-wp11`, `test-wp11-attn`, and `test-wp11-conv`. The
WP11 checks need the Metal GPU. Historical comparison results live in
`docs/benchmarks/RESULTS.md`; remaining Mojo-only microbenchmarks can be run
directly, for example:

```
pixi run mojo run -I . benchmarks/microbench_linear.mojo
```

The end-to-end runner needs the local Hugging Face cache with
microsoft/TRELLIS.2-4B, microsoft/TRELLIS-image-large and
facebook/dinov3-vitl16-pretrain-lvd1689m. Checkpoint loading itself is native
Mojo and reads 10+ GB over a complete run.

## Running image → 3D

Input must be a **PAM P7** file, RGBA with a real alpha channel (object
already cut out — the rembg path is intentionally unsupported).
Convert an RGBA PNG with ImageMagick and verify that the PAM header reports
`DEPTH 4` and `TUPLTYPE RGB_ALPHA`:

```
magick input.png -alpha on PAM:input.pam
```

Then, from the repo root:

```
TRELLIS2_GPU=1 TRELLIS2_GPU_F16=1 pixi run e2e -- input.pam \
    [--seed N] [--steps N] [--out prefix] [--no-tex] \
    [--pipeline 512|1024] \
    [--remesh] [--remesh-res N] [--remesh-band B] [--remesh-project P] \
    [--simplify-faces N]
```

- Defaults: seed 42, 12 steps, prefix from the image name.
- `--pipeline 1024` runs the upstream `1024_cascade` (their default):
  the shape slat cascades 512 → 1024 through the decoder's subdivision
  upsample; everything downstream runs at 1024³ (~4x the vertices).
- `--remesh` swaps the GLB post-processing (hole filling, non-manifold
  repair, fragment removal, seam sewing, orientation unify) for the
  dual-contouring remesh — a guaranteed-watertight rebuild.
  `--remesh-res 256` is a good size/quality trade-off for viewing.
- `--simplify-faces N` runs native QEM decimation after `--remesh`, while
  the mesh is still in world coordinates and before PBR sampling/normals.
  It targets at most `N` triangles without lowering the remesh grid; a
  conflict-free collapse round may land slightly below the requested count.
- Outputs: `<prefix>.obj` (raw mesh), `<prefix>_texvoxels.npz` (full
  per-voxel PBR payload) and `<prefix>.glb` (directly viewable:
  per-vertex base color + alpha in COLOR_0, trilinearly sampled from
  the voxel volume; global metallic/roughness factors).

Timings on an M4 Pro for the 512 pipeline (12 steps + texture +
remesh): ~4.2 min CPU-only → **~3.5 min** with `TRELLIS2_GPU=1
TRELLIS2_GPU_F16=1` → **~2.8 min** with `--steps 8`. The full CPU-only
run without GPU flags is ~25–30 min; `--steps 2 --no-tex` is a smoke
run. Peak RSS ~11–12 GB (512) / ~18 GB (1024).

Numeric variants (GPU, f16, flash attention) flip borderline
thresholds, so compare runs structurally (bbox, nearest-neighbor
distances), never exact V/F counts. Same seed + same settings is
bit-reproducible end to end with the native RNG. Seeds are not stream-compatible
with releases that used an external framework's generator.

## Example output

A complete 12-step 512 run is checked in at
[`examples/shoe_512_native_rng_qem200k_band2.glb`](examples/shoe_512_native_rng_qem200k_band2.glb),
with its console output in
[`examples/shoe_512_native_rng_qem200k_band2.log`](examples/shoe_512_native_rng_qem200k_band2.log).
It was generated on an M4 Pro with Metal/f16, seed 42, remesh resolution 512,
remesh band 2.0, projection 0.9 and a 200,000-face QEM target. The final GLB
contains 96,042 vertices and 192,302 triangles and completed in 226 seconds.
Band 2.0 was selected after a visual and topological A/B against band 1.0: it
removed the small isolated shells and visible pinholes while preserving the
shoe opening and lace detail. Raw OBJ and texture-voxel NPZ outputs remain
excluded because they are large reproducible intermediates.

## License and acknowledgements

The Mojo port is released under the MIT license (`LICENSE`). Every
upstream it builds on is MIT (or Apache-2.0) — audited before
publication:

- **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft
  Research** (MIT) — the original model and codebase. The port mirrors
  its inference semantics; its source and framework runtime are not vendored.
- **[DINOv3](https://github.com/facebookresearch/dinov3) by Meta** —
  image conditioning. The pure-Mojo reimplementation mirrors the
  Hugging Face
  [transformers](https://github.com/huggingface/transformers)
  implementation (Apache-2.0). The weights
  ([facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m))
  are downloaded from Hugging Face by the user and carry the
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
  (Meta's own, not open source).
- **[CuMesh](https://github.com/JeffreyXiang/CuMesh) by Jianfeng
  Xiang** (MIT) — the mesh cleanup (hole filling, non-manifold repair,
  component removal, orientation unify) and narrow-band
  dual-contouring remesh algorithms, re-implemented in Mojo from the
  published sources (read as reference only — no CUDA code is used or
  shipped).
- **[trellis-mac](https://github.com/shivampkumar/trellis-mac) by
  @shivampkumar** (MIT) — the working MPS port used as the structural
  A/B reference during the port.
- **[@pedronaugusto](https://github.com/pedronaugusto)** — the
  mtlmesh / mtlbvh / mtldiffrast Metal ports (MIT) whose bundled
  CuMesh sources were the readable reference for the remeshing and
  cleanup semantics.
- **[RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) (BRIA AI) is
  deliberately NOT used**: its license (bria-rmbg-2.0, see the model
  card) is non-commercial, so the runner requires pre-cut RGBA input
  instead of shipping background removal.

Model weights (microsoft/TRELLIS.2-4B, microsoft/TRELLIS-image-large,
facebook/dinov3-vitl16) are not distributed with this repo — they are
fetched from Hugging Face by the user under their respective licenses.
