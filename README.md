# TRELLIS.2 → Mojo: pure-Mojo image→3D inference on Apple Silicon

A pure-[Mojo](https://www.modular.com/mojo) inference port of Microsoft's
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image→3D pipeline,
built and verified op-by-op against the PyTorch original. The whole
pipeline — PAM decode + preprocess, DINOv3 conditioning, flow-matching
DiT sampling, VAE decoding, FDG mesh extraction, mesh post-processing /
remeshing, textured GLB export — runs in Mojo. The only Python left in
the runner is `torch.randn` (noise-stream compatibility with the
original); Python/torch otherwise remain only in `tests/parity/` and
`benchmarks/` as ground truth.

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
- **Op-for-op parity suite**: 18 test files compare every ported
  component against the seeded PyTorch original (`pixi run test-all`);
  real-checkpoint and conditioning parity run separately against the
  local HF cache.
- Six empirically probed Metal/Mojo-1.0.0b2 pitfalls (binding limits,
  commit semantics, write-combined memory, 4 GiB binding offsets, …)
  are documented in the `trellis2_mojo/gpu/` file headers, and eight
  upstream bugs found during porting are listed in
  `docs/08_HANDOVER.md`.

Project journal: `docs/08_HANDOVER.md` (state + next steps),
`docs/07_PORT_TRACKER.md` (file-by-file), `docs/06_MASTER_PLAN.md`
(work packages), `MOJO_STATUS.md` (English summary). The upstream
README is preserved as `docs/README_upstream_trellis2.md`.

## Setup

Everything is pinned in `pixi.toml` (Mojo 1.0.0b2, torch 2.12).
The upstream reference code is not vendored — the parity tests
compare against the untouched original, fetched into `trellis2/`,
`o-voxel/` and `configs/` by:

```
scripts/fetch_upstream.sh   # clones microsoft/TRELLIS.2 @ the pinned commit
```

Then:

```
pixi run test-all      # full parity suite (18 test files, no HF cache needed)
pixi run bench         # Mojo vs torch benchmarks (docs/benchmarks/RESULTS.md)
```

(The runner itself — `pixi run e2e` — is pure Mojo and does not need
the upstream code.)

Real-checkpoint tests (need the local HF cache with
microsoft/TRELLIS.2-4B, microsoft/TRELLIS-image-large and
facebook/dinov3-vitl16-pretrain-lvd1689m; read 10+ GB per run):
`pixi run test-real`, `pixi run test-cond`, `pixi run test-io`.

## Running image → 3D

Input must be a **PAM P7** file, RGBA with a real alpha channel (object
already cut out — the rembg path is intentionally unsupported).
Convert any RGBA PNG with this one-liner (Pillow cannot write PAM):

```
python3 -c "from PIL import Image; import sys; \
im = Image.open(sys.argv[1]).convert('RGBA'); \
f = open(sys.argv[2], 'wb'); \
f.write(f'P7\nWIDTH {im.width}\nHEIGHT {im.height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n'.encode()); \
f.write(im.tobytes())" input.png input.pam
```

Then, from the repo root:

```
TRELLIS2_GPU=1 TRELLIS2_GPU_F16=1 pixi run e2e -- input.pam \
    [--seed N] [--steps N] [--out prefix] [--no-tex] \
    [--pipeline 512|1024] \
    [--remesh] [--remesh-res N] [--remesh-band B] [--remesh-project P]
```

- Defaults: seed 42, 12 steps, prefix from the image name.
- `--pipeline 1024` runs the upstream `1024_cascade` (their default):
  the shape slat cascades 512 → 1024 through the decoder's subdivision
  upsample; everything downstream runs at 1024³ (~4x the vertices).
- `--remesh` swaps the GLB post-processing (hole filling, non-manifold
  repair, fragment removal, seam sewing, orientation unify) for the
  dual-contouring remesh — a guaranteed-watertight rebuild.
  `--remesh-res 256` is a good size/quality trade-off for viewing.
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
bit-reproducible end to end.

## License and acknowledgements

The Mojo port is released under the MIT license (`LICENSE`). Every
upstream it builds on is MIT (or Apache-2.0) — audited before
publication:

- **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft
  Research** (MIT) — the original model and codebase. The port mirrors
  its inference semantics op for op; the reference code is fetched by
  `scripts/fetch_upstream.sh` for the parity tests, not vendored.
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
  A/B reference throughout, and the origin of the mac-compat patch set
  and the vendored pure-torch mesh-extraction stub in `tests/parity/`.
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
