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
(work packages), `MOJO_STATUS.md` (English summary). The `trellis2/`
tree is the untouched original — the parity reference. The upstream
README is preserved as `docs/README_upstream_trellis2.md`.

## Setup

Everything is pinned in `pixi.toml` (Mojo 1.0.0b2, torch 2.12):

```
pixi run test-all      # full parity suite (18 test files, no HF cache needed)
pixi run bench         # Mojo vs torch benchmarks (docs/benchmarks/RESULTS.md)
```

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

## License and attribution

- Upstream [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
  (Microsoft, MIT — see `LICENSE`): the `trellis2/` reference tree,
  assets, configs and the original tooling; the Mojo port mirrors its
  inference semantics.
- [CuMesh](https://github.com/JeffreyXiang/CuMesh) (MIT): the mesh
  post-processing and remeshing algorithms were re-implemented in Mojo
  from its published sources (read as reference only — no CUDA code is
  used or shipped).
- The Mojo port itself is released under the same MIT license.
