# Native Mojo runtime (2026-08-11)

The repository's executable image-to-3D path now uses Mojo throughout.
Framework interop, framework-backed random generation, comparison harnesses
and their environment packages have been removed.

## Current runtime

- `run_image_to_3d.mojo` parses CLI values with `trellis2_mojo/cli.mojo`.
- `trellis2_mojo/rng.mojo` supplies deterministic standard-normal tensors
  with xorshift64*, splitmix64 seed expansion and Box–Muller conversion.
- Safetensors load directly into the native `StateDict` implementation.
- `trellis2_mojo/io/npz.mojo` writes the texture-voxel sidecar as an
  uncompressed ZIP containing NumPy v1.0 arrays.
- The environment pins `mojo ==1.0.0` and the minimal `max-core ==26.5.0`
  module package required for `max.algorithm`/`max.gpu`. It does not install
  a tensor framework, NumPy, or the broad `modular` meta-package.

## Removed surface

- Python-object tensor bridges and the hybrid sampler adapter.
- Framework reference scripts and cross-implementation parity programs.
- The framework benchmark driver and obsolete pixi tasks.
- The upstream-fetch helper and local ignored reference checkouts.

Historical parity and performance results remain in the documentation because
they record how the port was validated before the harness was retired. They are
not descriptions of the current dependency graph or runnable task set.

## Tests

`pixi run test-all` runs eight native Mojo tasks: runtime/RNG/NPZ, sparse
tensor basics, hole filling, remeshing, QEM simplification, and three Metal
CPU-vs-GPU checks.
The end-to-end executable can also be compile-checked with:

```sh
pixi run mojo build -I . run_image_to_3d.mojo -o /tmp/trellis2_mojo_runner
```

## Seed compatibility

A seed remains deterministic for a fixed build and configuration, but the new
project-owned stream intentionally differs from the stream used by older
framework-backed releases. Existing same-seed geometry should therefore be
treated as a historical baseline rather than an output compatibility promise.
