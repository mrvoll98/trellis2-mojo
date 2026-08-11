# Mojo 1.0 migration (1.0.0b2 → 1.0.0 stable)

## Current target

The repository now targets the stable Mojo 1.0 release:

```toml
channels = ["https://conda.modular.com/max", "conda-forge"]
mojo = "==1.0.0"
max-core = "==26.5.0"
```

`max-core` is the minimal package providing `max.algorithm` and `max.gpu`;
the repository does not use the broad `modular` meta-package.

Verified 2026-08-11 on macOS arm64:

- `pixi run mojo --version` → `Mojo 1.0.0 (ed45d567)`
- The original 18-task migration suite passed before the external-framework
  harness was retired.
- Its cache-heavy checks established real-model conditioning/forward parity
  and bit-identical loading of eight checkpoints (7,483,411,862 values).
- The current `test-all` contains eight native Mojo regression tasks; see
  [PURE_MOJO_RUNTIME.md](PURE_MOJO_RUNTIME.md).

The nightly below was an intermediate migration step and is retained so the
API-change history remains useful.

## Intermediate nightly pin (historical)

Pin (pixi):

```toml
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
modular = "==26.5.0.dev2026080406"
mojo = "==1.0.0b3.dev2026080406"
```

## Breaking API changes applied in this tree

| Before (b2) | After (1.0 series) |
|---|---|
| `from std.gpu.host import DeviceContext, DeviceBuffer` | `from max.gpu.host import DeviceContext, DeviceBuffer` |
| `from std.gpu.memory import AddressSpace` | `from std.memory import AddressSpace` |
| `from std.algorithm import parallelize` | `from max.algorithm import parallelize` |
| GPU kernel scalar `Int` / `UInt` | `Int32` / `Int64` (DevicePassable); cast with `Int(...)` inside kernel |
| enqueue scalar dims as `Int` | pass `Int32(dim)` |
| All kernel buffers `MutAnyOrigin` | Read-only inputs: `ImmutAnyOrigin`; outputs: `MutAnyOrigin` |
| Host↔kernel origin mismatch | `rebind[Pointer[T, Origin]](ptr)` (builtin; do **not** import from `std.builtin`) |
| `ptr.load[width=W](i)` | `ptr.unsafe_load[width=W](i)` |
| `ptr.store(i, v)` | `ptr.unsafe_store(i, v)` |
| `ptr[i]` (`Pointer`) | preferred: `ptr[unsafe_offset=i]` (still warning if positional) |
| Implicit copy of tensors / matrices | Explicit `.copy()` (or `^` transfer) |
| Tuple interior refs multi-use | Bind `var v = t[0].copy()` before reuse (invalidated interior refs) |

## Additional 1.0 numeric fixes

- `Float32 ** 0.5` is unsupported → use `sqrt(...)` (`std.math`)
- GPU kernel launch grids must keep original thread counts (e.g. `bias_rms_*`:
  `grid_dim = ((rows * h + 255) // 256,)`, not `rows` alone)

## Migration verification

| Project | Result |
|---|---|
| `mini-mojo-llm` | `test-all` PASS on the intermediate nightly (no `PYTHONPATH`) |
| `instella-moe-mojo` | all 7 unit tests PASS on the intermediate nightly |
| `trellis-mojo` | Migration suite passed on stable 1.0.0; current native eight-task suite documented separately |

## Additional cleanup (this pass)

- Widespread `ptr[i]` → `ptr[unsafe_offset=i]` on real pointer vars (`unsafe_ptr` /
  kernel params / `stack_allocation`)
- **Do not** use `unsafe_offset=` on `List[T]` indexing (List API differs)
- `comptime InlineArray` tables cannot be materialised at runtime on nightly →
  index via small `def` helpers that keep a local `InlineArray`
- `fn` keyword removed → use `def`
- Repo root discovery is **pure Mojo** (`find_repo_root()` via `std.os` /
  `std.os.path`), with no `PYTHONPATH` requirement.

## Arithmetic footguns from bulk pointer rewrites

Bulk search/replace for pointer APIs accidentally rewrote **scalar arithmetic**:

| Wrong (rewrite) | Correct (conv / im2col) |
|---|---|
| `zh - (p + kh)` | `zh - p + kh` |
| `zh * s - (p + kh)` | `zh * s - p + kh` |
| `py * (p + i)` | `py * p + i` |
| `cc * (p + i)` | `cc * p + i` |

These broke wp8/wp9 (ResBlock3d / SS-VAE via dense `Conv3d`) and wp13
(DINOv3 patch embed OOB). Keep parentheses only when the math needs them.

## transformers 5.x DINOv3 key nesting (historical migration note)

`DINOv3ViTModel.state_dict()` now emits `model.layer.N...` while the
facebook/dinov3 HF safetensors (and `dinov3_from`) use bare `layer.N...`.
`tests/parity/torch_ref_wp13.py` strips the `model.` wrapper prefix and
iterates `m.model.layer` for the extractor-style reference loop.

## Residual cleanup (done)

- `UnsafePointer` → `Pointer` (annotations + `unsafe_from_address` constructors)
  across GPU kernels, safetensors, remesh, and microbench_gpu_gemm
- Remaining pointer arithmetic: `p + n` → `p.unsafe_offset(n)`
- Remaining pointer index: `p[i]` → `p[unsafe_offset=i]`
- `memcpy` already on `unsafe_memcpy` where used

## Residual warnings

- Mojo compiler warnings from the migrated tree were cleaned up on
  2026-08-11. The removed comparison subprocesses are no longer part of the
  environment or test suite.

## Stable-only API delta

Delta from the intermediate 1.0.0b3 nightly pin:

- GPU `barrier` lives in `max.gpu` (not `std.gpu`); `thread_idx`/`block_idx` stay in `std.gpu`.
- `List.unsafe_ptr()` retains a concrete origin. The safetensors conversion
  boundary explicitly rebinds raw input to `ImmutAnyOrigin` and tensor
  output to `MutAnyOrigin`; without this, the cache-independent suite passes
  but all real-checkpoint tasks fail to compile.
