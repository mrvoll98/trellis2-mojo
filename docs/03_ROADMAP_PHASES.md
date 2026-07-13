# Roadmap Phases

## Phase 0 — Foundation (this plan)
- [x] Copy source to trellis-mojo/
- [ ] Create 40+ .md files (inventory, architecture, mapping, decisions)
- [ ] Baseline capture (run original example.py, save outputs)
- [ ] README_MOJO.md , MOJO_STATUS.md

## Phase 1 — Project Setup
- Mojo toolchain in env.
- Python package with Mojo interop (mojo build or python bindings).
- Skeleton for trellis2_mojo or mixed.
- Simple interop test (Mojo "hello tensor").
- Reproduce original example.py for golden outputs.

## Phase 2 — Data & Primitives
- Mojo SparseTensor / VarLenTensor.
- Reuse/adapt: manual_cast, str_to_dtype, modulate from modules/utils.py
- RoPE, norms, nonlin, linear, flow_euler math.

## Phase 3 — Attention & Spatial
- Full & windowed attention kernels.
- Spatial down/up, channel<->spatial.

## Phase 4 — Conv & Blocks
- Pure "none" conv fallback in Mojo.
- Sparse FFN + modulated transformer blocks.

## Phase 5 — Models + Samplers
- SLatFlowModel, Elastic, SS flow (dense first).
- FlowEuler + CFG.

## Phase 6 — Pipelines & End-to-End
- Trellis2ImageTo3DPipeline.
- Verification vs original.

## Phase 7 — Output
- Representations thin.
- o_voxel FFI for postprocess.to_glb (keep as is).

## Phase 8 — Polish
- Tests, apps, benchmarks.
