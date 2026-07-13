# Executive Summary

TRELLIS.2 (trellis.2 package) is a state-of-the-art 3D generative model for image-to-3D using structured latents (SLAT), sparse VAEs, and multi-stage flow-matching DiTs on a custom O-Voxel representation.

**Goal of Mojo port:** Create a high-performance hybrid (Python + Mojo) version focused on inference, leveraging Mojo for tensor ops, SparseTensor, attention, and DiT blocks to improve speed and control, especially on Apple Silicon.

**Primary starting actions (per user request):**
1. Exact copy of source (completed: trellis-mojo/).
2. Create very many .md files for thorough mapping ("kartlegge") before any .mojo code.

**Scope (MVP):**
- Hybrid: Python for orchestration, Mojo for compute kernels and core data structures.
- Inference only initially.
- o-voxel kept as FFI.
- End goal: run example.py producing comparable output.

See full plan and Scope Boundaries section for exact stopping points.
