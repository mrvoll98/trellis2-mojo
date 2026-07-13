# ADR 0004: Keep o-voxel as FFI

**Decision:** Do not reimplement o_voxel native in initial port.

**Rationale:** Complex QEF/Eigen/CUDA, postprocess pipeline heavy on other natives. Use existing installed package. Wrap if needed for Mojo tensors.

See o_voxel/ffi_plan.md
