# o-voxel Current Implementation

Key: separate package with C++/CUDA for mesh <-> O-Voxel (flex dual grid QEF), .vxz IO, serialize (z-order/hilbert), rasterize, postprocess.to_glb (uses cumesh, nvdiffrast, flex_gemm).

Public entrypoints used:
- convert.mesh_to_flexible_dual_grid, flexible_dual_grid_to_mesh
- io.read/write , vxz
- serialize.encode_seq
- postprocess.to_glb
- rasterize.VoxelRenderer

**For port:** Keep FFI. Document wrappers.
