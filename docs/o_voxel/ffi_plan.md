# o-voxel FFI Plan

In Mojo: use Python interop or FFI to call o_voxel.convert.* etc when needed (mainly final output).

For tensors passed: convert Mojo Tensor <-> torch via numpy or direct.

Keep postprocess.to_glb in Python.
