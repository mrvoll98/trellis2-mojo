# Risks and Tradeoffs

- Complexity of SparseTensor + caches: Design clean in Mojo first.
- Numerical parity: Target <1e-3 relative, use fp32 initially.
- o-voxel: Keep FFI, no native port.
- Backends: Start with pure, add accel later.
- Effort: Large, focus MVP inference.
- Mac/MPS: Leverage Mojo where possible, fallbacks.
