# Conversion: SparseTensor in Mojo

Design:
struct SparseTensor {
  feats: Tensor[DType.float32]
  coords: Tensor[DType.int32]  // N x 4
  // layout, caches...
}

fn replace(...) -> Self { ... }

Interop: from_torch, to_torch for now.

See architecture/sparse_tensor.md and mapping/sparse/basic.md
