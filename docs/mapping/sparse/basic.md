# Mapping: modules/sparse/basic.py

**Original:** Defines VarLenTensor and SparseTensor (feats + coords [N,4 batch+x y z], layout slices, caches, backend data).

Key methods: replace, to_dense, cat/unbind helpers, seqlen/cum_seqlen.

**Mojo plan:** 
- Struct with Tensor feats, coords.
- Owned layout arrays.
- Methods as fn or methods.
- Interop with Python torch tensors for now.

Difficulty: 4/5
Start here in Phase 2.
