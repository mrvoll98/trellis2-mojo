# Mapping: sparse/conv/

Backends via config: flex_gemm (default, Metal), spconv, torchsparse, none (pure py torch fallback - good start for Mojo).

**Mojo:** Port conv_none.py logic (hashmap neighbors + scatter gemm) first as pure.

See conversion/backend_selection.md
