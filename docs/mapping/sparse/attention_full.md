# Mapping: sparse/attention/full_attn.py

Dispatches scaled dot product for varlen (qkv packed or separate).

Backends via config: flash_attn, xformers, sdpa, naive.

**Mojo:** Implement core or interop; start with vectorized version.
