# Conversion Strategy

**Recommended: Hybrid progressive replacement**

- Keep high-level Python (pipelines, from_pretrained using safetensors, device mgmt, low_vram).
- Port core to Mojo: SparseTensor, basic ops, RoPE, norms, linear/nonlin, attention (full/windowed), spatial, transformer blocks, flow samplers, then full models (start with dense SS flow).
- Use Python interop for weights, external models (DINO, BiRefNet), o_voxel.
- Start with pure "none" backend logic for sparse conv/attn (port conv_none.py style).
- Verify parity at each layer.

**Why hybrid?**
- Full pure-Mojo too slow to validate.
- Allows incremental: run original, swap components.
- Mojo good for perf critical loops (kernels, no Python overhead in sampling steps).
