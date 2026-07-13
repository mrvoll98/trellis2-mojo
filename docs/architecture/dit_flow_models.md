# Architecture: DiT Flow Models

DiT-style (Diffusion Transformer) for flow matching on latents.

- Timestep embed
- adaLN modulation (shift/scale/gate)
- Cross attention to image cond (DINO features)
- RoPE or APE
- Sparse or dense blocks

See mapping/models/*.md
