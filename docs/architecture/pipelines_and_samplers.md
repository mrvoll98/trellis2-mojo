# Architecture: Pipelines and Samplers

Pipelines orchestrate multi-stage generation:
1. Sparse structure sampling (low res)
2. Shape SLAT (with cascade for high res)
3. Texture SLAT

Samplers: FlowEuler (with CFG, guidance interval).

Low VRAM: model.to/cpu() between stages.
