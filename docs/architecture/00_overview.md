# Architecture Overview

See README.md in root for high level.

Pipeline: image -> cond (DINO) -> SS flow (dense) -> shape SLAT flow (sparse) -> tex SLAT -> decode with VAEs -> o_voxel post to glb.

Core innovation: O-Voxel + sparse structured latent + DiT flows.
