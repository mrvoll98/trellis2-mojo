# File Inventory - trellis2/

This is a mapping of the Python source.

## trellis2/ (main)

(Full list from copy, ~80+ .py)

From exploration:
- trellis2/__init__.py (exports models, modules, pipelines, renderers, representations, utils)
- pipelines/: base.py, trellis2_image_to_3d.py, trellis2_texturing.py, samplers/ (base, flow_euler, mixins), rembg/BiRefNet.py
- models/: __init__.py (from_pretrained with safetensors + json), sparse_structure_*.py, structured_latent_flow.py, sparse_elastic_mixin.py, sc_vaes/ (fdg_vae.py, sparse_unet_vae.py)
- modules/: sparse/ (basic.py - SparseTensor core, config.py, linear, norm, nonlinearity, conv/* (flex_gemm, none, etc), attention/* (full, windowed, rope, modules), spatial/*, transformer/* ), attention/, transformer/, image_feature_extractor.py, norm.py, utils.py, spatial.py
- representations/: mesh/base.py, voxel/voxel_model.py
- renderers/: mesh_renderer.py, pbr_mesh_renderer.py, voxel_renderer.py
- utils/: general_utils.py, random_utils.py, etc.
- datasets/, trainers/ (training mostly, lower priority)
- And more...

See python_files_trellis2.md for detailed.
