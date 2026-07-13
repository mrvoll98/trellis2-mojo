# TRELLIS.2 Mojo Port - Documentation Index

**Project:** trellis-mojo (port of trellis.2 / TRELLIS.2 to Mojo)

**Phase 1 Status:** Copy complete. Many .md files for mapping created.

See README_MOJO.md for overview.

## Core Documents
- [01_EXECUTIVE_SUMMARY.md](01_EXECUTIVE_SUMMARY.md)
- [02_CONVERSION_STRATEGY.md](02_CONVERSION_STRATEGY.md)
- [03_ROADMAP_PHASES.md](03_ROADMAP_PHASES.md)
- [04_RISKS_TRADEOFFS.md](04_RISKS_TRADEOFFS.md)
- [05_VERIFICATION_PLAN.md](05_VERIFICATION_PLAN.md)
- **[06_MASTER_PLAN.md](06_MASTER_PLAN.md) — operativ planlegger: arbeidspakker WP0–WP10 med avhengigheter og akseptkriterier**
- **[07_PORT_TRACKER.md](07_PORT_TRACKER.md) — fil-for-fil fremdriftsstatus (sannhetskilden)**
- **[08_HANDOVER.md](08_HANDOVER.md) — handover: tilstand per WP7, Mojo-syntaksnotater, hvordan fortsette med WP8**

## Architecture
- [architecture/00_overview.md](architecture/00_overview.md)
- [architecture/sparse_tensor.md](architecture/sparse_tensor.md)
- [architecture/dit_flow_models.md](architecture/dit_flow_models.md)
- [architecture/vae_architecture.md](architecture/vae_architecture.md)
- [architecture/pipelines_and_samplers.md](architecture/pipelines_and_samplers.md)
- [architecture/o_voxel_representation.md](architecture/o_voxel_representation.md)
- [architecture/dataflow_inference.md](architecture/dataflow_inference.md)

## Inventory (Kartlegging)
- [inventory/00_file_list.md](inventory/00_file_list.md)
- [inventory/python_files_trellis2.md](inventory/python_files_trellis2.md)
- [inventory/o_voxel_files.md](inventory/o_voxel_files.md)
- [inventory/configs_summary.md](inventory/configs_summary.md)
- [inventory/external_deps.md](inventory/external_deps.md)

## Mapping
- [mapping/00_mapping_overview.md](mapping/00_mapping_overview.md)
- **sparse/**
  - [mapping/sparse/00_sparse_overview.md](mapping/sparse/00_sparse_overview.md)
  - [mapping/sparse/basic.md](mapping/sparse/basic.md)
  - [mapping/sparse/conv_backends.md](mapping/sparse/conv_backends.md)
  - [mapping/sparse/attention_full.md](mapping/sparse/attention_full.md)
  - [mapping/sparse/attention_windowed.md](mapping/sparse/attention_windowed.md)
  - [mapping/sparse/spatial.md](mapping/sparse/spatial.md)
  - [mapping/sparse/transformer_blocks.md](mapping/sparse/transformer_blocks.md)
  - [mapping/sparse/norms_linear.md](mapping/sparse/norms_linear.md)
  - [mapping/sparse/nonlinearity.md](mapping/sparse/nonlinearity.md)
- **models/**
  - [mapping/models/00_models_overview.md](mapping/models/00_models_overview.md)
  - [mapping/models/sparse_structure_flow.md](mapping/models/sparse_structure_flow.md)
  - [mapping/models/structured_latent_flow.md](mapping/models/structured_latent_flow.md)
  - [mapping/models/elastic_mixin.md](mapping/models/elastic_mixin.md)
  - [mapping/models/sc_vaes.md](mapping/models/sc_vaes.md)
  - [mapping/models/sparse_structure_vae.md](mapping/models/sparse_structure_vae.md)
- **pipelines/**
  - [mapping/pipelines/00_pipelines_overview.md](mapping/pipelines/00_pipelines_overview.md)
  - [mapping/pipelines/image_to_3d.md](mapping/pipelines/image_to_3d.md)
  - [mapping/pipelines/texturing.md](mapping/pipelines/texturing.md)
  - [mapping/pipelines/base.md](mapping/pipelines/base.md)
  - **samplers/**
    - [mapping/pipelines/samplers/flow_euler.md](mapping/pipelines/samplers/flow_euler.md)
    - [mapping/pipelines/samplers/mixins.md](mapping/pipelines/samplers/mixins.md)
- **modules/**
  - [mapping/modules/attention.md](mapping/modules/attention.md)
  - [mapping/modules/transformer.md](mapping/modules/transformer.md)
  - [mapping/modules/utils.md](mapping/modules/utils.md)
- [mapping/representations_render.md](mapping/representations_render.md)

## Conversion
- [conversion/00_getting_started_mojo.md](conversion/00_getting_started_mojo.md)
- [conversion/phase0_copy_and_docs.md](conversion/phase0_copy_and_docs.md)
- [conversion/phase1_setup_interop.md](conversion/phase1_setup_interop.md)
- [conversion/phase2_utils_and_data.md](conversion/phase2_utils_and_data.md)
- [conversion/phase3_primitives.md](conversion/phase3_primitives.md)
- [conversion/phase4_kernels.md](conversion/phase4_kernels.md)
- [conversion/phase5_blocks_models.md](conversion/phase5_blocks_models.md)
- [conversion/phase6_pipelines.md](conversion/phase6_pipelines.md)
- [conversion/phase7_ovoxel_and_output.md](conversion/phase7_ovoxel_and_output.md)
- [conversion/phase8_apps_tests.md](conversion/phase8_apps_tests.md)
- [conversion/mojo_idioms_mapping.md](conversion/mojo_idioms_mapping.md)
- [conversion/safetensors_loading.md](conversion/safetensors_loading.md)
- [conversion/sparse_tensor_in_mojo.md](conversion/sparse_tensor_in_mojo.md)
- [conversion/numerical_parity.md](conversion/numerical_parity.md)
- [conversion/backend_selection.md](conversion/backend_selection.md)

## Decisions (ADRs)
- [decisions/0001-hybrid-vs-full-port.md](decisions/0001-hybrid-vs-full-port.md)
- [decisions/0002-start-with-inference.md](decisions/0002-start-with-inference.md)
- [decisions/0003-sparse-tensor-design.md](decisions/0003-sparse-tensor-design.md)
- [decisions/0004-keep-ovoxel-ffi.md](decisions/0004-keep-ovoxel-ffi.md)
- [decisions/0005-first-ports.md](decisions/0005-first-ports.md)
- [decisions/0007-pure-mojo-inference.md](decisions/0007-pure-mojo-inference.md)

## o-voxel
- [o_voxel/current_impl.md](o_voxel/current_impl.md)
- [o_voxel/port_options.md](o_voxel/port_options.md)
- [o_voxel/ffi_plan.md](o_voxel/ffi_plan.md)

## Benchmarks & References
- [benchmarks/target_metrics.md](benchmarks/target_metrics.md)
- [benchmarks/comparison_plan.md](benchmarks/comparison_plan.md)
- [references/original_readme.md](references/original_readme.md)
- [references/paper_notes.md](references/paper_notes.md)

## Root
- [../README_MOJO.md](../README_MOJO.md)
- [../MOJO_STATUS.md](../MOJO_STATUS.md)

**Status:** Phase 1 in progress - many MD files for kartlegging created to start the process.
