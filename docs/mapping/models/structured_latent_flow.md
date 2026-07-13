# Mapping: models/structured_latent_flow.py

SLatFlowModel / ElasticSLatFlowModel: DiT on SparseTensor.

Timestep embed, input SparseLinear, ModulatedSparseTransformerCrossBlock x30, out SparseLinear.

**Port:** Start with non-elastic. Port blocks to Mojo.

Uses SparseTensor heavily.
