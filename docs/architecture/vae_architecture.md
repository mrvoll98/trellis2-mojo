# Architecture: VAE

- Sparse Structure VAE (dense conv3d for low res occupancy)
- SC-VAE (Sparse Content VAE): SparseUnet or FlexiDualGrid for SLAT (32ch feats + coords)

Used for encode/decode in training and inference (decoder in pipeline).
