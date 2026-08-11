# Example output

`shoe_512_native_rng_qem200k_band2.glb` is a complete textured output from
the native Mojo pipeline. No Python tensor bridge or external framework was
used for inference, mesh processing or export.

Generation settings:

```text
pipeline:          512
seed:              42
sampling steps:    12
GPU:               Metal
weight GEMM:       f16
remesh resolution: 512
remesh band:       2.0
remesh projection: 0.9
QEM target:        200000 faces
```

Result:

```text
raw mesh:          561745 vertices / 1156612 faces
remeshed:          1023080 vertices / 2046900 faces
final GLB:         96042 vertices / 192302 faces
boundary edges:    0
degenerate faces:  0
used components:   1
non-manifold edges: 116
total runtime:     226 seconds
```

The band-2 result was selected after an A/B comparison with band 1. It
removed four small isolated surface components and the visible pinholes under
the laces while retaining the main shoe opening and lace detail. The GLB is
5.9 MB; the reproducible 45 MB raw OBJ and 19 MB texture-voxel NPZ are not
versioned.

SHA-256:

```text
3321d45041ca246d83bc96842d8284e4d753bf6fb16cc70599d6bf686b9ebc6d  shoe_512_native_rng_qem200k_band2.glb
```
