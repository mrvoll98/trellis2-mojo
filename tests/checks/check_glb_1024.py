# WP17 golden check (1024-cascade): same as the 512 check but with a
# SPARSE vectorized lookup (a dense 1024^3 x 6 volume is 26 GB) — packed
# coord keys + searchsorted instead of dense indexing.
import sys

import numpy as np

sys.path.insert(0, "tests/parity")  # kjør fra repo-rot
from torch_ref_wp15 import read_glb  # noqa: E402

GLB = "outputs/shoe_1024_final.glb"
NPZ = "outputs/shoe_1024_final_texvoxels.npz"
GRID = 1024

doc, pos, nrm, col, idx = read_glb(GLB)
d = np.load(NPZ)
coords = d["coords"].astype(np.int64)
attrs = d["attrs"].astype(np.float32)
vs = float(d["voxel_size"])
origin = d["origin"].astype(np.float32)

world = pos.copy()
world[:, 1] = -pos[:, 2]
world[:, 2] = pos[:, 1]
q = (world - origin[None, :]) / np.float32(vs)

keys = coords[:, 0] * GRID * GRID + coords[:, 1] * GRID + coords[:, 2]
order = np.argsort(keys)
skeys = keys[order]

offs = np.array([
    [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
], dtype=np.float32)
acc = np.zeros((q.shape[0], attrs.shape[1]), dtype=np.float32)
wsum = np.zeros((q.shape[0],), dtype=np.float32)
for k in range(8):
    n = (q + offs[k]).astype(np.int32)  # trunc toward zero
    inb = (n >= 0).all(axis=1) & (n < GRID).all(axis=1)
    nk = (
        n[:, 0].astype(np.int64) * GRID * GRID
        + n[:, 1].astype(np.int64) * GRID
        + n[:, 2].astype(np.int64)
    )
    p = np.searchsorted(skeys, nk)
    p_c = np.clip(p, 0, len(skeys) - 1)
    valid = inb & (skeys[p_c] == nk)
    w = np.prod(1 - np.abs(n.astype(np.float32) + 0.5 - q), axis=1).astype(np.float32)
    w = np.where(valid, w, np.float32(0))
    sel = np.where(valid)[0]
    acc[sel] += w[sel, None] * attrs[order[p_c[sel]]]
    wsum += w
ref = acc / np.maximum(wsum, np.float32(1e-12))[:, None]
ref_col = np.clip(
    np.concatenate([ref[:, 0:3], ref[:, 5:6]], axis=1), 0, 1
).astype(np.float32)

cd = np.abs(ref_col - col).max()
print("V:", pos.shape[0], " tris:", idx.shape[0] // 3)
print("COLOR_0 vs numpy-ref max|diff|:", cd)
mat = doc["materials"][0]["pbrMetallicRoughness"]
mm = float(np.clip(ref[:, 3].astype(np.float64).mean(), 0, 1))
rr = float(np.clip(ref[:, 4].astype(np.float64).mean(), 0, 1))
print("metallicFactor glb", mat["metallicFactor"], " ref-mean", mm)
print("roughnessFactor glb", mat["roughnessFactor"], " ref-mean", rr)
assert cd < 1e-5, "COLOR_0 mismatch"
assert abs(mat["metallicFactor"] - mm) < 1e-4
assert abs(mat["roughnessFactor"] - rr) < 1e-4
nl = np.linalg.norm(nrm, axis=1)
print("normal lengths min/max:", nl.min(), nl.max())
assert nl.max() < 1.0001 and nl.min() > 0.999

import trimesh  # noqa: E402

scene = trimesh.load(GLB)
geom = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
print("trimesh loaded:", geom.vertices.shape[0], "V,", geom.faces.shape[0], "F")
print("trimesh bounds:", geom.bounds.tolist())
assert geom.vertices.shape[0] == pos.shape[0]
print("1024 GLB golden check PASSED")
