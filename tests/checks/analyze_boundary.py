# Measure boundary loops (holes) in the golden mesh: edges referenced by
# exactly one triangle, chained into loops, with perimeter stats vs
# upstream's fill_holes(max_hole_perimeter=3e-2) threshold.
from collections import defaultdict

import numpy as np

from glb_reader import read_glb

doc, pos, nrm, col, idx = read_glb("outputs/shoe_512_final.glb")
faces = idx.reshape(-1, 3).astype(np.int64)
V = pos.shape[0]
print(f"mesh: {V} V, {faces.shape[0]} F")

# edge -> count (undirected), and directed boundary edges
e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
und = np.sort(e, axis=1)
key = und[:, 0] * V + und[:, 1]
uniq, counts = np.unique(key, return_counts=True)
n_boundary = int((counts == 1).sum())
n_nonmani = int((counts > 2).sum())
print(f"boundary edges: {n_boundary}  non-manifold edges (count>2): {n_nonmani}")

# directed boundary edges: keep the original direction of edges whose
# undirected key is unique
bkey_set = set(uniq[counts == 1].tolist())
mask = np.array([k in bkey_set for k in key.tolist()])
bedges = e[mask]
print(f"directed boundary edges: {bedges.shape[0]}")

# chain into loops: next[a] = b for boundary edge (a, b)
nxt = {}
multi_start = 0
for a, b in bedges.tolist():
    if a in nxt:
        multi_start += 1
    nxt[a] = b
print(f"vertices with >1 outgoing boundary edge (overwritten): {multi_start}")

visited = set()
loops = []
for start in list(nxt.keys()):
    if start in visited:
        continue
    loop = [start]
    visited.add(start)
    cur = nxt.get(start)
    ok = False
    while cur is not None and len(loop) < 10000:
        if cur == start:
            ok = True
            break
        if cur in visited:
            break
        loop.append(cur)
        visited.add(cur)
        cur = nxt.get(cur)
    if ok:
        loops.append(loop)

sizes = np.array([len(l) for l in loops])
perims = []
for l in loops:
    p = 0.0
    for i in range(len(l)):
        a, b = l[i], l[(i + 1) % len(l)]
        p += float(np.linalg.norm(pos[a] - pos[b]))
    perims.append(p)
perims = np.array(perims)
print(f"closed boundary loops: {len(loops)}")
if len(loops):
    print(f"loop sizes: min {sizes.min()} p50 {int(np.median(sizes))} p95 {int(np.percentile(sizes,95))} max {sizes.max()}")
    print(f"perimeters: min {perims.min():.5f} p50 {np.median(perims):.5f} p95 {np.percentile(perims,95):.5f} max {perims.max():.5f}")
    thr = 3e-2
    print(f"loops with perimeter < {thr} (upstream fill threshold): {(perims < thr).sum()} ({100*(perims < thr).mean():.1f}%)")
    print(f"size histogram (<=12): {[int((sizes==k).sum()) for k in range(3,13)]}")
