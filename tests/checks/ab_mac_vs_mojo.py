# A/B (handoverens neste-steg 1 for prikk-saken): sammenlign rand-/
# fragmentstrukturen i de RÅ FDG-meshene — vår GPU-golden mot trellis-mac
# (oppstrøms MPS-port), samme seed 42. Har oppstrøms-meshen samme åpne
# søm-stier, non-manifold kanter og mikro-fragmenter, er prikkene
# modellens natur (jf. GitHub-issue #105), ikke en porteringsartefakt.
#
# Kjør fra repo-rot med trellis-mac-venvets python (kun numpy trengs):
#   ../trellis-mac/.venv/bin/python tests/checks/ab_mac_vs_mojo.py
import numpy as np

MESHES = [
    ("mojo-gpu", "outputs/shoe_3q_mojo_gpu_seed42.obj"),
    ("trellis-mac", "outputs/shoe_3q_mac_seed42.obj"),
]
MIN_AREA = 1e-5  # cumesh remove_small_connected_components-terskelen


def load_obj(path):
    vs, fs = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                fs.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(vs, dtype=np.float64), np.asarray(fs, dtype=np.int64)


def find(parent, f):
    while parent[f] != f:
        parent[f] = parent[parent[f]]
        f = parent[f]
    return f


def stats(name, path):
    pos, faces = load_obj(path)
    V, F = len(pos), len(faces)
    print(f"\n=== {name}: {path}")
    print(f"  {V} V / {F} F")

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    face_ids = np.tile(np.arange(F, dtype=np.int64), 3)
    und = np.sort(e, axis=1)
    key = und[:, 0] * V + und[:, 1]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inst_counts = counts[inv]
    n_boundary = int((counts == 1).sum())
    n_nonmani = int((counts > 2).sum())
    print(f"  boundary edges: {n_boundary}  non-manifold edges (count>2): {n_nonmani}")

    # dead-end vertices on the boundary graph (degree 1)
    bedges = e[inst_counts == 1]
    if len(bedges):
        deg = np.bincount(bedges.ravel(), minlength=V)
        n_dead = int(((deg == 1)).sum())
        n_bverts = int((deg > 0).sum())
        print(f"  boundary vertices: {n_bverts}  dead ends (degree 1): {n_dead}")

    # face components across MANIFOLD edges (cumesh semantics)
    order = np.argsort(inv, kind="stable")
    offs = np.zeros(len(uniq) + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    mani = np.where(counts == 2)[0]
    f1 = face_ids[order[offs[mani]]]
    f2 = face_ids[order[offs[mani] + 1]]
    parent = np.arange(F, dtype=np.int64)
    for a, b in zip(f1.tolist(), f2.tolist()):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(parent, f) for f in range(F)], dtype=np.int64)
    _, comp_ids = np.unique(roots, return_inverse=True)
    n_comps = comp_ids.max() + 1

    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    comp_areas = np.bincount(comp_ids, weights=areas)
    comp_faces = np.bincount(comp_ids)
    small = comp_areas < MIN_AREA
    print(f"  face components (manifold-edge adjacency): {n_comps}")
    print(
        f"  small components (area < {MIN_AREA}): {int(small.sum())}"
        f"  ({int(comp_faces[small].sum())} faces, "
        f"{comp_areas[small].sum():.3e} total area of {comp_areas.sum():.4f})"
    )
    big = np.sort(comp_areas)[::-1][:5]
    print(f"  largest component areas: {[f'{a:.4f}' for a in big]}")
    return n_boundary, n_nonmani, int(small.sum())


for name, path in MESHES:
    stats(name, path)
