# WP15 reference (fase 2, ADR 0008): plain-torch reimplementation of
# flex_gemm.ops.grid_sample.grid_sample_3d_torch's TRILINEAR math with the
# CUDA hashmap replaced by a dict — same truncated (.int()) neighbor
# coordinates (incl. the p < 0.5 duplicate-neighbor quirk, which counts a
# voxel twice), same weight formula prod(1 - |neigh + 0.5 - p|), misses
# weigh 0, renormalization clamp_min(1e-12). Plus area-weighted vertex
# normals and a dependency-free GLB reader for the writer round-trip
# (trimesh is not in the pixi env; the golden check uses trellis-mac's).

import json
import struct

import numpy as np
import torch

_OFFS = torch.tensor([
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
], dtype=torch.float32)


def trilinear_ref(feats, coords, grid_size, query):
    """feats [N, C] f32, coords [N, 3] int, query [L, 3] f32 -> [L, C]."""
    lut = {}
    for i, (x, y, z) in enumerate(coords.tolist()):
        lut[(x, y, z)] = i
    L, C = query.shape[0], feats.shape[1]
    out = torch.zeros(L, C, dtype=torch.float32)
    for q in range(L):
        p = query[q]
        neigh = (p.unsqueeze(0) + _OFFS).int()  # trunc toward zero
        acc = torch.zeros(C, dtype=torch.float32)
        wsum = torch.zeros((), dtype=torch.float32)
        for k in range(8):
            nx, ny, nz = int(neigh[k, 0]), int(neigh[k, 1]), int(neigh[k, 2])
            if nx < 0 or ny < 0 or nz < 0:
                continue
            if nx >= grid_size or ny >= grid_size or nz >= grid_size:
                continue
            i = lut.get((nx, ny, nz))
            if i is None:
                continue
            d = 1 - torch.abs(neigh[k].float() + 0.5 - p)
            w = d[0] * d[1] * d[2]
            wsum = wsum + w
            acc = acc + w * feats[i]
        out[q] = acc / torch.clamp_min(wsum, 1e-12)
    return out


def ref_sample(seed, grid, n, l):
    """Seeded sparse volume + query mix (voxel centers, interior points,
    p < 0.5 boundary points, empty-region points, out-of-grid points)."""
    g = torch.Generator().manual_seed(seed)
    # unique random coords in [0, grid)
    flat = torch.randperm(grid * grid * grid, generator=g)[:n]
    coords = torch.stack([
        flat // (grid * grid), (flat // grid) % grid, flat % grid
    ], dim=1).int()
    feats = torch.randn(n, 6, generator=g, dtype=torch.float32)

    parts = []
    # voxel centers of existing voxels
    parts.append(coords[: min(n, l // 4)].float() + 0.5)
    # random interior points
    parts.append(torch.rand(l // 4, 3, generator=g) * grid)
    # boundary band p < 0.5 (duplicate-trunc quirk)
    parts.append(torch.rand(l // 4, 3, generator=g) * 0.49)
    # outside / far corner band (partial and full misses)
    rest = l - sum(p.shape[0] for p in parts)
    parts.append(torch.rand(rest, 3, generator=g) * 2 + (grid - 1))
    query = torch.cat(parts, dim=0).contiguous()
    out = trilinear_ref(feats, coords, grid, query)
    return {
        "coords": coords, "feats": feats, "query": query, "out": out,
        "grid": grid,
    }


def ref_normals(seed, v, f):
    """Random mesh + area-weighted vertex normals. Mirrors the Mojo
    semantics: EVERY normal ends unit length — cancelled/zero vertices
    get the average of their incident faces' other corners' unit normals
    (from the first pass), then a +z fallback."""
    g = torch.Generator().manual_seed(seed)
    verts = torch.randn(v, 3, generator=g, dtype=torch.float32)
    faces = torch.randint(0, v, (f, 3), generator=g, dtype=torch.int64)
    e1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    e2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    fn = torch.cross(e1, e2, dim=1)
    vn = torch.zeros(v, 3, dtype=torch.float32)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    norm = vn.norm(dim=1, keepdim=True)
    zero = (norm <= 1e-20).squeeze(1)
    vn = torch.where(norm > 1e-20, vn / norm.clamp_min(1e-20), vn)
    if zero.any():
        fix = torch.zeros(v, 3, dtype=torch.float32)
        for k in range(3):
            i = faces[:, k]
            other = vn[faces[:, (k + 1) % 3]] + vn[faces[:, (k + 2) % 3]]
            fix.index_add_(0, i, other * zero[i].unsqueeze(1).float())
        fnorm = fix.norm(dim=1, keepdim=True)
        fixed = torch.where(
            fnorm > 1e-12,
            fix / fnorm.clamp_min(1e-12),
            torch.tensor([0.0, 0.0, 1.0]).expand(v, 3),
        )
        vn = torch.where(zero.unsqueeze(1), fixed, vn)
    return {"verts": verts, "faces": faces.int(), "normals": vn}


def read_glb(path):
    """Dependency-free GLB 2.0 reader for the round-trip check: returns
    (json_dict, positions, normals, colors, indices)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    magic, version, total = struct.unpack_from("<III", blob, 0)
    assert magic == 0x46546C67, "bad magic"
    assert version == 2, "bad version"
    assert total == len(blob), "length mismatch"
    jlen, jtype = struct.unpack_from("<II", blob, 12)
    assert jtype == 0x4E4F534A, "first chunk must be JSON"
    doc = json.loads(blob[20:20 + jlen].decode("utf-8"))
    boff = 20 + jlen
    blen, btype = struct.unpack_from("<II", blob, boff)
    assert btype == 0x004E4942, "second chunk must be BIN"
    binbuf = blob[boff + 8: boff + 8 + blen]
    assert doc["buffers"][0]["byteLength"] == blen

    def acc_array(idx, dtype, comps):
        acc = doc["accessors"][idx]
        view = doc["bufferViews"][acc["bufferView"]]
        off = view.get("byteOffset", 0)
        count = acc["count"]
        arr = np.frombuffer(
            binbuf, dtype=dtype, count=count * comps, offset=off
        )
        return arr.reshape(count, comps) if comps > 1 else arr

    prim = doc["meshes"][0]["primitives"][0]
    pos = acc_array(prim["attributes"]["POSITION"], np.float32, 3)
    nrm = acc_array(prim["attributes"]["NORMAL"], np.float32, 3)
    col = acc_array(prim["attributes"]["COLOR_0"], np.float32, 4)
    idx = acc_array(prim["indices"], np.uint32, 1)
    return doc, pos, nrm, col, idx


def check_glb(path, verts, normals, colors_clamped, faces, metallic, roughness):
    """Round-trip assertions: everything read back must be BIT-identical
    to what the Mojo writer was given (colors after [0,1] clamping).
    Returns the max abs diffs for printing."""
    doc, pos, nrm, col, idx = read_glb(path)
    verts = verts.numpy()
    normals = normals.numpy()
    colors = colors_clamped.numpy()
    faces = faces.numpy().astype(np.uint32).reshape(-1)
    assert pos.shape == verts.shape, "position count mismatch"
    assert (pos == verts).all(), "positions not bit-identical"
    assert (nrm == normals).all(), "normals not bit-identical"
    assert (col == colors).all(), "colors not bit-identical"
    assert (idx == faces).all(), "indices not bit-identical"
    mat = doc["materials"][0]["pbrMetallicRoughness"]
    assert abs(mat["metallicFactor"] - metallic) < 1e-9
    assert abs(mat["roughnessFactor"] - roughness) < 1e-9
    assert doc["materials"][0]["doubleSided"] is True
    # JSON numbers parse as f64 of the shortest f32 repr — cast back to
    # f32 before the exact compare (f32 -> repr -> f64 -> f32 round-trips)
    pos_acc = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    mins = np.array(doc["accessors"][pos_acc]["min"], dtype=np.float32)
    maxs = np.array(doc["accessors"][pos_acc]["max"], dtype=np.float32)
    assert (mins == verts.min(axis=0)).all(), "min mismatch"
    assert (maxs == verts.max(axis=0)).all(), "max mismatch"
    return {"ok": True, "V": int(pos.shape[0]), "I": int(idx.shape[0])}
