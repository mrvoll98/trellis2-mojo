# WP16 (fase 2, ADR 0008-tillegg): small-hole filling for the GLB export.
#
# The FDG extraction is non-watertight by construction; upstream fills the
# micro holes in the CUDA postprocess (cumesh.fill_holes(3e-2)). This is a
# pure-Mojo port of cumesh's ACTUAL formulation (read from the published
# source, github.com/JeffreyXiang/CuMesh src/{connectivity,clean_up}.cu —
# the first WP16 version walked boundary chains into clean cycles, which
# silently left every BRAIDED micro-hole cluster at the FDG's non-manifold
# junction vertices open; the user could still see holes):
#
#   1. boundary edges = undirected edges used by exactly one triangle;
#   2. CONNECTED COMPONENTS over boundary edges (union-find on shared
#      vertices — no chaining, no walk heuristics);
#   3. a component is fillable iff NO vertex in it is a dead end (every
#      vertex touches >= 2 boundary edges; junction vertices with > 2 are
#      allowed — this is cumesh's is_bound_conn_comp_loop criterion);
#   4. fill each fillable component with total perimeter < max_perimeter
#      by ONE centroid (mean of the component's EDGE MIDPOINTS, like
#      cumesh's compute_loop_boundary_midpoints + segment mean) and a
#      triangle (b, a, centroid) per directed boundary edge (a, b) — the
#      shared edge runs opposite ways in the old and new triangle, so the
#      fill faces the same way as its surroundings.
#
# Applied to the GLB export only — the OBJ/npz stay raw like upstream's
# MeshWithVoxel payload. Deterministic: edges are collected in face order,
# components are processed in order of their first edge.
#
# The module now holds the full cumesh cleanup set PLUS one own pass,
# chained by the runner in upstream's to_glb order (minus the CUDA-bound
# simplify steps): fill_small_holes -> repair_non_manifold_edges ->
# remove_small_connected_components(1e-5) -> fill_small_holes ->
# sew_boundary_seams (WP16 v7, OWN semantics — not upstream) ->
# fill_small_holes -> unify_face_orientations. Never run repair after
# sew: the corner split would sever the welds again. The CuMesh source
# lives locally in trellis-mac/deps/mtlmesh/src/{clean_up,
# connectivity}.cu (a Metal port of CuMesh) — read it there before
# porting more of the family.

from std.collections import Dict
from std.math import sqrt

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def _find(mut parent: Dict[Int, Int], v: Int) raises -> Int:
    """Union-find root with path compression (iterative)."""
    var r = v
    while parent[r] != r:
        r = parent[r]
    var c = v
    while parent[c] != r:
        var nxt = parent[c]
        parent[c] = r
        c = nxt
    return r


def _find_arr(mut parent: List[Int], v: Int) -> Int:
    """Array union-find root with path compression."""
    var r = v
    while parent[r] != r:
        r = parent[r]
    var c = v
    while parent[c] != r:
        var nxt = parent[c]
        parent[c] = r
        c = nxt
    return r


def repair_non_manifold_edges(
    mut vertices: Tensor[F32], mut faces: IntMatrix
) raises -> Int:
    """Port of cumesh repair_non_manifold_edges (clean_up.cu): every face
    CORNER starts as its own vertex; corners merge (union-find) only
    across MANIFOLD edges (undirected edges used by exactly two faces),
    matched by equal original vertex id. Fans held together only by a
    non-manifold edge (>= 3 incident faces) split into separate manifold
    sheets with duplicated vertices. On a manifold-with-boundary sheet
    every boundary structure is a CLOSED loop, so a fill_small_holes pass
    AFTER this closes the rings that previously read as dead-end paths
    (measured on the 1024 golden: 100% of dead-end vertices sat on a
    non-manifold edge — those are the user-visible leftover holes).
    Returns the new vertex count. Positions are preserved (duplicated
    where sheets split); callers re-derive colors/normals afterwards."""
    var v0 = vertices.shape[0]
    var nf = faces.rows

    # undirected edge -> (count, first instance, second instance);
    # instance = f * 3 + k for the edge (corner k, corner k+1) of face f
    var e_cnt = Dict[Int, Int]()
    var e_i1 = Dict[Int, Int]()
    var e_i2 = Dict[Int, Int]()
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if key in e_cnt:
                e_cnt[key] += 1
                if e_cnt[key] == 2:
                    e_i2[key] = f * 3 + k
            else:
                e_cnt[key] = 1
                e_i1[key] = f * 3 + k
    # corner union-find over manifold edges only
    var parent = List[Int](length=3 * nf, fill=0)
    for i in range(3 * nf):
        parent[i] = i
    for e in e_cnt.items():
        if e.value != 2:
            continue
        var i1 = e_i1[e.key]
        var i2 = e_i2[e.key]
        var f1 = i1 // 3
        var k1 = i1 % 3
        var f2 = i2 // 3
        var k2 = i2 % 3
        var a1 = faces.at(f1, k1)
        var ca1 = f1 * 3 + k1
        var cb1 = f1 * 3 + (k1 + 1) % 3
        var a2 = faces.at(f2, k2)
        var ca2 = f2 * 3 + k2
        var cb2 = f2 * 3 + (k2 + 1) % 3
        # match corners by original vertex id (handles both windings)
        var ra: Int
        var rb: Int
        if a1 == a2:
            ra = _find_arr(parent, ca1)
            rb = _find_arr(parent, ca2)
            if ra != rb:
                parent[rb] = ra
            ra = _find_arr(parent, cb1)
            rb = _find_arr(parent, cb2)
            if ra != rb:
                parent[rb] = ra
        else:
            ra = _find_arr(parent, ca1)
            rb = _find_arr(parent, cb2)
            if ra != rb:
                parent[rb] = ra
            ra = _find_arr(parent, cb1)
            rb = _find_arr(parent, ca2)
            if ra != rb:
                parent[rb] = ra

    # compress corner components -> new vertices (first-seen order keeps
    # the rebuild deterministic); every corner in a component holds the
    # same original vertex id by construction
    var new_id = Dict[Int, Int]()
    var new_pos = List[Float32]()
    var nv = 0
    for f in range(nf):
        for k in range(3):
            var c = f * 3 + k
            var r = _find_arr(parent, c)
            var idx: Int
            if r in new_id:
                idx = new_id[r]
            else:
                idx = nv
                new_id[r] = idx
                var ov = faces.at(f, k)
                new_pos.append(vertices.data[ov * 3 + 0])
                new_pos.append(vertices.data[ov * 3 + 1])
                new_pos.append(vertices.data[ov * 3 + 2])
                nv += 1
            faces.data[f * 3 + k] = Int32(idx)
    vertices.data = new_pos^
    vertices.shape[0] = nv
    return nv


def remove_small_connected_components(
    mut vertices: Tensor[F32],
    mut faces: IntMatrix,
    min_area: Float64,
) raises -> Tuple[Int, Int]:
    """Port of cumesh remove_small_connected_components (clean_up.cu):
    face components are unioned ONLY across manifold edges (undirected
    edges with exactly two incident faces — cumesh's
    get_manifold_face_adjacency; a shared vertex or a non-manifold edge
    does NOT connect), a component's area is the sum of its faces'
    0.5*|cross(v1-v0, v2-v0)|, and every face of a component with area
    < min_area is removed, after which unreferenced vertices are
    compacted away in original order (cumesh's _remove_faces +
    remove_unreferenced_vertices). Intent port: areas accumulate in f64
    (cumesh sums f32 on device), so exact-threshold components could in
    principle land differently than CUDA — which is unavailable anyway.
    Face and vertex order are preserved for the survivors, so the pass
    is deterministic. Returns (components_removed, faces_removed)."""
    var v0 = vertices.shape[0]
    var nf = faces.rows
    if nf == 0:
        return (0, 0)

    # undirected edge -> (count, first two incident faces) — same
    # pattern as unify_face_orientations above
    var e_cnt = Dict[Int, Int]()
    var e_f1 = Dict[Int, Int]()
    var e_f2 = Dict[Int, Int]()
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if key in e_cnt:
                e_cnt[key] += 1
                if e_cnt[key] == 2:
                    e_f2[key] = f
            else:
                e_cnt[key] = 1
                e_f1[key] = f

    # face union-find across manifold edges only (exactly 2 faces)
    var parent = List[Int](length=nf, fill=0)
    for i in range(nf):
        parent[i] = i
    for e in e_cnt.items():
        if e.value != 2:
            continue
        var f1 = e_f1[e.key]
        var f2 = e_f2[e.key]
        if f1 == f2:
            continue
        var r1 = _find_arr(parent, f1)
        var r2 = _find_arr(parent, f2)
        if r1 != r2:
            parent[r2] = r1

    # segmented area sum per component root
    var comp_area = Dict[Int, Float64]()
    for f in range(nf):
        var ia = faces.at(f, 0)
        var ib = faces.at(f, 1)
        var ic = faces.at(f, 2)
        var ux = Float64(vertices.data[ib * 3 + 0]) - Float64(vertices.data[ia * 3 + 0])
        var uy = Float64(vertices.data[ib * 3 + 1]) - Float64(vertices.data[ia * 3 + 1])
        var uz = Float64(vertices.data[ib * 3 + 2]) - Float64(vertices.data[ia * 3 + 2])
        var wx = Float64(vertices.data[ic * 3 + 0]) - Float64(vertices.data[ia * 3 + 0])
        var wy = Float64(vertices.data[ic * 3 + 1]) - Float64(vertices.data[ia * 3 + 1])
        var wz = Float64(vertices.data[ic * 3 + 2]) - Float64(vertices.data[ia * 3 + 2])
        var cx = uy * wz - uz * wy
        var cy = uz * wx - ux * wz
        var cz = ux * wy - uy * wx
        var area = 0.5 * sqrt(cx * cx + cy * cy + cz * cz)
        var r = _find_arr(parent, f)
        if r in comp_area:
            comp_area[r] += area
        else:
            comp_area[r] = area

    var removed_comps = 0
    for e in comp_area.items():
        if e.value < min_area:
            removed_comps += 1
    if removed_comps == 0:
        return (0, 0)

    # compact surviving faces in order (cumesh DeviceSelect::Flagged)
    var new_faces = List[Int32]()
    var kept = 0
    for f in range(nf):
        var r = _find_arr(parent, f)
        if comp_area[r] < min_area:
            continue
        new_faces.append(faces.data[f * 3 + 0])
        new_faces.append(faces.data[f * 3 + 1])
        new_faces.append(faces.data[f * 3 + 2])
        kept += 1
    var removed_faces = nf - kept
    faces.data = new_faces^
    faces.rows = kept

    # remove unreferenced vertices, keeping original order (cumesh's
    # exclusive-sum vertex map in remove_unreferenced_vertices)
    var referenced = List[Bool](length=v0, fill=False)
    for i in range(kept * 3):
        referenced[Int(faces.data[i])] = True
    var vmap = List[Int](length=v0, fill=0)
    var new_pos = List[Float32]()
    var nv = 0
    for v in range(v0):
        if referenced[v]:
            vmap[v] = nv
            new_pos.append(vertices.data[v * 3 + 0])
            new_pos.append(vertices.data[v * 3 + 1])
            new_pos.append(vertices.data[v * 3 + 2])
            nv += 1
    for i in range(kept * 3):
        faces.data[i] = Int32(vmap[Int(faces.data[i])])
    vertices.data = new_pos^
    vertices.shape[0] = nv
    return (removed_comps, removed_faces)


def sew_boundary_seams(
    mut vertices: Tensor[F32], mut faces: IntMatrix
) raises -> Tuple[Int, Int]:
    """Seam sewing (WP16 v7 — OWN semantics, not an upstream port; the
    user chose this over the remesh branch after the 2026-07-12 A/B).
    After all fill passes the remaining boundary edges are closed seam
    rings where two sheets meet with BIT-IDENTICAL duplicate positions
    (the non-manifold corner split duplicates positions exactly, and
    the FDG's self-touching folds emit exact duplicates too). Weld
    every group of position-coincident BOUNDARY vertices into one
    vertex — exact f32-bit equality, NO epsilon (near-coincident
    geometry is real geometry) — drop faces that degenerate, compact
    unreferenced vertices in original order. The crack disappears
    topologically, vertex normals then average across the former seam
    (killing the dark shading line), a fill pass AFTER this closes
    rings that only became closable by the weld, and unify propagates
    winding across it. May recreate non-manifold edges at junctions
    (upstream's raw mesh is non-manifold there anyway; unify skips
    edges with != 2 faces). Returns
    (vertices_merged_away, degenerate_faces_dropped)."""
    var v0 = vertices.shape[0]
    var nf = faces.rows
    if nf == 0:
        return (0, 0)

    # undirected edge counts -> boundary vertex mask
    var cnt = Dict[Int, Int]()
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if key in cnt:
                cnt[key] += 1
            else:
                cnt[key] = 1
    var is_bnd = List[Bool](length=v0, fill=False)
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if cnt[key] == 1:
                is_bnd[a] = True
                is_bnd[b] = True

    # group boundary vertices on EXACT position bits: 63-bit spatial
    # hash from the low 21 bits of each f32 (mantissa tail — the
    # distinguishing bits; no Int overflow) + exact-equality walk of a
    # per-bucket chain. First-seen vertex is the representative, so
    # the weld is deterministic.
    var vmap = List[Int](length=v0, fill=0)
    var nxt = List[Int](length=v0, fill=-1)
    for v in range(v0):
        vmap[v] = v
    var bucket_head = Dict[Int, Int]()
    var merged = 0
    for v in range(v0):
        if not is_bnd[v]:
            continue
        var xb = Int(vertices.data[v * 3 + 0].to_bits[DType.uint32]())
        var yb = Int(vertices.data[v * 3 + 1].to_bits[DType.uint32]())
        var zb = Int(vertices.data[v * 3 + 2].to_bits[DType.uint32]())
        var key = ((xb & 0x1FFFFF) << 42) | ((yb & 0x1FFFFF) << 21) | (zb & 0x1FFFFF)
        if key not in bucket_head:
            bucket_head[key] = v
            continue
        var u = bucket_head[key]
        var last = u
        var found = False
        while True:
            if (
                vertices.data[u * 3 + 0] == vertices.data[v * 3 + 0]
                and vertices.data[u * 3 + 1] == vertices.data[v * 3 + 1]
                and vertices.data[u * 3 + 2] == vertices.data[v * 3 + 2]
            ):
                vmap[v] = u
                merged += 1
                found = True
                break
            last = u
            u = nxt[u]
            if u == -1:
                break
        if not found:
            nxt[last] = v
    if merged == 0:
        return (0, 0)

    # remap faces, dropping degenerates (a sliver can span the seam),
    # then compact unreferenced vertices in original order
    var new_faces = List[Int32]()
    var kept = 0
    var dropped = 0
    for f in range(nf):
        var a = vmap[faces.at(f, 0)]
        var b = vmap[faces.at(f, 1)]
        var c = vmap[faces.at(f, 2)]
        if a == b or b == c or a == c:
            dropped += 1
            continue
        new_faces.append(Int32(a))
        new_faces.append(Int32(b))
        new_faces.append(Int32(c))
        kept += 1
    faces.data = new_faces^
    faces.rows = kept

    var referenced = List[Bool](length=v0, fill=False)
    for i in range(kept * 3):
        referenced[Int(faces.data[i])] = True
    var cmap = List[Int](length=v0, fill=0)
    var new_pos = List[Float32]()
    var nv = 0
    for v in range(v0):
        if referenced[v]:
            cmap[v] = nv
            new_pos.append(vertices.data[v * 3 + 0])
            new_pos.append(vertices.data[v * 3 + 1])
            new_pos.append(vertices.data[v * 3 + 2])
            nv += 1
    for i in range(kept * 3):
        faces.data[i] = Int32(cmap[Int(faces.data[i])])
    vertices.data = new_pos^
    vertices.shape[0] = nv
    return (merged, dropped)


def _find_par(
    mut parent: List[Int], mut par: List[Int], f: Int
) -> Tuple[Int, Int]:
    """Parity union-find root: returns (root, parity of f relative to
    root), path-compressing with accumulated parities."""
    var r = f
    var p = 0
    while parent[r] != r:
        p ^= par[r]
        r = parent[r]
    # second pass: compress
    var c = f
    var pc = 0
    while parent[c] != r:
        var nxt = parent[c]
        var pn = par[c]
        par[c] = p ^ pc
        parent[c] = r
        pc ^= pn
        c = nxt
    return (r, p)


def unify_face_orientations(
    mut faces: IntMatrix,
    vertices: Tensor[F32],
) raises -> Tuple[Int, Int]:
    """Port of cumesh unify_face_orientations (clean_up.cu): parity
    union-find over faces — two faces sharing a manifold edge must
    traverse it in opposite directions (a same-direction pair sets flip
    parity 1); faces with parity 1 relative to their sheet root get
    their winding swapped in place. Motivation: the raw FDG output has
    ~50/50 mixed orientation (measured on the 1024 golden: 995 684
    same-direction manifold pairs -> 6 845 after this) — culled
    backfaces and cancelled vertex normals render exactly like micro
    holes. NOTE deliberately NO global per-sheet in/out decision, like
    cumesh: an occupancy-probe vote was tried and MEASURED useless
    (51% — the FDG surface is folded/double-layered in places, so a
    consistent orientation necessarily shows both signs to any local
    probe; the vote is a zero sum). Viewers get correct rendering from
    doubleSided=true + consistent local winding. Run AFTER
    repair_non_manifold_edges (every edge then has <= 2 faces).
    Returns (faces_flipped, sheets)."""
    var v0 = vertices.shape[0]
    var nf = faces.rows

    # manifold edge -> its two face-edge instances (post-repair: <= 2)
    var e_cnt = Dict[Int, Int]()
    var e_i1 = Dict[Int, Int]()
    var e_i2 = Dict[Int, Int]()
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if key in e_cnt:
                e_cnt[key] += 1
                if e_cnt[key] == 2:
                    e_i2[key] = f * 3 + k
            else:
                e_cnt[key] = 1
                e_i1[key] = f * 3 + k

    var parent = List[Int](length=nf, fill=0)
    var par = List[Int](length=nf, fill=0)
    for i in range(nf):
        parent[i] = i
    for e in e_cnt.items():
        if e.value != 2:
            continue
        var i1 = e_i1[e.key]
        var i2 = e_i2[e.key]
        var f1 = i1 // 3
        var f2 = i2 // 3
        if f1 == f2:
            continue
        # same-direction traversal <=> the two instances start at the
        # same vertex -> relative flip parity 1
        var w: Int
        if faces.at(f1, i1 % 3) == faces.at(f2, i2 % 3):
            w = 1
        else:
            w = 0
        var rp1 = _find_par(parent, par, f1)
        var rp2 = _find_par(parent, par, f2)
        if rp1[0] == rp2[0]:
            continue  # non-orientable conflict: keep the first constraint
        parent[rp2[0]] = rp1[0]
        par[rp2[0]] = rp1[1] ^ rp2[1] ^ w

    # apply: flip faces whose parity relative to their sheet root is 1
    var flipped = 0
    var sheets = Dict[Int, Bool]()
    for f in range(nf):
        var rp = _find_par(parent, par, f)
        sheets[rp[0]] = True
        if rp[1] == 1:
            var tmp = faces.data[f * 3 + 1]
            faces.data[f * 3 + 1] = faces.data[f * 3 + 2]
            faces.data[f * 3 + 2] = tmp
            flipped += 1
    return (flipped, len(sheets))


def fill_small_holes(
    mut vertices: Tensor[F32],
    mut faces: IntMatrix,
    max_perimeter: Float64,
) raises -> Tuple[Int, Int]:
    """Fill boundary components (cumesh semantics — see the header) with
    perimeter < max_perimeter. Mutates vertices/faces in place; returns
    (components_filled, boundary_edges_found)."""
    var v0 = vertices.shape[0]
    var nf = faces.rows

    # undirected edge use counts; boundary = exactly one incident triangle
    var cnt = Dict[Int, Int]()
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if key in cnt:
                cnt[key] += 1
            else:
                cnt[key] = 1

    # directed boundary edges in face order
    var ea = List[Int]()
    var eb = List[Int]()
    for f in range(nf):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v0 + b
            else:
                key = b * v0 + a
            if cnt[key] != 1:
                continue
            ea.append(a)
            eb.append(b)
    var n_edges = len(ea)
    if n_edges == 0:
        return (0, 0)

    # union-find over boundary vertices + per-vertex boundary degree
    var parent = Dict[Int, Int]()
    var degree = Dict[Int, Int]()
    for i in range(n_edges):
        var a = ea[i]
        var b = eb[i]
        if a not in parent:
            parent[a] = a
            degree[a] = 0
        if b not in parent:
            parent[b] = b
            degree[b] = 0
        degree[a] += 1
        degree[b] += 1
        var ra = _find(parent, a)
        var rb = _find(parent, b)
        if ra != rb:
            parent[rb] = ra

    # a component with any dead-end vertex (degree 1) is an open seam
    # path, not a hole — cumesh skips it and so do we
    var bad = Dict[Int, Bool]()
    for e in degree.items():
        if e.value == 1:
            bad[_find(parent, e.key)] = True

    # per-component perimeter, midpoint sums and edge lists (first-seen
    # component order keeps the fill deterministic)
    var comp_order = List[Int]()
    var comp_index = Dict[Int, Int]()
    var comp_perim = List[Float64]()
    var comp_mx = List[Float64]()
    var comp_my = List[Float64]()
    var comp_mz = List[Float64]()
    var comp_edges = List[List[Int]]()
    for i in range(n_edges):
        var r = _find(parent, ea[i])
        if r in bad:
            continue
        var ci: Int
        if r in comp_index:
            ci = comp_index[r]
        else:
            ci = len(comp_order)
            comp_index[r] = ci
            comp_order.append(r)
            comp_perim.append(0)
            comp_mx.append(0)
            comp_my.append(0)
            comp_mz.append(0)
            comp_edges.append(List[Int]())
        var a = ea[i]
        var b = eb[i]
        var ax = Float64(vertices.data[a * 3 + 0])
        var ay = Float64(vertices.data[a * 3 + 1])
        var az = Float64(vertices.data[a * 3 + 2])
        var bx = Float64(vertices.data[b * 3 + 0])
        var by = Float64(vertices.data[b * 3 + 1])
        var bz = Float64(vertices.data[b * 3 + 2])
        var dx = ax - bx
        var dy = ay - by
        var dz = az - bz
        comp_perim[ci] += sqrt(dx * dx + dy * dy + dz * dz)
        comp_mx[ci] += (ax + bx) * 0.5
        comp_my[ci] += (ay + by) * 0.5
        comp_mz[ci] += (az + bz) * 0.5
        comp_edges[ci].append(i)

    var filled = 0
    for ci in range(len(comp_order)):
        if comp_perim[ci] >= max_perimeter:
            continue
        var m = Float64(len(comp_edges[ci]))
        var c_idx = vertices.shape[0]
        vertices.data.append(Float32(comp_mx[ci] / m))
        vertices.data.append(Float32(comp_my[ci] / m))
        vertices.data.append(Float32(comp_mz[ci] / m))
        vertices.shape[0] = c_idx + 1
        for j in range(len(comp_edges[ci])):
            var i = comp_edges[ci][j]
            faces.data.append(Int32(eb[i]))
            faces.data.append(Int32(ea[i]))
            faces.data.append(Int32(c_idx))
            faces.rows += 1
        filled += 1
    return (filled, n_edges)
