# WP18 (fase 2): narrow-band dual-contouring remesh — port av
# cumesh.remeshing.remesh_narrow_band_dc (trellis-mac/deps/mtlmesh/
# cumesh/remeshing.py + src/remesh/simple_dual_contour.cu, LEST som
# referanse; ingen CUDA i bruk). Oppstrøms' svar på sprekk-/prikk-
# klassen: i stedet for å lappe FDG-topologien ekstraheres OFFSET-
# flaten UDF(p) - eps = 0 (eps = band voxler) fra en USIGNERT
# avstandsfunksjon rundt originalmeshen — alle sprekker/hull smalere
# enn ~2*eps svelges av skallet per konstruksjon, og project_back
# snapper vertekser tilbake mot originalflaten etterpå.
#
# Avvik fra referansen (dokumentert):
# - Oppstrøms bygger båndet coarse-to-fine med BVH-avstander; vi
#   stempler kandidat-voxels direkte fra triangel-AABB-er (superset)
#   og filtrerer eksakt — samme resulterende voxelsett, uten oktre.
# - Avstands-/projeksjonsoppslag bruker et uniformt triangel-grid
#   (R/4-celler, CSR) med 3^3-nabolagssøk og r_max = 4 voxler i
#   stedet for cuBVH — eksakt innenfor r_max, som er alt DC-en og
#   projeksjonen trenger (kryssende kanter har |d| < eps + 1 voxel).
# - FUNNET OPPSTRØMS-BUG (remeshing.py:220-231): split-valget leser
#   kolonne 1,2,3 av 6-indeks-raden i stedet for trekant nr. 2
#   (kolonne 3,4,5) — «align» sammenligner trekant 1 med seg selv
#   (split 1) og med en degenerert null-normal (split 2), så split 1
#   velges ALLTID. Vi porterer INTENSJONEN: |n(tri1) . n(tri2)| per
#   reell splitt, velg størst, split 1 ved likhet (= deres
#   observerbare oppførsel når kvadene er plane).
# - Alt i f64 internt (referansen er f32 på device); intensjonsport,
#   ikke bit-paritet — CUDA finnes ikke på maskinen.
#
# Determinisme: bitset-enumerasjon gir sorterte voxel-/verteksnøkler,
# quad-byggingen går i voxel-så-akse-rekkefølge, kompaktering i
# stigende indeks (speiler torch.unique), og alle parallelle pass
# skriver disjunkte slots.

from std.algorithm import parallelize
from std.math import sqrt

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32
comptime I32 = DType.int32


def _closest_pt_tri(
    px: Float64, py: Float64, pz: Float64,
    ax: Float64, ay: Float64, az: Float64,
    bx: Float64, by: Float64, bz: Float64,
    cx: Float64, cy: Float64, cz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """Closest point on triangle abc to p (Ericson, RTCD 5.1.5)."""
    var abx = bx - ax
    var aby = by - ay
    var abz = bz - az
    var acx = cx - ax
    var acy = cy - ay
    var acz = cz - az
    var apx = px - ax
    var apy = py - ay
    var apz = pz - az
    var d1 = abx * apx + aby * apy + abz * apz
    var d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0 and d2 <= 0:
        return (ax, ay, az)
    var bpx = px - bx
    var bpy = py - by
    var bpz = pz - bz
    var d3 = abx * bpx + aby * bpy + abz * bpz
    var d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0 and d4 <= d3:
        return (bx, by, bz)
    var vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        var v = d1 / (d1 - d3)
        return (ax + v * abx, ay + v * aby, az + v * abz)
    var cpx = px - cx
    var cpy = py - cy
    var cpz = pz - cz
    var d5 = abx * cpx + aby * cpy + abz * cpz
    var d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0 and d5 <= d6:
        return (cx, cy, cz)
    var vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        var w = d2 / (d2 - d6)
        return (ax + w * acx, ay + w * acy, az + w * acz)
    var va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        var w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return (bx + w2 * (cx - bx), by + w2 * (cy - by), bz + w2 * (cz - bz))
    var denom = 1.0 / (va + vb + vc)
    var v3 = vb * denom
    var w3 = vc * denom
    return (
        ax + abx * v3 + acx * w3,
        ay + aby * v3 + acy * w3,
        az + abz * v3 + acz * w3,
    )


def _grid_closest(
    px: Float64, py: Float64, pz: Float64,
    c0: Float64, c1: Float64, c2: Float64,
    scale: Float64, G: Int,
    starts: UnsafePointer[Scalar[I32], _],
    entries: UnsafePointer[Scalar[I32], _],
    vp: UnsafePointer[Scalar[F32], _],
    fp: UnsafePointer[Scalar[I32], _],
    r_max: Float64,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Closest point on the binned mesh within r_max (world units).
    Scans the 3^3 coarse-cell neighborhood of p's cell — complete for
    r_max <= one coarse cell. Returns (dist, qx, qy, qz); dist ==
    r_max means nothing found."""
    var gx = ((px - c0) / scale + 0.5) * Float64(G)
    var gy = ((py - c1) / scale + 0.5) * Float64(G)
    var gz = ((pz - c2) / scale + 0.5) * Float64(G)
    var ix = Int(gx)
    var iy = Int(gy)
    var iz = Int(gz)
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if iz < 0:
        iz = 0
    if ix > G - 1:
        ix = G - 1
    if iy > G - 1:
        iy = G - 1
    if iz > G - 1:
        iz = G - 1
    var best = r_max * r_max
    var qx = px
    var qy = py
    var qz = pz
    for dx in range(-1, 2):
        var x = ix + dx
        if x < 0 or x >= G:
            continue
        for dy in range(-1, 2):
            var y = iy + dy
            if y < 0 or y >= G:
                continue
            for dz in range(-1, 2):
                var z = iz + dz
                if z < 0 or z >= G:
                    continue
                var cell = (x * G + y) * G + z
                var t0 = Int(starts[cell])
                var t1 = Int(starts[cell + 1])
                for e in range(t0, t1):
                    var t = Int(entries[e])
                    var ia = Int(fp[t * 3 + 0])
                    var ib = Int(fp[t * 3 + 1])
                    var ic = Int(fp[t * 3 + 2])
                    var q = _closest_pt_tri(
                        px, py, pz,
                        Float64(vp[ia * 3 + 0]), Float64(vp[ia * 3 + 1]), Float64(vp[ia * 3 + 2]),
                        Float64(vp[ib * 3 + 0]), Float64(vp[ib * 3 + 1]), Float64(vp[ib * 3 + 2]),
                        Float64(vp[ic * 3 + 0]), Float64(vp[ic * 3 + 1]), Float64(vp[ic * 3 + 2]),
                    )
                    var ddx = q[0] - px
                    var ddy = q[1] - py
                    var ddz = q[2] - pz
                    var d2 = ddx * ddx + ddy * ddy + ddz * ddz
                    if d2 < best:
                        best = d2
                        qx = q[0]
                        qy = q[1]
                        qz = q[2]
    if best >= r_max * r_max:
        return (r_max, px, py, pz)
    return (sqrt(best), qx, qy, qz)


def _bsearch(keys: UnsafePointer[Int, _], n: Int, key: Int) -> Int:
    """Index of key in sorted keys, or -1."""
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        var k = keys[mid]
        if k == key:
            return mid
        if k < key:
            lo = mid + 1
        else:
            hi = mid
    return -1


def _bit_set(mut words: List[UInt64], key: Int):
    words[key >> 6] |= UInt64(1) << UInt64(key & 63)


def remesh_narrow_band_dc(
    vertices: Tensor[F32],
    faces: IntMatrix,
    c0: Float64, c1: Float64, c2: Float64,
    scale: Float64,
    resolution: Int,
    band: Float64,
    project_back: Float64,
) raises -> Tuple[Tensor[F32], IntMatrix]:
    """Narrow-band dual-contouring remesh of (vertices, faces) — see
    the module header for semantics and deviations. center (c0,c1,c2)
    and scale define the cubic domain like upstream to_glb: scale is
    PRE-inflated by (resolution + 3*band)/resolution by the caller.
    Returns a brand-new closed mesh with globally consistent winding
    (quad orientation follows the distance-field crossing signs)."""
    var R = resolution
    var nf = faces.rows
    var out_v = Tensor[F32]([0, 3])
    var out_f = IntMatrix(0, 3)
    if nf == 0:
        return (out_v^, out_f^)
    var vp = vertices.data.unsafe_ptr()
    var fp = faces.data.unsafe_ptr()
    var eps = band * scale / Float64(R)
    var cell = scale / Float64(R)
    var r_max = 4.0 * cell

    # --- uniform triangle grid (G^3 CSR, cell = 4 fine voxels) ---
    var G = R // 4
    if G < 1:
        G = 1
    var n_cells = G * G * G
    var counts = List[Int32](length=n_cells + 1, fill=0)
    # per-triangle coarse-cell AABB, two passes (count, fill)
    var tri_lo = List[Int](length=nf * 3, fill=0)
    var tri_hi = List[Int](length=nf * 3, fill=0)
    for t in range(nf):
        for c in range(3):
            var lo = 1.0e30
            var hi = -1.0e30
            for k in range(3):
                var w = Float64(vp[Int(fp[t * 3 + k]) * 3 + c])
                if w < lo:
                    lo = w
                if w > hi:
                    hi = w
            var cc: Float64
            if c == 0:
                cc = c0
            elif c == 1:
                cc = c1
            else:
                cc = c2
            var glo = Int(((lo - cc) / scale + 0.5) * Float64(G))
            var ghi = Int(((hi - cc) / scale + 0.5) * Float64(G))
            if glo < 0:
                glo = 0
            if ghi > G - 1:
                ghi = G - 1
            if glo > G - 1:
                glo = G - 1
            if ghi < 0:
                ghi = 0
            tri_lo[t * 3 + c] = glo
            tri_hi[t * 3 + c] = ghi
    for t in range(nf):
        for x in range(tri_lo[t * 3 + 0], tri_hi[t * 3 + 0] + 1):
            for y in range(tri_lo[t * 3 + 1], tri_hi[t * 3 + 1] + 1):
                for z in range(tri_lo[t * 3 + 2], tri_hi[t * 3 + 2] + 1):
                    counts[(x * G + y) * G + z + 1] += 1
    for i in range(n_cells):
        counts[i + 1] += counts[i]
    var starts = counts.copy()
    var n_entries = Int(counts[n_cells])
    var entries = List[Int32](length=n_entries, fill=0)
    var fill = List[Int32](length=n_cells, fill=0)
    for t in range(nf):
        for x in range(tri_lo[t * 3 + 0], tri_hi[t * 3 + 0] + 1):
            for y in range(tri_lo[t * 3 + 1], tri_hi[t * 3 + 1] + 1):
                for z in range(tri_lo[t * 3 + 2], tri_hi[t * 3 + 2] + 1):
                    var cix = (x * G + y) * G + z
                    entries[Int(starts[cix]) + Int(fill[cix])] = Int32(t)
                    fill[cix] += 1
    var sp = starts.unsafe_ptr()
    var ep = entries.unsafe_ptr()

    # --- candidate voxels: stamp triangle AABBs dilated 3 voxels ---
    var vol = R * R * R
    var vox_bits = List[UInt64](length=(vol + 63) // 64, fill=0)
    for t in range(nf):
        var lo3 = List[Int](length=3, fill=0)
        var hi3 = List[Int](length=3, fill=0)
        for c in range(3):
            var lo = 1.0e30
            var hi = -1.0e30
            for k in range(3):
                var w = Float64(vp[Int(fp[t * 3 + k]) * 3 + c])
                if w < lo:
                    lo = w
                if w > hi:
                    hi = w
            var cc: Float64
            if c == 0:
                cc = c0
            elif c == 1:
                cc = c1
            else:
                cc = c2
            var glo = Int(((lo - cc) / scale + 0.5) * Float64(R)) - 3
            var ghi = Int(((hi - cc) / scale + 0.5) * Float64(R)) + 3
            if glo < 0:
                glo = 0
            if ghi > R - 1:
                ghi = R - 1
            lo3[c] = glo
            hi3[c] = ghi
        for x in range(lo3[0], hi3[0] + 1):
            for y in range(lo3[1], hi3[1] + 1):
                for z in range(lo3[2], hi3[2] + 1):
                    _bit_set(vox_bits, (x * R + y) * R + z)
    var cand = List[Int]()
    for w in range(len(vox_bits)):
        var word = vox_bits[w]
        if word == 0:
            continue
        for b in range(64):
            if (word >> UInt64(b)) & 1 == 1:
                cand.append(w * 64 + b)
    var n_cand = len(cand)

    # --- exact UDF filter: |UDF(center) - eps| < 0.87 * cell ---
    var keep = List[Int32](length=n_cand, fill=0)
    var cp = cand.unsafe_ptr()
    var kp = keep.unsafe_ptr()

    @parameter
    def cand_body(i: Int):
        var key = cp[i]
        var vx = key // (R * R)
        var vy = (key // R) % R
        var vz = key % R
        var wx = ((Float64(vx) + 0.5) / Float64(R) - 0.5) * scale + c0
        var wy = ((Float64(vy) + 0.5) / Float64(R) - 0.5) * scale + c1
        var wz = ((Float64(vz) + 0.5) / Float64(R) - 0.5) * scale + c2
        var q = _grid_closest(wx, wy, wz, c0, c1, c2, scale, G, sp, ep, vp, fp, r_max)
        var d = q[0] - eps
        if d < 0:
            d = -d
        if d < 0.87 * cell:
            kp[i] = 1

    parallelize[cand_body](n_cand)
    var vox_keys = List[Int]()
    for i in range(n_cand):
        if keep[i] == 1:
            vox_keys.append(cand[i])
    var n_vox = len(vox_keys)
    if n_vox == 0:
        return (out_v^, out_f^)

    # --- grid vertices (corners of active voxels), sorted keys ---
    var R1 = R + 1
    var vert_bits = List[UInt64](length=(R1 * R1 * R1 + 63) // 64, fill=0)
    for i in range(n_vox):
        var key = vox_keys[i]
        var vx = key // (R * R)
        var vy = (key // R) % R
        var vz = key % R
        for ox in range(2):
            for oy in range(2):
                for oz in range(2):
                    _bit_set(vert_bits, ((vx + ox) * R1 + vy + oy) * R1 + vz + oz)
    var vert_keys = List[Int]()
    for w in range(len(vert_bits)):
        var word = vert_bits[w]
        if word == 0:
            continue
        for b in range(64):
            if (word >> UInt64(b)) & 1 == 1:
                vert_keys.append(w * 64 + b)
    var n_gvert = len(vert_keys)
    var gvp = vert_keys.unsafe_ptr()

    # signed band value d = UDF - eps at each grid vertex (f32, like
    # the CUDA udf tensor); misses clamp to r_max - eps > 0 — crossing
    # edges always have both endpoints well inside r_max
    var vert_val = List[Float32](length=n_gvert, fill=0)
    var vvp = vert_val.unsafe_ptr()

    @parameter
    def vert_body(i: Int):
        var key = gvp[i]
        var vx = key // (R1 * R1)
        var vy = (key // R1) % R1
        var vz = key % R1
        var wx = (Float64(vx) / Float64(R) - 0.5) * scale + c0
        var wy = (Float64(vy) / Float64(R) - 0.5) * scale + c1
        var wz = (Float64(vz) / Float64(R) - 0.5) * scale + c2
        var q = _grid_closest(wx, wy, wz, c0, c1, c2, scale, G, sp, ep, vp, fp, r_max)
        vvp[i] = Float32(q[0] - eps)

    parallelize[vert_body](n_gvert)

    # --- simple dual contouring (exact port of the .cu kernel) ---
    var dual = List[Float32](length=n_vox * 3, fill=0)
    var flags = List[Int32](length=n_vox * 3, fill=0)
    var vkp = vox_keys.unsafe_ptr()
    var dp = dual.unsafe_ptr()
    var flp = flags.unsafe_ptr()

    @parameter
    def dc_body(i: Int):
        var key = vkp[i]
        var vx = key // (R * R)
        var vy = (key // R) % R
        var vz = key % R
        var sx = 0.0
        var sy = 0.0
        var sz = 0.0
        var cnt = 0
        # axis X edges: (vx, vy+u, vz+v) -> (vx+1, vy+u, vz+v)
        for u in range(2):
            for v in range(2):
                var k1 = ((vx) * R1 + vy + u) * R1 + vz + v
                var k2 = ((vx + 1) * R1 + vy + u) * R1 + vz + v
                var v1 = Float64(vvp[_bsearch(gvp, n_gvert, k1)])
                var v2 = Float64(vvp[_bsearch(gvp, n_gvert, k2)])
                if (v1 < 0 and v2 >= 0) or (v1 >= 0 and v2 < 0):
                    var t = -v1 / (v2 - v1)
                    sx += Float64(vx) + t
                    sy += Float64(vy + u)
                    sz += Float64(vz + v)
                    cnt += 1
                if u == 1 and v == 1:
                    if v1 < 0 and v2 >= 0:
                        flp[i * 3 + 0] = 1
                    elif v1 >= 0 and v2 < 0:
                        flp[i * 3 + 0] = -1
                    else:
                        flp[i * 3 + 0] = 0
        # axis Y edges
        for u in range(2):
            for v in range(2):
                var k1 = ((vx + u) * R1 + vy) * R1 + vz + v
                var k2 = ((vx + u) * R1 + vy + 1) * R1 + vz + v
                var v1 = Float64(vvp[_bsearch(gvp, n_gvert, k1)])
                var v2 = Float64(vvp[_bsearch(gvp, n_gvert, k2)])
                if (v1 < 0 and v2 >= 0) or (v1 >= 0 and v2 < 0):
                    var t = -v1 / (v2 - v1)
                    sx += Float64(vx + u)
                    sy += Float64(vy) + t
                    sz += Float64(vz + v)
                    cnt += 1
                if u == 1 and v == 1:
                    if v1 < 0 and v2 >= 0:
                        flp[i * 3 + 1] = 1
                    elif v1 >= 0 and v2 < 0:
                        flp[i * 3 + 1] = -1
                    else:
                        flp[i * 3 + 1] = 0
        # axis Z edges
        for u in range(2):
            for v in range(2):
                var k1 = ((vx + u) * R1 + vy + v) * R1 + vz
                var k2 = ((vx + u) * R1 + vy + v) * R1 + vz + 1
                var v1 = Float64(vvp[_bsearch(gvp, n_gvert, k1)])
                var v2 = Float64(vvp[_bsearch(gvp, n_gvert, k2)])
                if (v1 < 0 and v2 >= 0) or (v1 >= 0 and v2 < 0):
                    var t = -v1 / (v2 - v1)
                    sx += Float64(vx + u)
                    sy += Float64(vy + v)
                    sz += Float64(vz) + t
                    cnt += 1
                if u == 1 and v == 1:
                    if v1 < 0 and v2 >= 0:
                        flp[i * 3 + 2] = 1
                    elif v1 >= 0 and v2 < 0:
                        flp[i * 3 + 2] = -1
                    else:
                        flp[i * 3 + 2] = 0
        if cnt > 0:
            dp[i * 3 + 0] = Float32(sx / Float64(cnt))
            dp[i * 3 + 1] = Float32(sy / Float64(cnt))
            dp[i * 3 + 2] = Float32(sz / Float64(cnt))
        else:
            dp[i * 3 + 0] = Float32(vx) + 0.5
            dp[i * 3 + 1] = Float32(vy) + 0.5
            dp[i * 3 + 2] = Float32(vz) + 0.5

    parallelize[dc_body](n_vox)

    # --- topology: one quad per intersected far-edge whose 4 voxels
    # are all active (upstream edge_neighbor_voxel_offset tables) ---
    var quad = List[Int]()
    var qdir = List[Int32]()
    for i in range(n_vox):
        var key = vox_keys[i]
        var vx = key // (R * R)
        var vy = (key // R) % R
        var vz = key % R
        for a in range(3):
            var f = flags[i * 3 + a]
            if f == 0:
                continue
            # neighbor voxel offsets per axis (cyclic around the edge)
            var ok = True
            var idx4 = List[Int](length=4, fill=0)
            for j in range(4):
                var ox: Int
                var oy: Int
                var oz: Int
                if a == 0:
                    if j == 0:
                        ox = 0
                        oy = 0
                        oz = 0
                    elif j == 1:
                        ox = 0
                        oy = 0
                        oz = 1
                    elif j == 2:
                        ox = 0
                        oy = 1
                        oz = 1
                    else:
                        ox = 0
                        oy = 1
                        oz = 0
                elif a == 1:
                    if j == 0:
                        ox = 0
                        oy = 0
                        oz = 0
                    elif j == 1:
                        ox = 1
                        oy = 0
                        oz = 0
                    elif j == 2:
                        ox = 1
                        oy = 0
                        oz = 1
                    else:
                        ox = 0
                        oy = 0
                        oz = 1
                else:
                    if j == 0:
                        ox = 0
                        oy = 0
                        oz = 0
                    elif j == 1:
                        ox = 0
                        oy = 1
                        oz = 0
                    elif j == 2:
                        ox = 1
                        oy = 1
                        oz = 0
                    else:
                        ox = 1
                        oy = 0
                        oz = 0
                var nx = vx + ox
                var ny = vy + oy
                var nz = vz + oz
                if nx >= R or ny >= R or nz >= R:
                    ok = False
                    break
                var nk = (nx * R + ny) * R + nz
                var ni = _bsearch(vkp, n_vox, nk)
                if ni < 0:
                    ok = False
                    break
                idx4[j] = ni
            if not ok:
                continue
            for j in range(4):
                quad.append(idx4[j])
            qdir.append(f)
    var n_quads = len(qdir)
    if n_quads == 0:
        return (out_v^, out_f^)

    # --- compact used dual verts (ascending, like torch.unique) ---
    var used = List[Bool](length=n_vox, fill=False)
    for i in range(n_quads * 4):
        used[quad[i]] = True
    var vmap = List[Int](length=n_vox, fill=-1)
    var world = List[Float32]()
    var n_out = 0
    for i in range(n_vox):
        if used[i]:
            vmap[i] = n_out
            world.append(Float32((Float64(dual[i * 3 + 0]) / Float64(R) - 0.5) * scale + c0))
            world.append(Float32((Float64(dual[i * 3 + 1]) / Float64(R) - 0.5) * scale + c1))
            world.append(Float32((Float64(dual[i * 3 + 2]) / Float64(R) - 0.5) * scale + c2))
            n_out += 1

    # --- triangles: orientation from crossing sign, diagonal by the
    # INTENDED alignment test (see module header for the upstream bug)
    var tris = List[Int32]()
    for qi in range(n_quads):
        var q = List[Int](length=4, fill=0)
        for j in range(4):
            q[j] = vmap[quad[qi * 4 + j]]
        var pos = qdir[qi] == 1
        # splitA: p (0,2,1),(0,3,2); n (0,1,2),(0,2,3)
        # splitB: p (0,3,1),(3,2,1); n (0,1,3),(3,1,2)
        var a0: Int
        var a1: Int
        var a2: Int
        var a3: Int
        var a4: Int
        var a5: Int
        var b0: Int
        var b1: Int
        var b2: Int
        var b3: Int
        var b4: Int
        var b5: Int
        if pos:
            a0 = q[0]
            a1 = q[2]
            a2 = q[1]
            a3 = q[0]
            a4 = q[3]
            a5 = q[2]
            b0 = q[0]
            b1 = q[3]
            b2 = q[1]
            b3 = q[3]
            b4 = q[2]
            b5 = q[1]
        else:
            a0 = q[0]
            a1 = q[1]
            a2 = q[2]
            a3 = q[0]
            a4 = q[2]
            a5 = q[3]
            b0 = q[0]
            b1 = q[1]
            b2 = q[3]
            b3 = q[3]
            b4 = q[1]
            b5 = q[2]
        var alignA = _tri_pair_align(world, a0, a1, a2, a3, a4, a5)
        var alignB = _tri_pair_align(world, b0, b1, b2, b3, b4, b5)
        if alignA >= alignB:
            tris.append(Int32(a0))
            tris.append(Int32(a1))
            tris.append(Int32(a2))
            tris.append(Int32(a3))
            tris.append(Int32(a4))
            tris.append(Int32(a5))
        else:
            tris.append(Int32(b0))
            tris.append(Int32(b1))
            tris.append(Int32(b2))
            tris.append(Int32(b3))
            tris.append(Int32(b4))
            tris.append(Int32(b5))

    # --- optional projection back to the original surface ---
    if project_back > 0:
        var wp2 = world.unsafe_ptr()

        @parameter
        def proj_body(i: Int):
            var wx = Float64(wp2[i * 3 + 0])
            var wy = Float64(wp2[i * 3 + 1])
            var wz = Float64(wp2[i * 3 + 2])
            var q = _grid_closest(wx, wy, wz, c0, c1, c2, scale, G, sp, ep, vp, fp, r_max)
            if q[0] < r_max:
                wp2[i * 3 + 0] = Float32(wx - project_back * (wx - q[1]))
                wp2[i * 3 + 1] = Float32(wy - project_back * (wy - q[2]))
                wp2[i * 3 + 2] = Float32(wz - project_back * (wz - q[3]))

        parallelize[proj_body](n_out)

    out_v = Tensor[F32]([n_out, 3])
    for i in range(n_out * 3):
        out_v.data[i] = world[i]
    out_f = IntMatrix(len(tris) // 3, 3)
    for i in range(len(tris)):
        out_f.data[i] = tris[i]
    return (out_v^, out_f^)


def _tri_pair_align(
    world: List[Float32],
    t0: Int, t1: Int, t2: Int, t3: Int, t4: Int, t5: Int,
) raises -> Float64:
    """|n(tri1) . n(tri2)| for a 6-index split candidate — the
    alignment test upstream INTENDED (their code compares triangle 1
    with itself / a zero normal, see the module header)."""
    var n1 = _tri_normal(world, t0, t1, t2)
    var n2 = _tri_normal(world, t3, t4, t5)
    var d = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
    if d < 0:
        d = -d
    return d


def _tri_normal(
    world: List[Float32], i0: Int, i1: Int, i2: Int
) raises -> Tuple[Float64, Float64, Float64]:
    var ax = Float64(world[i0 * 3 + 0])
    var ay = Float64(world[i0 * 3 + 1])
    var az = Float64(world[i0 * 3 + 2])
    var ux = Float64(world[i1 * 3 + 0]) - ax
    var uy = Float64(world[i1 * 3 + 1]) - ay
    var uz = Float64(world[i1 * 3 + 2]) - az
    var wx = Float64(world[i2 * 3 + 0]) - ax
    var wy = Float64(world[i2 * 3 + 1]) - ay
    var wz = Float64(world[i2 * 3 + 2]) - az
    return (uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx)
