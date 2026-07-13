# WP15 (fase 2, ADR 0008): vertex attributes for the textured GLB export —
# a pure-Mojo port of flex_gemm.ops.grid_sample.grid_sample_3d
# (mode='trilinear', B=1) plus area-weighted vertex normals.
#
# The trilinear semantics mirror the flex_gemm torch reference EXACTLY:
# the 8 neighbor voxels are the TRUNCATED (toward zero, torch .int())
# p±0.5 coordinates — for p < 0.5 two offsets truncate to the SAME voxel
# and it counts twice, a reference quirk we keep — each neighbor weighs
# prod(1 - |neigh + 0.5 - p|) in x,y,z order, voxels absent from the
# sparse set weigh 0 (hashmap miss upstream, Dict miss here; negative or
# out-of-range coords are misses too), and the result is renormalized by
# the weight sum clamped to 1e-12 (all-miss queries return 0).
# Parity: pixi run test-wp15 (plain-torch reimplementation as reference).

from std.collections import Dict
from std.math import sqrt

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32

# the reference's neighbor-offset order (torch tensor rows, ±0.5 each axis)
comptime _OFFS: InlineArray[Float32, 24] = [
    -0.5, -0.5, -0.5,
    -0.5, -0.5, 0.5,
    -0.5, 0.5, -0.5,
    -0.5, 0.5, 0.5,
    0.5, -0.5, -0.5,
    0.5, -0.5, 0.5,
    0.5, 0.5, -0.5,
    0.5, 0.5, 0.5,
]


def _pack(x: Int, y: Int, z: Int) raises -> Int:
    """Voxel coord -> dict key; coords are non-negative and < 2^21
    (same packing as fdg_mesh)."""
    return (x << 42) | (y << 21) | z


def grid_sample_trilinear(
    feats: Tensor[F32],
    coords_xyz: IntMatrix,
    grid_size: Int,
    query: Tensor[F32],
) raises -> Tensor[F32]:
    """feats [N, C] + sparse voxel coords [N, 3] (batch column stripped,
    ints in [0, grid_size)) sampled at query points [L, 3] given in VOXEL
    units ((pos - aabb_min) / voxel_size) -> [L, C]."""
    var n = coords_xyz.rows
    if feats.shape[0] != n:
        raise Error("grid_sample_trilinear: feats/coords row mismatch")
    var c = feats.shape[1]
    if query.shape[1] != 3:
        raise Error("grid_sample_trilinear: query must be [L, 3]")
    var l = query.shape[0]

    var lut = Dict[Int, Int]()
    for i in range(n):
        lut[_pack(coords_xyz.at(i, 0), coords_xyz.at(i, 1), coords_xyz.at(i, 2))] = i

    var out = Tensor[F32]([l, c])
    var acc = List[Float32](length=c, fill=0)
    for q in range(l):
        var px = query.data[q * 3 + 0]
        var py = query.data[q * 3 + 1]
        var pz = query.data[q * 3 + 2]
        for ch in range(c):
            acc[ch] = 0
        var wsum: Float32 = 0
        for k in range(8):
            # torch .int() truncates toward zero — keep that, NOT floor
            var nx = Int(px + _OFFS[k * 3 + 0])
            var ny = Int(py + _OFFS[k * 3 + 1])
            var nz = Int(pz + _OFFS[k * 3 + 2])
            if nx < 0 or ny < 0 or nz < 0:
                continue  # upstream hashmap miss: contributes nothing
            if nx >= grid_size or ny >= grid_size or nz >= grid_size:
                continue
            var key = _pack(nx, ny, nz)
            if key not in lut:
                continue
            var idx = lut[key]
            var w = (
                (1.0 - abs(Float32(nx) + 0.5 - px))
                * (1.0 - abs(Float32(ny) + 0.5 - py))
                * (1.0 - abs(Float32(nz) + 0.5 - pz))
            )
            wsum += w
            var base = idx * c
            for ch in range(c):
                acc[ch] += w * feats.data[base + ch]
        var denom = wsum
        if denom < 1e-12:
            denom = 1e-12
        var obase = q * c
        for ch in range(c):
            out.data[obase + ch] = acc[ch] / denom
    return out^


def vertex_normals(
    vertices: Tensor[F32], faces: IntMatrix
) raises -> Tensor[F32]:
    """Area-weighted vertex normals [V, 3]: accumulate each face's
    (unnormalized) cross product onto its three vertices, then normalize.
    EVERY returned normal is unit length (the glTF validator requires
    it): vertices whose accumulation CANCELS to ~zero — the hole-fill
    centroids of braided/folded components, which otherwise shade as
    black specks that look exactly like micro holes — get the average of
    their incident faces' other corners' unit normals; anything still
    zero after that (fully cancelled neighborhoods, unreferenced
    vertices) gets a deterministic +z."""
    var v = vertices.shape[0]
    var out = Tensor[F32]([v, 3])
    for f in range(faces.rows):
        var i0 = faces.at(f, 0)
        var i1 = faces.at(f, 1)
        var i2 = faces.at(f, 2)
        var ax = vertices.data[i1 * 3 + 0] - vertices.data[i0 * 3 + 0]
        var ay = vertices.data[i1 * 3 + 1] - vertices.data[i0 * 3 + 1]
        var az = vertices.data[i1 * 3 + 2] - vertices.data[i0 * 3 + 2]
        var bx = vertices.data[i2 * 3 + 0] - vertices.data[i0 * 3 + 0]
        var by = vertices.data[i2 * 3 + 1] - vertices.data[i0 * 3 + 1]
        var bz = vertices.data[i2 * 3 + 2] - vertices.data[i0 * 3 + 2]
        var cx = ay * bz - az * by
        var cy = az * bx - ax * bz
        var cz = ax * by - ay * bx
        out.data[i0 * 3 + 0] += cx
        out.data[i0 * 3 + 1] += cy
        out.data[i0 * 3 + 2] += cz
        out.data[i1 * 3 + 0] += cx
        out.data[i1 * 3 + 1] += cy
        out.data[i1 * 3 + 2] += cz
        out.data[i2 * 3 + 0] += cx
        out.data[i2 * 3 + 1] += cy
        out.data[i2 * 3 + 2] += cz
    var zero = List[Bool](length=v, fill=False)
    var n_zero = 0
    for i in range(v):
        var nx = out.data[i * 3 + 0]
        var ny = out.data[i * 3 + 1]
        var nz = out.data[i * 3 + 2]
        var norm = sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-20:
            out.data[i * 3 + 0] = nx / norm
            out.data[i * 3 + 1] = ny / norm
            out.data[i * 3 + 2] = nz / norm
        else:
            zero[i] = True
            n_zero += 1
    if n_zero > 0:
        # neighbor pass: average the unit normals of the other corners of
        # every incident face (accumulated into a separate buffer so the
        # pass reads only first-pass results — deterministic)
        var fix = Tensor[F32]([v, 3])
        for f in range(faces.rows):
            for k in range(3):
                var i = faces.at(f, k)
                if not zero[i]:
                    continue
                var j1 = faces.at(f, (k + 1) % 3)
                var j2 = faces.at(f, (k + 2) % 3)
                fix.data[i * 3 + 0] += out.data[j1 * 3 + 0] + out.data[j2 * 3 + 0]
                fix.data[i * 3 + 1] += out.data[j1 * 3 + 1] + out.data[j2 * 3 + 1]
                fix.data[i * 3 + 2] += out.data[j1 * 3 + 2] + out.data[j2 * 3 + 2]
        for i in range(v):
            if not zero[i]:
                continue
            var nx = fix.data[i * 3 + 0]
            var ny = fix.data[i * 3 + 1]
            var nz = fix.data[i * 3 + 2]
            var norm = sqrt(nx * nx + ny * ny + nz * nz)
            if norm > 1e-12:
                out.data[i * 3 + 0] = nx / norm
                out.data[i * 3 + 1] = ny / norm
                out.data[i * 3 + 2] = nz / norm
            else:
                out.data[i * 3 + 0] = 0
                out.data[i * 3 + 1] = 0
                out.data[i * 3 + 2] = 1
    return out^
