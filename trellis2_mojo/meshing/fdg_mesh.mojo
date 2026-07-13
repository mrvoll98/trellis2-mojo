# Pure-Mojo mesh extraction from the sparse voxel dual grid (WP9 part 3
# step 4, ADR 0007 — the o_voxel CUDA path is not used for inference).
#
# Port of trellis-mac stubs/o_voxel_override_convert.py
# `flexible_dual_grid_to_mesh`, whose docstring guarantees output identical
# to the CUDA version for inference. The pipeline always calls it with
# split_weight = quad_lerp (softplus > 0); the normal-alignment branch
# (split_weight absent) is ported too for completeness. Parity:
# pixi run test-mesh (vendored stub as reference).

from std.collections import Dict

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32

# edge_neighbor_voxel_offset[axis][corner] -> (dx, dy, dz): the four voxels
# sharing the edge along `axis` leaving this voxel's origin corner.
comptime _EDGE_OFFSETS: InlineArray[Int, 36] = [
    0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
    0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
]
# quad -> two triangles, the stub's two diagonal splits
comptime _SPLIT_1: InlineArray[Int, 6] = [0, 1, 2, 0, 2, 3]
comptime _SPLIT_2: InlineArray[Int, 6] = [0, 1, 3, 3, 1, 2]


def _pack(x: Int, y: Int, z: Int) raises -> Int:
    """Voxel coord -> dict key; coords are non-negative and < 2^21."""
    return (x << 42) | (y << 21) | z


def flexible_dual_grid_to_mesh(
    coords_xyz: IntMatrix,
    dual_vertices: Tensor[F32],
    intersected: Tensor[F32],
    split_weight: Tensor[F32],
    aabb_min: List[Float64],
    aabb_max: List[Float64],
    grid_size: Int,
) raises -> Tuple[Tensor[F32], IntMatrix]:
    """coords [N,3] (batch column stripped), dual_vertices [N,3],
    intersected [N,3] as 1.0/0.0 (fdg_head output), split_weight [N,1]
    (pass a 0-element tensor for the normal-alignment split). Uniform
    grid_size per axis (the pipeline uses aabb [-0.5,0.5]^3 and
    grid_size=resolution). Returns (vertices [N,3], triangles [2L,3]) —
    like the stub/CUDA version, ALL N vertices are returned, including
    ones no triangle references."""
    var n = coords_xyz.rows
    if dual_vertices.shape[0] != n or intersected.shape[0] != n:
        raise Error("fdg_mesh: row mismatch")
    var use_split = split_weight.numel() > 0
    if use_split and split_weight.shape[0] != n:
        raise Error("fdg_mesh: split_weight row mismatch")

    var voxel = List[Float64]()
    for d in range(3):
        voxel.append((aabb_max[d] - aabb_min[d]) / Float64(grid_size))

    # coord -> index lookup (the stub's Python dict / CUDA hashmap)
    var lut = Dict[Int, Int]()
    for i in range(n):
        lut[_pack(coords_xyz.at(i, 0), coords_xyz.at(i, 1), coords_xyz.at(i, 2))] = i

    # quads: for every intersected edge (row-major (i, axis) order like the
    # stub's boolean mask flattening), the 4 edge-sharing voxels — kept
    # only if all of them exist
    var quads = List[Int]()
    for i in range(n):
        for axis in range(3):
            if intersected.data[i * 3 + axis] == 0:
                continue
            var q0 = -1
            var q1 = -1
            var q2 = -1
            var q3 = -1
            var ok = True
            for corner in range(4):
                var base = axis * 12 + corner * 3
                var key = _pack(
                    coords_xyz.at(i, 0) + _EDGE_OFFSETS[base],
                    coords_xyz.at(i, 1) + _EDGE_OFFSETS[base + 1],
                    coords_xyz.at(i, 2) + _EDGE_OFFSETS[base + 2],
                )
                var idx = lut.get(key, -1)
                if idx < 0:
                    ok = False
                    break
                if corner == 0:
                    q0 = idx
                elif corner == 1:
                    q1 = idx
                elif corner == 2:
                    q2 = idx
                else:
                    q3 = idx
            if ok:
                quads.append(q0)
                quads.append(q1)
                quads.append(q2)
                quads.append(q3)
    var l = len(quads) // 4
    if l == 0:
        # both stub early-outs (no flagged edges / no complete quads)
        # return EMPTY vertices, not the N unused ones
        var eshape: List[Int] = [0, 3]
        return (Tensor[F32](eshape), IntMatrix(0, 3))

    # world-space vertices: (coords + dual) * voxel_size + aabb_min
    var vshape: List[Int] = [n, 3]
    var vertices = Tensor[F32](vshape)
    for i in range(n):
        for d in range(3):
            vertices.data[i * 3 + d] = Float32(
                (Float64(coords_xyz.at(i, d)) + Float64(dual_vertices.data[i * 3 + d]))
                * voxel[d] + aabb_min[d]
            )

    var tris = IntMatrix(2 * l, 3)

    for k in range(l):
        var q: InlineArray[Int, 4] = [
            quads[4 * k], quads[4 * k + 1], quads[4 * k + 2], quads[4 * k + 3]
        ]
        var pick_first: Bool
        if use_split:
            # sw[q0]*sw[q2] > sw[q1]*sw[q3] -> split 1 (strict, like torch.where cond)
            var sw02 = split_weight.data[q[0]] * split_weight.data[q[2]]
            var sw13 = split_weight.data[q[1]] * split_weight.data[q[3]]
            pick_first = sw02 > sw13
        else:
            pick_first = _align(vertices, q, _SPLIT_1) > _align(vertices, q, _SPLIT_2)
        for j in range(6):
            var perm: Int
            if pick_first:
                perm = _SPLIT_1[j]
            else:
                perm = _SPLIT_2[j]
            tris.set(2 * k + j // 3, j % 3, q[perm])
    return (vertices^, tris^)


def _align(
    vertices: Tensor[F32], q: InlineArray[Int, 4], split: InlineArray[Int, 6]
) raises -> Float32:
    """|dot(n0, n1)| of the two triangle normals for one diagonal split
    (the stub's normal-alignment criterion), in f32 like the reference."""
    var a0 = q[split[0]]
    var a1 = q[split[1]]
    var a2 = q[split[2]]
    var a3 = q[split[3]]
    var n0: InlineArray[Float32, 3] = [0, 0, 0]
    var n1: InlineArray[Float32, 3] = [0, 0, 0]
    _cross(vertices, a1, a0, a2, a0, n0)
    _cross(vertices, a2, a1, a3, a1, n1)
    var dot: Float32 = 0
    for d in range(3):
        dot += n0[d] * n1[d]
    if dot < 0:
        dot = -dot
    return dot


def _cross(
    vertices: Tensor[F32], p: Int, p0: Int, r: Int, r0: Int, mut out: InlineArray[Float32, 3]
) raises:
    """out = (v[p] - v[p0]) x (v[r] - v[r0])."""
    var a: InlineArray[Float32, 3] = [0, 0, 0]
    var b: InlineArray[Float32, 3] = [0, 0, 0]
    for d in range(3):
        a[d] = vertices.data[p * 3 + d] - vertices.data[p0 * 3 + d]
        b[d] = vertices.data[r * 3 + d] - vertices.data[r0 * 3 + d]
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]


def write_obj(path: String, vertices: Tensor[F32], triangles: IntMatrix) raises:
    """Wavefront OBJ (the format the trellis-mac reference exports too).
    Face indices are 1-based; buffered writes to keep String appends short."""
    var f = open(path, "w")
    var buf = String()
    for i in range(vertices.shape[0]):
        buf += "v " + String(vertices.data[i * 3])
        buf += " " + String(vertices.data[i * 3 + 1])
        buf += " " + String(vertices.data[i * 3 + 2]) + "\n"
        if i % 4096 == 4095:
            f.write(buf)
            buf = String()
    for k in range(triangles.rows):
        buf += "f " + String(triangles.at(k, 0) + 1)
        buf += " " + String(triangles.at(k, 1) + 1)
        buf += " " + String(triangles.at(k, 2) + 1) + "\n"
        if k % 4096 == 4095:
            f.write(buf)
            buf = String()
    f.write(buf)
    f.close()
