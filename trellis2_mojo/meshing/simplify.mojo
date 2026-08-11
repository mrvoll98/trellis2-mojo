# Native-Mojo QEM mesh simplification.
#
# Intent port of CuMesh/mtlmesh simplify.cu, used after the watertight
# narrow-band remesh. Each round:
#   1. builds vertex/face adjacency and unique undirected edges;
#   2. accumulates a normalized face-plane quadric per vertex;
#   3. scores boundary-aware midpoint collapses with QEM, edge-length and
#      skinny-triangle terms, rejecting normal flips;
#   4. chooses a deterministic conflict-free set (an edge must be the best
#      candidate for every face incident to either endpoint);
#   5. collapses the set and compacts the mesh.
#
# The reference runs the same rounds as parallel device kernels. This CPU
# implementation uses f64 for costs/geometry guards and preserves stable
# face/vertex order. It deliberately keeps the reference's midpoint policy
# instead of solving the quadric optimum: one boundary endpoint is pinned;
# otherwise the new point is the midpoint.

from std.collections import Dict
from std.math import sqrt

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32
comptime HUGE = 1e300
comptime SQRT3_X4 = 6.928203230275509


struct _Topology(Movable):
    var offsets: List[Int]
    var vert_faces: List[Int]
    var edge_a: List[Int]
    var edge_b: List[Int]
    var boundary: List[Bool]

    def __init__(
        out self,
        var offsets: List[Int],
        var vert_faces: List[Int],
        var edge_a: List[Int],
        var edge_b: List[Int],
        var boundary: List[Bool],
    ):
        self.offsets = offsets^
        self.vert_faces = vert_faces^
        self.edge_a = edge_a^
        self.edge_b = edge_b^
        self.boundary = boundary^


def _build_topology(vertices: Tensor[F32], faces: IntMatrix) raises -> _Topology:
    var nv = vertices.shape[0]
    var counts = List[Int](length=nv, fill=0)
    var edge_index = Dict[Int, Int]()
    var edge_a = List[Int]()
    var edge_b = List[Int]()
    var edge_count = List[Int]()

    for f in range(faces.rows):
        for k in range(3):
            var v = faces.at(f, k)
            if v < 0 or v >= nv:
                raise Error("simplify: face index out of range")
            counts[v] += 1
            var a = v
            var b = faces.at(f, (k + 1) % 3)
            if a > b:
                var swap = a
                a = b
                b = swap
            if a == b:
                continue
            var key = a * nv + b
            if key in edge_index:
                edge_count[edge_index[key]] += 1
            else:
                edge_index[key] = len(edge_a)
                edge_a.append(a)
                edge_b.append(b)
                edge_count.append(1)

    var offsets = List[Int](length=nv + 1, fill=0)
    for v in range(nv):
        offsets[v + 1] = offsets[v] + counts[v]
    var vert_faces = List[Int](length=faces.rows * 3, fill=0)
    var cursor = List[Int](length=nv, fill=0)
    for v in range(nv):
        cursor[v] = offsets[v]
    for f in range(faces.rows):
        for k in range(3):
            var v = faces.at(f, k)
            vert_faces[cursor[v]] = f
            cursor[v] += 1

    var boundary = List[Bool](length=nv, fill=False)
    for e in range(len(edge_a)):
        if edge_count[e] == 1:
            boundary[edge_a[e]] = True
            boundary[edge_b[e]] = True
    return _Topology(offsets^, vert_faces^, edge_a^, edge_b^, boundary^)


def _add_plane(
    mut qem: List[Float64], base: Int,
    a: Float64, b: Float64, c: Float64, d: Float64,
):
    qem[base + 0] += a * a
    qem[base + 1] += a * b
    qem[base + 2] += a * c
    qem[base + 3] += a * d
    qem[base + 4] += b * b
    qem[base + 5] += b * c
    qem[base + 6] += b * d
    qem[base + 7] += c * c
    qem[base + 8] += c * d
    qem[base + 9] += d * d


def _get_qem(vertices: Tensor[F32], faces: IntMatrix) raises -> List[Float64]:
    var qem = List[Float64](length=vertices.shape[0] * 10, fill=0)
    for f in range(faces.rows):
        var ia = faces.at(f, 0)
        var ib = faces.at(f, 1)
        var ic = faces.at(f, 2)
        var ax = Float64(vertices.data[ia * 3 + 0])
        var ay = Float64(vertices.data[ia * 3 + 1])
        var az = Float64(vertices.data[ia * 3 + 2])
        var ux = Float64(vertices.data[ib * 3 + 0]) - ax
        var uy = Float64(vertices.data[ib * 3 + 1]) - ay
        var uz = Float64(vertices.data[ib * 3 + 2]) - az
        var vx = Float64(vertices.data[ic * 3 + 0]) - ax
        var vy = Float64(vertices.data[ic * 3 + 1]) - ay
        var vz = Float64(vertices.data[ic * 3 + 2]) - az
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        var n2 = nx * nx + ny * ny + nz * nz
        if n2 <= 1e-30:
            continue
        var inv = 1.0 / sqrt(n2)
        nx *= inv
        ny *= inv
        nz *= inv
        var d = -(nx * ax + ny * ay + nz * az)
        _add_plane(qem, ia * 10, nx, ny, nz, d)
        _add_plane(qem, ib * 10, nx, ny, nz, d)
        _add_plane(qem, ic * 10, nx, ny, nz, d)
    return qem^


def _qem_eval_pair(
    qem: List[Float64], a: Int, b: Int,
    x: Float64, y: Float64, z: Float64,
) -> Float64:
    var ba = a * 10
    var bb = b * 10
    var e0 = qem[ba + 0] + qem[bb + 0]
    var e1 = qem[ba + 1] + qem[bb + 1]
    var e2 = qem[ba + 2] + qem[bb + 2]
    var e3 = qem[ba + 3] + qem[bb + 3]
    var e4 = qem[ba + 4] + qem[bb + 4]
    var e5 = qem[ba + 5] + qem[bb + 5]
    var e6 = qem[ba + 6] + qem[bb + 6]
    var e7 = qem[ba + 7] + qem[bb + 7]
    var e8 = qem[ba + 8] + qem[bb + 8]
    var e9 = qem[ba + 9] + qem[bb + 9]
    return (
        e0 * x * x + 2.0 * e1 * x * y + 2.0 * e2 * x * z
        + 2.0 * e3 * x + e4 * y * y + 2.0 * e5 * y * z
        + 2.0 * e6 * y + e7 * z * z + 2.0 * e8 * z + e9
    )


def _contains_vertex(faces: IntMatrix, f: Int, v: Int) raises -> Bool:
    return (
        faces.at(f, 0) == v or faces.at(f, 1) == v
        or faces.at(f, 2) == v
    )


def _incident_valid(
    tri: Int,
    keep: Int,
    other: Int,
    vertices: Tensor[F32],
    faces: IntMatrix,
    nx: Float64,
    ny: Float64,
    nz: Float64,
    mut skinny: Float64,
    mut num_tri: Int,
) raises -> Bool:
    # Shared faces disappear when the edge collapses.
    if _contains_vertex(faces, tri, other):
        return True

    var ia = faces.at(tri, 0)
    var ib = faces.at(tri, 1)
    var ic = faces.at(tri, 2)
    var ax = Float64(vertices.data[ia * 3 + 0])
    var ay = Float64(vertices.data[ia * 3 + 1])
    var az = Float64(vertices.data[ia * 3 + 2])
    var bx = Float64(vertices.data[ib * 3 + 0])
    var by = Float64(vertices.data[ib * 3 + 1])
    var bz = Float64(vertices.data[ib * 3 + 2])
    var cx = Float64(vertices.data[ic * 3 + 0])
    var cy = Float64(vertices.data[ic * 3 + 1])
    var cz = Float64(vertices.data[ic * 3 + 2])

    var oux = bx - ax
    var ouy = by - ay
    var ouz = bz - az
    var ovx = cx - ax
    var ovy = cy - ay
    var ovz = cz - az
    var onx = ouy * ovz - ouz * ovy
    var ony = ouz * ovx - oux * ovz
    var onz = oux * ovy - ouy * ovx

    if ia == keep:
        ax = nx
        ay = ny
        az = nz
    if ib == keep:
        bx = nx
        by = ny
        bz = nz
    if ic == keep:
        cx = nx
        cy = ny
        cz = nz
    var ux = bx - ax
    var uy = by - ay
    var uz = bz - az
    var vx = cx - ax
    var vy = cy - ay
    var vz = cz - az
    var nnx = uy * vz - uz * vy
    var nny = uz * vx - ux * vz
    var nnz = ux * vy - uy * vx
    if onx * nnx + ony * nny + onz * nnz < 0.0:
        return False

    var area = 0.5 * sqrt(nnx * nnx + nny * nny + nnz * nnz)
    var ex = cx - bx
    var ey = cy - by
    var ez = cz - bz
    var denom = (
        ex * ex + ey * ey + ez * ez
        + ux * ux + uy * uy + uz * uz
        + vx * vx + vy * vy + vz * vz
    )
    if denom < 1e-30:
        denom = 1e-30
    var shape = SQRT3_X4 * area / denom
    if shape < 0:
        shape = 0
    if shape > 1:
        shape = 1
    skinny += 1.0 - shape
    num_tri += 1
    return True


def _edge_position(
    vertices: Tensor[F32], boundary: List[Bool], a: Int, b: Int,
) -> Tuple[Float64, Float64, Float64]:
    var w = 0.5
    if boundary[a] and not boundary[b]:
        w = 1.0
    elif not boundary[a] and boundary[b]:
        w = 0.0
    return (
        Float64(vertices.data[a * 3 + 0]) * w
        + Float64(vertices.data[b * 3 + 0]) * (1.0 - w),
        Float64(vertices.data[a * 3 + 1]) * w
        + Float64(vertices.data[b * 3 + 1]) * (1.0 - w),
        Float64(vertices.data[a * 3 + 2]) * w
        + Float64(vertices.data[b * 3 + 2]) * (1.0 - w),
    )


def _edge_costs(
    vertices: Tensor[F32],
    faces: IntMatrix,
    topo: _Topology,
    qem: List[Float64],
    lambda_edge_length: Float64,
    lambda_skinny: Float64,
) raises -> List[Float64]:
    var ne = len(topo.edge_a)
    var costs = List[Float64](length=ne, fill=HUGE)
    for e in range(ne):
        var a = topo.edge_a[e]
        var b = topo.edge_b[e]
        var p = _edge_position(vertices, topo.boundary, a, b)
        var dx = Float64(vertices.data[b * 3 + 0]) - Float64(vertices.data[a * 3 + 0])
        var dy = Float64(vertices.data[b * 3 + 1]) - Float64(vertices.data[a * 3 + 1])
        var dz = Float64(vertices.data[b * 3 + 2]) - Float64(vertices.data[a * 3 + 2])
        var length2 = dx * dx + dy * dy + dz * dz
        var skinny: Float64 = 0
        var num_tri = 0
        var valid = True
        for i in range(topo.offsets[a], topo.offsets[a + 1]):
            if not _incident_valid(
                topo.vert_faces[i], a, b, vertices, faces,
                p[0], p[1], p[2], skinny, num_tri,
            ):
                valid = False
                break
        if valid:
            for i in range(topo.offsets[b], topo.offsets[b + 1]):
                if not _incident_valid(
                    topo.vert_faces[i], b, a, vertices, faces,
                    p[0], p[1], p[2], skinny, num_tri,
                ):
                    valid = False
                    break
        if not valid:
            continue
        if num_tri > 0:
            skinny /= Float64(num_tri)
        costs[e] = (
            _qem_eval_pair(qem, a, b, p[0], p[1], p[2])
            + lambda_edge_length * length2
            + lambda_skinny * skinny * length2
        )
    return costs^


def _prefer(cost: Float64, edge: Int, best_cost: Float64, best_edge: Int) -> Bool:
    return cost < best_cost or (cost == best_cost and edge < best_edge)


def _simplify_step(
    mut vertices: Tensor[F32],
    mut faces: IntMatrix,
    lambda_edge_length: Float64,
    lambda_skinny: Float64,
    threshold: Float64,
) raises -> Int:
    if faces.rows == 0:
        return 0
    var topo = _build_topology(vertices, faces)
    if len(topo.edge_a) == 0:
        return 0
    var qem = _get_qem(vertices, faces)
    var costs = _edge_costs(
        vertices, faces, topo, qem,
        lambda_edge_length, lambda_skinny,
    )

    # Propagate the lexicographic (cost, edge id) minimum to every face
    # incident to either endpoint. This is the deterministic CPU form of
    # the packed atomic-min reference kernel.
    var best_edge = List[Int](length=faces.rows, fill=-1)
    var best_cost = List[Float64](length=faces.rows, fill=HUGE)
    for e in range(len(topo.edge_a)):
        if costs[e] >= HUGE:
            continue
        var a = topo.edge_a[e]
        var b = topo.edge_b[e]
        for i in range(topo.offsets[a], topo.offsets[a + 1]):
            var f = topo.vert_faces[i]
            if _prefer(costs[e], e, best_cost[f], best_edge[f]):
                best_cost[f] = costs[e]
                best_edge[f] = e
        for i in range(topo.offsets[b], topo.offsets[b + 1]):
            var f = topo.vert_faces[i]
            if _prefer(costs[e], e, best_cost[f], best_edge[f]):
                best_cost[f] = costs[e]
                best_edge[f] = e

    var selected = List[Bool](length=len(topo.edge_a), fill=False)
    for e in range(len(topo.edge_a)):
        if costs[e] > threshold:
            continue
        var a = topo.edge_a[e]
        var b = topo.edge_b[e]
        var owns = True
        for i in range(topo.offsets[a], topo.offsets[a + 1]):
            if best_edge[topo.vert_faces[i]] != e:
                owns = False
                break
        if owns:
            for i in range(topo.offsets[b], topo.offsets[b + 1]):
                if best_edge[topo.vert_faces[i]] != e:
                    owns = False
                    break
        selected[e] = owns

    var keep_vertex = List[Bool](length=vertices.shape[0], fill=True)
    var keep_face = List[Bool](length=faces.rows, fill=True)
    var collapsed = 0
    for e in range(len(topo.edge_a)):
        if not selected[e]:
            continue
        var a = topo.edge_a[e]
        var b = topo.edge_b[e]
        var p = _edge_position(vertices, topo.boundary, a, b)
        vertices.data[a * 3 + 0] = Float32(p[0])
        vertices.data[a * 3 + 1] = Float32(p[1])
        vertices.data[a * 3 + 2] = Float32(p[2])
        keep_vertex[b] = False
        collapsed += 1
        for i in range(topo.offsets[a], topo.offsets[a + 1]):
            var f = topo.vert_faces[i]
            if _contains_vertex(faces, f, b):
                keep_face[f] = False
        for i in range(topo.offsets[b], topo.offsets[b + 1]):
            var f = topo.vert_faces[i]
            if not keep_face[f]:
                continue
            for k in range(3):
                if faces.at(f, k) == b:
                    faces.set(f, k, a)
    if collapsed == 0:
        return 0

    var vmap = List[Int](length=vertices.shape[0], fill=-1)
    var new_vertices = List[Float32]()
    var nv = 0
    for v in range(vertices.shape[0]):
        if keep_vertex[v]:
            vmap[v] = nv
            new_vertices.append(vertices.data[v * 3 + 0])
            new_vertices.append(vertices.data[v * 3 + 1])
            new_vertices.append(vertices.data[v * 3 + 2])
            nv += 1

    var new_faces = List[Int32]()
    var nf = 0
    for f in range(faces.rows):
        if not keep_face[f]:
            continue
        var a = vmap[faces.at(f, 0)]
        var b = vmap[faces.at(f, 1)]
        var c = vmap[faces.at(f, 2)]
        if a < 0 or b < 0 or c < 0 or a == b or b == c or a == c:
            continue
        new_faces.append(Int32(a))
        new_faces.append(Int32(b))
        new_faces.append(Int32(c))
        nf += 1
    vertices.data = new_vertices^
    vertices.shape[0] = nv
    faces.data = new_faces^
    faces.rows = nf
    return collapsed


def simplify_qem(
    vertices: Tensor[F32],
    faces: IntMatrix,
    target_faces: Int,
    lambda_edge_length: Float64 = 1e-2,
    lambda_skinny: Float64 = 1e-3,
) raises -> Tuple[Tensor[F32], IntMatrix]:
    """Simplify to at most target_faces when topology permits it.

    The threshold schedule matches the reference driver: start at 1e-8 and
    multiply by ten whenever a round removes less than one percent. Since a
    conflict-free round can remove several triangles, the result may land
    below the requested target. If every remaining edge fails the flip guard,
    the best valid mesh reached so far is returned above target.
    """
    if target_faces <= 0:
        raise Error("simplify: target_faces must be positive")
    var out_v = vertices.copy()
    var out_f = faces.copy()
    if out_f.rows <= target_faces:
        return (out_v^, out_f^)

    var threshold: Float64 = 1e-8
    var rounds = 0
    while out_f.rows > target_faces and rounds < 256:
        var before = out_f.rows
        var collapsed = _simplify_step(
            out_v, out_f, lambda_edge_length, lambda_skinny, threshold
        )
        var removed = before - out_f.rows
        if collapsed == 0 or removed * 100 < before:
            threshold *= 10.0
        if threshold > 1e12 and removed == 0:
            break
        rounds += 1
    return (out_v^, out_f^)
