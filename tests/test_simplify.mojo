# Native QEM simplification regression tests.

from std.collections import Dict

from trellis2_mojo.meshing.remesh import remesh_narrow_band_dc
from trellis2_mojo.meshing.simplify import simplify_qem
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def cube_mesh() raises -> Tuple[Tensor[F32], IntMatrix]:
    var vertices = Tensor[F32]([8, 3])
    var i = 0
    for z in range(2):
        for y in range(2):
            for x in range(2):
                vertices.data[i * 3 + 0] = Float32(x) * 0.4 - 0.2
                vertices.data[i * 3 + 1] = Float32(y) * 0.4 - 0.2
                vertices.data[i * 3 + 2] = Float32(z) * 0.4 - 0.2
                i += 1
    var rows: List[Int] = [
        0, 2, 1, 1, 2, 3,
        4, 5, 6, 5, 7, 6,
        0, 1, 4, 1, 5, 4,
        2, 6, 3, 3, 6, 7,
        0, 4, 2, 2, 4, 6,
        1, 3, 5, 3, 7, 5,
    ]
    var faces = IntMatrix(12, 3)
    for j in range(len(rows)):
        faces.data[j] = Int32(rows[j])
    return (vertices^, faces^)


def boundary_edges(faces: IntMatrix, nv: Int) raises -> Int:
    var counts = Dict[Int, Int]()
    for f in range(faces.rows):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key = a * nv + b if a < b else b * nv + a
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
    var result = 0
    for edge in counts.items():
        if edge.value == 1:
            result += 1
    return result


def check_closed_oriented_manifold(faces: IntMatrix, nv: Int) raises:
    var undirected = Dict[Int, Int]()
    var directed = Dict[Int, Bool]()
    for f in range(faces.rows):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var uk = a * nv + b if a < b else b * nv + a
            if uk in undirected:
                undirected[uk] += 1
            else:
                undirected[uk] = 1
            var dk = a * nv + b
            if dk in directed:
                raise Error("simplify: repeated directed edge")
            directed[dk] = True
    for edge in undirected.items():
        if edge.value != 2:
            raise Error(
                "simplify: expected manifold edge count 2, got "
                + String(edge.value)
            )


def validate(vertices: Tensor[F32], faces: IntMatrix) raises:
    for f in range(faces.rows):
        var a = faces.at(f, 0)
        var b = faces.at(f, 1)
        var c = faces.at(f, 2)
        if a < 0 or b < 0 or c < 0:
            raise Error("simplify: negative index")
        if a >= vertices.shape[0] or b >= vertices.shape[0] or c >= vertices.shape[0]:
            raise Error("simplify: index out of range")
        if a == b or b == c or a == c:
            raise Error("simplify: degenerate index triangle")


def main() raises:
    var cube = cube_mesh()
    var scale = (24.0 + 3.0) / 24.0
    var remeshed = remesh_narrow_band_dc(
        cube[0], cube[1], 0, 0, 0, scale, 24, 1.0, 0.9
    )
    var source_v = remeshed[0].copy()
    var source_f = remeshed[1].copy()
    var target = source_f.rows // 3
    var simplified = simplify_qem(source_v, source_f, target)
    var out_v = simplified[0].copy()
    var out_f = simplified[1].copy()
    validate(out_v, out_f)
    if out_f.rows > target:
        raise Error(
            "simplify did not reach target: " + String(out_f.rows)
            + " > " + String(target)
        )
    if boundary_edges(out_f, out_v.shape[0]) != 0:
        raise Error("simplify opened a watertight remesh")
    check_closed_oriented_manifold(out_f, out_v.shape[0])
    if out_f.rows >= source_f.rows or out_v.shape[0] >= source_v.shape[0]:
        raise Error("simplify did not reduce the mesh")

    # Deterministic topology and positions.
    var simplified2 = simplify_qem(source_v, source_f, target)
    if simplified2[0].shape[0] != out_v.shape[0] or simplified2[1].rows != out_f.rows:
        raise Error("simplify determinism: sizes differ")
    for i in range(len(out_v.data)):
        if simplified2[0].data[i] != out_v.data[i]:
            raise Error("simplify determinism: vertices differ")
    for i in range(len(out_f.data)):
        if simplified2[1].data[i] != out_f.data[i]:
            raise Error("simplify determinism: faces differ")

    # A target above the current count is a byte-identical no-op.
    var noop = simplify_qem(out_v, out_f, out_f.rows + 1)
    if noop[0].shape[0] != out_v.shape[0] or noop[1].rows != out_f.rows:
        raise Error("simplify no-op: sizes differ")
    for i in range(len(out_v.data)):
        if noop[0].data[i] != out_v.data[i]:
            raise Error("simplify no-op: vertices differ")
    for i in range(len(out_f.data)):
        if noop[1].data[i] != out_f.data[i]:
            raise Error("simplify no-op: faces differ")

    print(
        "native QEM simplify:", source_v.shape[0], "V /", source_f.rows,
        "F ->", out_v.shape[0], "V /", out_f.rows,
        "F; watertight/manifold/oriented + deterministic + no-op passed",
    )
