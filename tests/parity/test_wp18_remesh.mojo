# WP18 (fase 2): narrow-band dual-contouring remesh — pure-Mojo tests
# (the reference, cumesh.remeshing.remesh_narrow_band_dc, is read from
# the local mtlmesh source; intent port, no CUDA). Covers: a closed
# cube remeshes to a WATERTIGHT double shell (outer + inner offset
# sheet) with globally consistent winding and positive signed volume
# (orientation sanity), a PUNCTURED cube — the speck-relevant property
# — still remeshes watertight (the offset shell of an open surface is
# the boundary of a closed set), projection pulls vertices to within a
# fraction of a voxel of the original surface, determinism, and empty
# input.
#
# Run from repo root: pixi run test-wp18

from std.collections import Dict
from std.math import sqrt

from trellis2_mojo.meshing.remesh import remesh_narrow_band_dc
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def make_faces(rows: List[Int]) raises -> IntMatrix:
    var m = IntMatrix(len(rows) // 3, 3)
    for i in range(len(rows)):
        m.data[i] = Int32(rows[i])
    return m^


def boundary_edges(faces: IntMatrix, v: Int) raises -> Int:
    var cnt = Dict[Int, Int]()
    for f in range(faces.rows):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key: Int
            if a < b:
                key = a * v + b
            else:
                key = b * v + a
            if key in cnt:
                cnt[key] += 1
            else:
                cnt[key] = 1
    var n = 0
    for e in cnt.items():
        if e.value == 1:
            n += 1
    return n


def check_consistent_winding(faces: IntMatrix, v: Int) raises:
    """Every directed edge at most once — the orientation invariant."""
    var d = Dict[Int, Int]()
    for f in range(faces.rows):
        for k in range(3):
            var a = faces.at(f, k)
            var b = faces.at(f, (k + 1) % 3)
            var key = a * v + b
            if key in d:
                raise Error("directed edge repeated -> inconsistent winding")
            d[key] = 1


def signed_volume(verts: Tensor[F32], faces: IntMatrix) raises -> Float64:
    var s: Float64 = 0
    for f in range(faces.rows):
        var i0 = faces.at(f, 0)
        var i1 = faces.at(f, 1)
        var i2 = faces.at(f, 2)
        var ax = Float64(verts.data[i0 * 3 + 0])
        var ay = Float64(verts.data[i0 * 3 + 1])
        var az = Float64(verts.data[i0 * 3 + 2])
        var bx = Float64(verts.data[i1 * 3 + 0])
        var by = Float64(verts.data[i1 * 3 + 1])
        var bz = Float64(verts.data[i1 * 3 + 2])
        var cx = Float64(verts.data[i2 * 3 + 0])
        var cy = Float64(verts.data[i2 * 3 + 1])
        var cz = Float64(verts.data[i2 * 3 + 2])
        s += (
            ax * (by * cz - bz * cy)
            - ay * (bx * cz - bz * cx)
            + az * (bx * cy - by * cx)
        ) / 6.0
    return s


def cube_mesh(drop_face: Int) raises -> Tuple[Tensor[F32], IntMatrix]:
    """Closed 0.4-cube centered at origin, outward winding; drop_face
    (0-11, -1 = none) removes one triangle (the puncture case)."""
    var verts = Tensor[F32]([8, 3])
    var k = 0
    for z in range(2):
        for y in range(2):
            for x in range(2):
                verts.data[k * 3 + 0] = Float32(x) * 0.4 - 0.2
                verts.data[k * 3 + 1] = Float32(y) * 0.4 - 0.2
                verts.data[k * 3 + 2] = Float32(z) * 0.4 - 0.2
                k += 1
    var crows: List[Int] = [
        0, 2, 1, 1, 2, 3,  # z-
        4, 5, 6, 5, 7, 6,  # z+
        0, 1, 4, 1, 5, 4,  # y-
        2, 6, 3, 3, 6, 7,  # y+
        0, 4, 2, 2, 4, 6,  # x-
        1, 3, 5, 3, 7, 5,  # x+
    ]
    var rows = List[Int]()
    for f in range(12):
        if f == drop_face:
            continue
        rows.append(crows[f * 3 + 0])
        rows.append(crows[f * 3 + 1])
        rows.append(crows[f * 3 + 2])
    return (verts^, make_faces(rows))


def dist_to_cube_surface(x: Float64, y: Float64, z: Float64) -> Float64:
    """Unsigned distance to the closed 0.4-cube SURFACE (exact for the
    axis-aligned box |x|,|y|,|z| <= 0.2)."""
    var qx = x
    if qx < 0:
        qx = -qx
    var qy = y
    if qy < 0:
        qy = -qy
    var qz = z
    if qz < 0:
        qz = -qz
    var dx = qx - 0.2
    var dy = qy - 0.2
    var dz = qz - 0.2
    if dx <= 0 and dy <= 0 and dz <= 0:
        # inside: distance to nearest wall
        var m = -dx
        if -dy < m:
            m = -dy
        if -dz < m:
            m = -dz
        return m
    var ox = dx
    if ox < 0:
        ox = 0
    var oy = dy
    if oy < 0:
        oy = 0
    var oz = dz
    if oz < 0:
        oz = 0
    return sqrt(ox * ox + oy * oy + oz * oz)


def main() raises:
    var R = 32
    var band = 1.0
    var scale = (Float64(R) + 3.0 * band) / Float64(R) * 1.0
    var eps = band * scale / Float64(R)

    # 1) closed cube -> watertight double shell, consistent winding,
    # positive signed volume (orientation), verts near the surface
    var c1 = cube_mesh(-1)
    var r1 = remesh_narrow_band_dc(c1[0], c1[1], 0, 0, 0, scale, R, band, 0.9)
    if r1[0].shape[0] == 0 or r1[1].rows == 0:
        raise Error("cube remesh: empty output")
    if boundary_edges(r1[1], r1[0].shape[0]) != 0:
        raise Error("cube remesh: not watertight")
    check_consistent_winding(r1[1], r1[0].shape[0])
    var vol = signed_volume(r1[0], r1[1])
    if vol <= 0:
        raise Error("cube remesh: signed volume must be positive (orientation)")
    if vol >= 0.47 * 0.47 * 0.47:
        raise Error("cube remesh: shell volume larger than outer box")
    var worst: Float64 = 0
    for i in range(r1[0].shape[0]):
        var d = dist_to_cube_surface(
            Float64(r1[0].data[i * 3 + 0]),
            Float64(r1[0].data[i * 3 + 1]),
            Float64(r1[0].data[i * 3 + 2]),
        )
        if d > worst:
            worst = d
    # projected 90% back from ~eps: expect ~0.1*eps + DC error
    if worst > 0.75 * eps:
        raise Error("cube remesh: vertices too far from surface: " + String(worst))
    print(
        "  cube: watertight shell,", r1[0].shape[0], "V /", r1[1].rows,
        "F, signed vol", vol, ", max surface dist", worst,
    )

    # 2) punctured cube (one triangle removed -> open input) STILL
    # remeshes watertight — the crack-swallowing property the whole
    # branch exists for
    var c2 = cube_mesh(3)
    var r2 = remesh_narrow_band_dc(c2[0], c2[1], 0, 0, 0, scale, R, band, 0.9)
    if boundary_edges(r2[1], r2[0].shape[0]) != 0:
        raise Error("punctured cube: remesh must be watertight")
    check_consistent_winding(r2[1], r2[0].shape[0])
    if signed_volume(r2[0], r2[1]) <= 0:
        raise Error("punctured cube: orientation flipped")
    print(
        "  punctured cube: watertight (", r2[0].shape[0], "V /",
        r2[1].rows, "F ) — crack swallowed",
    )

    # 3) determinism
    var c3 = cube_mesh(3)
    var r3 = remesh_narrow_band_dc(c3[0], c3[1], 0, 0, 0, scale, R, band, 0.9)
    if r3[0].shape[0] != r2[0].shape[0] or r3[1].rows != r2[1].rows:
        raise Error("determinism: sizes differ")
    for i in range(len(r2[0].data)):
        if r2[0].data[i] != r3[0].data[i]:
            raise Error("determinism: vertex data differs")
    for i in range(len(r2[1].data)):
        if r2[1].data[i] != r3[1].data[i]:
            raise Error("determinism: face data differs")
    print("  deterministic across reruns")

    # 4) no projection: verts sit on the offset surface ~eps out
    var c4 = cube_mesh(-1)
    var r4 = remesh_narrow_band_dc(c4[0], c4[1], 0, 0, 0, scale, R, band, 0.0)
    var worst4: Float64 = 1e30
    var best4: Float64 = 0
    for i in range(r4[0].shape[0]):
        var d = dist_to_cube_surface(
            Float64(r4[0].data[i * 3 + 0]),
            Float64(r4[0].data[i * 3 + 1]),
            Float64(r4[0].data[i * 3 + 2]),
        )
        if d < worst4:
            worst4 = d
        if d > best4:
            best4 = d
    if worst4 < 0.25 * eps or best4 > 2.5 * eps:
        raise Error(
            "offset surface: expected ~eps from surface, got ["
            + String(worst4) + ", " + String(best4) + "] vs eps " + String(eps)
        )
    print("  unprojected shell sits at ~eps (", worst4, "-", best4, ", eps", eps, ")")

    # 5) empty input -> empty output
    var ev = Tensor[F32]([0, 3])
    var ef = IntMatrix(0, 3)
    var r5 = remesh_narrow_band_dc(ev, ef, 0, 0, 0, scale, R, band, 0.9)
    if r5[0].shape[0] != 0 or r5[1].rows != 0:
        raise Error("empty input must give empty output")
    print("  empty input -> empty output")

    print("wp18 remesh: cube + punctured cube + determinism + offset + empty passed")
