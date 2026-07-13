# WP15 parity (fase 2, ADR 0008): pure-Mojo trilinear grid sampling
# (meshing/vertex_attrs.mojo) vs a plain-torch reimplementation of the
# flex_gemm formula (tests/parity/torch_ref_wp15.py), area-weighted vertex
# normals vs a torch index_add reference, the GLB writer round-trip
# (dependency-free Python reader asserts BIT-identical payloads), and the
# upstream axis-swap helper.
#
# Run from repo root: pixi run test-wp15

from std.python import Python, PythonObject

from trellis2_mojo.interop import (
    intmatrix_from_torch,
    tensor_from_torch,
    tensor_to_torch,
)
from trellis2_mojo.io.glb import to_glb_axes, write_glb
from trellis2_mojo.meshing.vertex_attrs import (
    grid_sample_trilinear,
    vertex_normals,
)
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def max_diff(a: Tensor[F32], b: Tensor[F32]) raises -> Float32:
    if len(a.data) != len(b.data):
        raise Error("shape mismatch in max_diff")
    var m: Float32 = 0
    for i in range(len(a.data)):
        var d = abs(a.data[i] - b.data[i])
        if d > m:
            m = d
    return m


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp15")
    var torch = Python.import_module("torch")

    # --- trilinear sampling vs the flex_gemm-formula reference ---
    for seed in range(3):
        var r = pyref.ref_sample(seed, 16, 300, 200)
        var coords = intmatrix_from_torch(r["coords"])
        var feats = tensor_from_torch(r["feats"])
        var query = tensor_from_torch(r["query"])
        var want = tensor_from_torch(r["out"])
        var got = grid_sample_trilinear(feats, coords, Int(py=r["grid"]), query)
        var d = max_diff(got, want)
        print("  trilinear seed", seed, " grid 16 n 300 l 200  max|diff|:", d)
        if d > 1e-6:
            raise Error("trilinear sampling parity failed")

    # empty volume: every query must return exactly 0 (0 / 1e-12 path)
    var e_feats = Tensor[F32]([0, 6])
    var e_coords = IntMatrix(0, 3)
    var e_query = Tensor[F32]([4, 3], fill=2.5)
    var e_out = grid_sample_trilinear(e_feats, e_coords, 8, e_query)
    for i in range(len(e_out.data)):
        if e_out.data[i] != 0:
            raise Error("empty-volume sampling must be exactly 0")
    print("  empty volume -> exact zeros")

    # --- vertex normals vs torch index_add reference ---
    var rn = pyref.ref_normals(7, 200, 400)
    var verts = tensor_from_torch(rn["verts"])
    var faces = intmatrix_from_torch(rn["faces"])
    var want_n = tensor_from_torch(rn["normals"])
    var got_n = vertex_normals(verts, faces)
    var dn = max_diff(got_n, want_n)
    print("  vertex normals V 200 F 400  max|diff|:", dn)
    if dn > 1e-5:
        raise Error("vertex normals parity failed")
    # every normal must be unit length (glTF validator requirement —
    # cancelled fill-centroid normals used to ship as zero and shaded
    # like black micro holes)
    for i in range(got_n.shape[0]):
        var l2 = (
            got_n.data[i * 3] * got_n.data[i * 3]
            + got_n.data[i * 3 + 1] * got_n.data[i * 3 + 1]
            + got_n.data[i * 3 + 2] * got_n.data[i * 3 + 2]
        )
        if abs(l2 - 1.0) > 1e-3:
            raise Error("non-unit vertex normal at " + String(i))
    # fully cancelling fan (two opposite triangles): neighborhood cancels
    # too -> deterministic +z fallback on all three vertices
    var cv = Tensor[F32]([3, 3])
    cv.data[3] = 1.0
    cv.data[7] = 1.0
    var cf = IntMatrix(2, 3)
    cf.set(0, 0, 0)
    cf.set(0, 1, 1)
    cf.set(0, 2, 2)
    cf.set(1, 0, 0)
    cf.set(1, 1, 2)
    cf.set(1, 2, 1)
    var cn = vertex_normals(cv, cf)
    for i in range(3):
        if cn.data[i * 3] != 0 or cn.data[i * 3 + 1] != 0 or cn.data[i * 3 + 2] != 1:
            raise Error("cancelled fan must fall back to +z")
    print("  cancelled-fan normals fall back to unit +z")

    # --- axis swap helper: y,z -> z,-y (upstream to_glb) ---
    var ax = Tensor[F32]([1, 3])
    ax.data[0] = 1.0
    ax.data[1] = 2.0
    ax.data[2] = 3.0
    to_glb_axes(ax)
    if ax.data[0] != 1.0 or ax.data[1] != 3.0 or ax.data[2] != -2.0:
        raise Error("to_glb_axes swap wrong")
    print("  to_glb_axes: (1,2,3) -> (1,3,-2)")

    # --- GLB writer round-trip (bit-identical payload) ---
    # colors deliberately include out-of-range values to exercise the
    # [0,1] clamp; the Python side clamps before comparing
    var g = torch.Generator()
    _ = g.manual_seed(99)
    var col_t = torch.rand([200, 4], generator=g) * 1.4 - 0.2
    var colors = tensor_from_torch(col_t)
    var glb_path = String("/tmp/wp15_roundtrip.glb")
    write_glb(glb_path, verts, got_n, colors, faces, 0.25, 0.75)
    var res = pyref.check_glb(
        glb_path,
        tensor_to_torch(verts),
        tensor_to_torch(got_n),
        torch.clamp(col_t, 0.0, 1.0),
        rn["faces"],
        0.25,
        0.75,
    )
    print(
        "  glb round-trip: V", Int(py=res["V"]), " indices", Int(py=res["I"]),
        " bit-identical + material/min/max checked",
    )

    print("wp15 parity vs torch: 3 trilinear seeds + empty volume + normals + axis swap + glb round-trip passed")
