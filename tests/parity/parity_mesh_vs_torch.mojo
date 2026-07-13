# Mesh-extraction parity (WP9 part 3 step 4): pure-Mojo
# flexible_dual_grid_to_mesh vs the vendored o_voxel stub (identical to
# the CUDA version for inference), on dense-block + scattered voxel cases,
# both split modes (split_weight = pipeline path, normal alignment =
# completeness), degenerate empties, and an OBJ writer round-trip
# verified by the Python side.
#
# Run from repo root: pixi run test-mesh

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, tensor_to_torch, intmatrix_from_torch
from trellis2_mojo.meshing.fdg_mesh import flexible_dual_grid_to_mesh, write_obj
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def run_case(r: PythonObject, name: String, use_split: Bool) raises -> Tuple[Tensor[F32], IntMatrix]:
    var coords = intmatrix_from_torch(r["coords"])
    var dual = tensor_from_torch(r["dual"])
    var flags = tensor_from_torch(r["flags"])
    var sw: Tensor[F32]
    if use_split:
        sw = tensor_from_torch(r["sw"])
    else:
        var empty: List[Int] = [0, 1]
        sw = Tensor[F32](empty)
    var aabb_min: List[Float64] = [-0.5, -0.5, -0.5]
    var aabb_max: List[Float64] = [0.5, 0.5, 0.5]
    var grid = Int(py=r["grid"])
    var pair = flexible_dual_grid_to_mesh(
        coords, dual, flags, sw, aabb_min, aabb_max, grid
    )

    # vertices: float compare; triangles: exact
    var rv = r["vertices"]
    var rt = r["triangles"]
    if Int(py=rv.shape[0]) != pair[0].shape[0]:
        raise Error("vertex count mismatch: " + name)
    if Int(py=rt.shape[0]) != pair[1].rows:
        raise Error(
            "triangle count mismatch: " + name + " mojo=" + String(pair[1].rows)
            + " torch=" + String(Int(py=rt.shape[0]))
        )
    var maxdiff: Float64 = 0.0
    if pair[0].numel() > 0:
        var mt = tensor_to_torch(pair[0])
        maxdiff = Float64(py=(mt - rv).abs().max().item())
        if maxdiff > 1e-6:
            raise Error("vertex mismatch: " + name + " diff " + String(maxdiff))
    var rt_flat = rt.flatten().tolist()
    var k = 0
    for row in range(pair[1].rows):
        for c in range(3):
            if Int(py=rt_flat[k]) != pair[1].at(row, c):
                raise Error("triangle mismatch: " + name + " @ row " + String(row))
            k += 1
    print(
        "  " + name + ": V=" + String(pair[0].shape[0]) + " F=" + String(pair[1].rows)
        + " max|dv| " + String(maxdiff)
    )
    return pair^


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_mesh")

    var have_obj_case = False
    var obj_ref: PythonObject = Python.none()
    var obj_pair: Tuple[Tensor[F32], IntMatrix] = (Tensor[F32]([1]), IntMatrix(0, 3))

    for seed in range(3):
        # pipeline path: quad split by split_weight
        var r1 = pyref.ref_mesh(seed, 12, 4, 30, 0.45, True)
        var pair1 = run_case(r1, "mesh(sw, seed " + String(seed) + ")", True)
        if not have_obj_case and pair1[1].rows > 0:
            obj_ref = r1
            obj_pair = pair1^
            have_obj_case = True
        # normal-alignment branch (split_weight=None upstream)
        var r2 = pyref.ref_mesh(seed, 12, 4, 30, 0.45, False)
        _ = run_case(r2, "mesh(align, seed " + String(seed) + ")", False)

    # degenerates: no flags at all / flags but only isolated voxels
    _ = run_case(pyref.ref_mesh_degenerate("noflags"), "mesh(noflags)", True)
    _ = run_case(pyref.ref_mesh_degenerate("noquads"), "mesh(noquads)", True)

    # OBJ writer round-trip (Python re-reads and compares)
    if not have_obj_case:
        raise Error("no case produced triangles — cannot test write_obj")
    var obj_path: String = "/tmp/trellis_mojo_test_mesh.obj"
    write_obj(obj_path, obj_pair[0], obj_pair[1])
    _ = pyref.check_obj(obj_path, obj_ref["vertices"], obj_ref["triangles"])

    print("mesh parity vs o_voxel-stub: 3 seeds x 2 split-moduser + degenerater + OBJ-roundtrip passed")
