# WP3 parity fuzz: Mojo SparseTensor/VarLenTensor vs the original PyTorch
# implementation, driven through Python interop (which also exercises the
# WP1 interop path). 20 seeded random cases, exact op-for-op comparison.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_sparse_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.sparse.tensor import (
    Tensor,
    IntMatrix,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    RED_SUM,
    RED_MEAN,
)
from trellis2_mojo.sparse.basic import SparseTensor, VarLenTensor, sparse_cat

comptime F32 = DType.float32


def to_tensor(py: PythonObject) raises -> Tensor[F32]:
    var nd = Int(py=py.dim())
    var shape = List[Int]()
    for i in range(nd):
        shape.append(Int(py=py.shape[i]))
    var flat = py.flatten().tolist()
    var vals = List[Scalar[F32]]()
    for i in range(Int(py=py.numel())):
        vals.append(Float32(Float64(py=flat[i])))
    return Tensor[F32].from_values(shape, vals)


def to_intmat(py: PythonObject) raises -> IntMatrix:
    var rows = Int(py=py.shape[0])
    var cols = Int(py=py.shape[1])
    var flat = py.flatten().tolist()
    var m = IntMatrix(rows, cols)
    var k = 0
    for r in range(rows):
        for c in range(cols):
            m.set(r, c, Int(py=flat[k]))
            k += 1
    return m^


def to_int_list(py: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(Int(py=py.__len__())):
        out.append(Int(py=py[i]))
    return out^


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64 = 1e-5) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error(
                "value mismatch: " + name + " @ " + String(i) + " diff " + String(d)
            )


def check_intmat(name: String, m: IntMatrix, py: PythonObject) raises:
    if Int(py=py.shape[0]) != m.rows or Int(py=py.shape[1]) != m.cols:
        raise Error("shape mismatch: " + name)
    var flat = py.flatten().tolist()
    var k = 0
    for r in range(m.rows):
        for c in range(m.cols):
            if Int(py=flat[k]) != m.at(r, c):
                raise Error("coord mismatch: " + name + " @ row " + String(r))
            k += 1


def run_case(pyref: PythonObject, seed: Int) raises:
    var gen = pyref.gen_case(seed)
    var feats_py = gen[0]
    var coords_py = gen[1]
    var res = pyref.ref_results(feats_py, coords_py, seed)

    var x = SparseTensor[F32](to_tensor(feats_py), to_intmat(coords_py))
    var other = x.replace(to_tensor(res["other_feats"]))

    # elementwise
    check_tensor("add", (x + other).vl.feats, res["add"])
    check_tensor("mul_s", (x * 2.5).vl.feats, res["mul_s"])
    check_tensor("rsub", x.elemwise_scalar(1.5, OP_SUB, reverse=True).vl.feats, res["rsub"])
    check_tensor("div_s", (x / 1.7).vl.feats, res["div_s"])
    check_tensor("neg", (-x).vl.feats, res["neg"])

    # batch broadcast: [B, C] and [B, 1]
    check_tensor(
        "badd_full", x.elemwise_batch(to_tensor(res["batch_full"]), OP_ADD).vl.feats, res["badd_full"]
    )
    check_tensor(
        "bmul_one", x.elemwise_batch(to_tensor(res["batch_one"]), OP_MUL).vl.feats, res["bmul_one"]
    )

    # getitem with permutation: feats, coords and layout
    var perm = to_int_list(res["perm"])
    var y = x[perm]
    check_tensor("get_feats", y.vl.feats, res["get_feats"])
    check_intmat("get_coords", y.coords, res["get_coords"])
    var stops = to_int_list(res["get_stops"])
    for b in range(y.batch_size()):
        if y.vl.offsets[b + 1] != stops[b]:
            raise Error("layout mismatch after getitem @ batch " + String(b))

    # concatenation
    var x2 = SparseTensor[F32](to_tensor(res["feats2"]), to_intmat(res["coords2"]))
    var z0 = sparse_cat([x.copy(), x2^], 0)
    check_tensor("cat0_feats", z0.vl.feats, res["cat0_feats"])
    check_intmat("cat0_coords", z0.coords, res["cat0_coords"])
    var z1 = sparse_cat([x.copy(), other^], 1)
    check_tensor("cat1_feats", z1.vl.feats, res["cat1_feats"])

    # dense conversion
    check_tensor("dense", x.to_dense(), res["dense"])

    # varlen to_dense: original leaves garbage (repeated rows) at masked
    # positions, we zero-pad — so compare only where mask == 1
    var dm = x.vl.to_dense()
    var vld = dm[0].copy()
    var mask = dm[1].copy()
    var vld_py = res["vl_dense"].flatten().tolist()
    var mask_py = res["vl_mask"].flatten().tolist()
    var rs = vld.row_size() // mask.shape[1]  # tail size per (b, l) slot
    for slot in range(mask.numel()):
        if Int(py=mask_py[slot]) != Int(mask.data[slot]):
            raise Error("mask mismatch @ " + String(slot))
        if mask.data[slot] == 0:
            continue
        for k in range(rs):
            var d = Float64(py=vld_py[slot * rs + k]) - Float64(vld.data[slot * rs + k])
            if d < 0:
                d = -d
            if d > 1e-5:
                raise Error("vl_dense mismatch @ slot " + String(slot))

    # segment reductions
    check_tensor("mean_b", x.reduce_batch(RED_MEAN), res["mean_b"])
    check_tensor("sum_b", x.reduce_batch(RED_SUM), res["sum_b"])


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref")

    var n_cases = 20
    for seed in range(n_cases):
        run_case(pyref, seed)

    # SparseTensor.full against the original
    var full_ref = pyref.ref_full()
    var f = SparseTensor[F32].full([0, 0, 0, 2, 1, 3], 2, 3, 0.5)
    check_intmat("full_coords", f.coords, full_ref["coords"])
    check_tensor("full_feats", f.vl.feats, full_ref["feats"])

    print("parity vs torch: all", n_cases, "fuzz cases + full() passed")
