# WP6 parity: SparseDownsample/Upsample, Spatial2Channel/Channel2Spatial
# and conv_none SparseConv3d vs the torch originals — including the cache
# handshake between the down/up pairs and scale bookkeeping.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_wp6_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.conv import SparseConv3d
from trellis2_mojo.sparse.spatial.basic import SparseDownsample, SparseUpsample, POOL_MEAN, POOL_MAX
from trellis2_mojo.sparse.spatial.spatial2channel import SparseSpatial2Channel, SparseChannel2Spatial
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix, Frac

comptime F32 = DType.float32


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64 = 2e-5) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error("value mismatch: " + name + " @ " + String(i) + " diff " + String(d))


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


def sparse_of(r: PythonObject, feats_key: String, coords_key: String) raises -> SparseTensor[F32]:
    return SparseTensor[F32](tensor_from_torch(r[feats_key]), intmatrix_from_torch(r[coords_key]))


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp6")

    for seed in range(3):
        # -- down + up through the shared cache, mean and max
        var modes: List[Int] = [POOL_MEAN, POOL_MAX]
        var mode_names: List[String] = ["mean", "max"]
        for mi in range(2):
            var rd = pyref.ref_down_up(seed, mode_names[mi])
            var x = sparse_of(rd, "feats", "coords")
            var y = SparseDownsample(2, modes[mi]).forward(x)
            check_tensor("down_feats(" + mode_names[mi] + ")", y.vl.feats, rd["down_feats"])
            check_intmat("down_coords(" + mode_names[mi] + ")", y.coords, rd["down_coords"])
            if not (y.scale[0] == Frac(2, 1)):
                raise Error("downsample scale not updated")
            var z = SparseUpsample(2).forward(y)
            check_tensor("up_feats(" + mode_names[mi] + ")", z.vl.feats, rd["up_feats"])
            check_intmat("up_coords(" + mode_names[mi] + ")", z.coords, rd["coords"])
            if not (z.scale[0] == Frac(1, 1)):
                raise Error("upsample scale not restored")

        # -- upsample via explicit subdivision tensor
        var ru = pyref.ref_up_subdivision(seed)
        var xu = sparse_of(ru, "feats", "coords")
        var sub = xu.replace(tensor_from_torch(ru["sub"]))
        var zu = SparseUpsample(2).forward_subdivision(xu, sub)
        check_tensor("up_subdiv_feats", zu.vl.feats, ru["out_feats"])
        check_intmat("up_subdiv_coords", zu.coords, ru["out_coords"])

        # -- spatial2channel + channel2spatial roundtrip
        var rs = pyref.ref_s2c_c2s(seed)
        var xs = sparse_of(rs, "feats", "coords")
        var ys = SparseSpatial2Channel(2).forward(xs)
        check_tensor("s2c_feats", ys.vl.feats, rs["s2c_feats"])
        check_intmat("s2c_coords", ys.coords, rs["s2c_coords"])
        var zs = SparseChannel2Spatial(2).forward(ys)
        check_tensor("c2s_feats", zs.vl.feats, rs["c2s_feats"])
        check_intmat("c2s_coords", zs.coords, rs["coords"])

        # -- conv_none: kernel 3 / kernel 1 / dilation 2, incl. cache reuse
        var kss: List[Int] = [3, 1, 3]
        var dils: List[Int] = [1, 1, 2]
        for ci in range(3):
            var rc = pyref.ref_conv(seed, kss[ci], dils[ci])
            var xc = sparse_of(rc, "feats", "coords")
            var conv = SparseConv3d(tensor_from_torch(rc["w"]), tensor_from_torch(rc["b"]), dilation=dils[ci])
            var name = "conv(ks=" + String(kss[ci]) + ",d=" + String(dils[ci]) + ")"
            check_tensor(name, conv.forward(xc).vl.feats, rc["out"])
            check_tensor(name + "_cached", conv.forward(xc).vl.feats, rc["out"])

    print("wp6 parity vs torch: 3 seeds, spatial + conv passed")
