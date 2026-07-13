# WP4 parity: linear, LayerNorm32/GroupNorm32/ChannelLayerNorm32,
# relu/silu/gelu, modulate and RoPE vs the torch originals.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_wp4_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.modules.nn import (
    SparseLinear,
    LayerNorm32,
    GroupNorm32,
    ChannelLayerNorm32,
    activation,
    modulate,
    ACT_RELU,
    ACT_SILU,
    ACT_GELU,
)
from trellis2_mojo.sparse.attention.rope import SparseRotaryPositionEmbedder
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor

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


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp4")

    for seed in range(5):
        # SparseLinear
        var rl = pyref.ref_linear(seed)
        var x = SparseTensor[F32](tensor_from_torch(rl["feats"]), intmatrix_from_torch(rl["coords"]))
        var lin = SparseLinear(tensor_from_torch(rl["w"]), tensor_from_torch(rl["b"]))
        check_tensor("linear", lin.forward(x).vl.feats, rl["out"])

        # LayerNorm32, affine and plain
        for affine_i in range(2):
            var affine = affine_i == 1
            var rn = pyref.ref_layernorm(seed, affine)
            var xt = tensor_from_torch(rn["x"])
            var ln = LayerNorm32(xt.shape[1], eps=1e-6, affine=affine)
            if affine:
                ln.weight = tensor_from_torch(rn["w"])
                ln.bias = tensor_from_torch(rn["b"])
            check_tensor("layernorm(affine=" + String(affine) + ")", ln.forward(xt), rn["out"])

        # GroupNorm32 on dense [N, C, D, H, W]
        var rg = pyref.ref_groupnorm(seed)
        var gn = GroupNorm32(4, 8)
        gn.weight = tensor_from_torch(rg["w"])
        gn.bias = tensor_from_torch(rg["b"])
        check_tensor("groupnorm", gn.forward(tensor_from_torch(rg["x"])), rg["out"])

        # ChannelLayerNorm32
        var rc = pyref.ref_channel_layernorm(seed)
        var cln = ChannelLayerNorm32(8, eps=1e-6)
        cln.inner.weight = tensor_from_torch(rc["w"])
        cln.inner.bias = tensor_from_torch(rc["b"])
        check_tensor("channel_layernorm", cln.forward(tensor_from_torch(rc["x"])), rc["out"])

        # nonlinearities
        var rn2 = pyref.ref_nonlin(seed)
        var nx = tensor_from_torch(rn2["x"])
        check_tensor("relu", activation(nx, ACT_RELU), rn2["relu"])
        check_tensor("silu", activation(nx, ACT_SILU), rn2["silu"])
        check_tensor("gelu", activation(nx, ACT_GELU), rn2["gelu"])

        # modulate
        var rm = pyref.ref_modulate(seed)
        check_tensor(
            "modulate",
            modulate(
                tensor_from_torch(rm["x"]),
                tensor_from_torch(rm["shift"]),
                tensor_from_torch(rm["scale"]),
            ),
            rm["out"],
        )

        # RoPE: head_dim 12 (no padding) and 8 (padded phases)
        var head_dims: List[Int] = [12, 8]
        for hd in head_dims:
            var rr = pyref.ref_rope(seed, hd)
            var q = SparseTensor[F32](tensor_from_torch(rr["qf"]), intmatrix_from_torch(rr["coords"]))
            var k = q.replace(tensor_from_torch(rr["kf"]))
            var rope = SparseRotaryPositionEmbedder(hd)
            var qk = rope.embed(q, k)
            check_tensor("rope_q(hd=" + String(hd) + ")", qk[0].vl.feats, rr["q_out"])
            check_tensor("rope_k(hd=" + String(hd) + ")", qk[1].vl.feats, rr["k_out"])
            # second call must hit the phase cache and give identical output
            var qk2 = rope.embed(q, k)
            check_tensor("rope_q_cached(hd=" + String(hd) + ")", qk2[0].vl.feats, rr["q_out"])

    print("wp4 parity vs torch: 5 seeds, all layers passed")
