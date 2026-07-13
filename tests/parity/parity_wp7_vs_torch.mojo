# WP7 parity: all eight transformer block variants + AbsolutePositionEmbedder
# vs the torch originals, with real randomized weights loaded through
# trellis2_mojo/loaders.mojo.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_wp7_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.loaders import (
    lin_from,
    ln_from,
    sparse_mha_from,
    dense_mha_from,
    sparse_ffn_from,
    dense_ffn_from,
    modulation_from,
)
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.transformer.blocks import SparseTransformerBlock, SparseTransformerCrossBlock
from trellis2_mojo.sparse.transformer.modulated import (
    ModulatedSparseTransformerBlock,
    ModulatedSparseTransformerCrossBlock,
)
from trellis2_mojo.modules.transformer.blocks import (
    AbsolutePositionEmbedder,
    TransformerBlock,
    TransformerCrossBlock,
)
from trellis2_mojo.modules.transformer.modulated import (
    ModulatedTransformerBlock,
    ModulatedTransformerCrossBlock,
)

comptime F32 = DType.float32
comptime HEADS = 2
comptime CH = 16


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64 = 1e-4) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error("value mismatch: " + name + " @ " + String(i) + " diff " + String(d))


def sparse_of(r: PythonObject) raises -> SparseTensor[F32]:
    return SparseTensor[F32](tensor_from_torch(r["feats"]), intmatrix_from_torch(r["coords"]))


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp7")

    for seed in range(3):
        # -- sparse plain block
        var r1 = pyref.ref_sparse_block(seed)
        var sd = r1["sd"]
        var b1 = SparseTransformerBlock(
            ln_from(sd, "norm1", CH), ln_from(sd, "norm2", CH),
            sparse_mha_from(sd, "attn", CH, HEADS),
            sparse_ffn_from(sd, "mlp"),
        )
        check_tensor("sparse_block", b1.forward(sparse_of(r1)).vl.feats, r1["out"])

        # -- sparse cross block, ln_affine both ways
        for affine_i in range(2):
            var affine = affine_i == 1
            var r2 = pyref.ref_sparse_cross_block(seed, affine)
            sd = r2["sd"]
            var b2 = SparseTransformerCrossBlock(
                ln_from(sd, "norm1", CH, affine=affine),
                ln_from(sd, "norm2", CH, affine=affine),
                ln_from(sd, "norm3", CH, affine=affine),
                sparse_mha_from(sd, "self_attn", CH, HEADS),
                sparse_mha_from(sd, "cross_attn", CH, HEADS, is_cross=True),
                sparse_ffn_from(sd, "mlp"),
            )
            check_tensor(
                "sparse_cross(affine=" + String(affine) + ")",
                b2.forward(sparse_of(r2), tensor_from_torch(r2["ctx"])).vl.feats,
                r2["out"],
            )

        # -- modulated sparse block, adaLN and share_mod
        for share_i in range(2):
            var share = share_i == 1
            var r3 = pyref.ref_mod_sparse_block(seed, share)
            sd = r3["sd"]
            var b3 = ModulatedSparseTransformerBlock(
                CH,
                ln_from(sd, "norm1", CH), ln_from(sd, "norm2", CH),
                sparse_mha_from(sd, "attn", CH, HEADS),
                sparse_ffn_from(sd, "mlp"),
                modulation_from(sd, share),
            )
            check_tensor(
                "mod_sparse(share=" + String(share) + ")",
                b3.forward(sparse_of(r3), tensor_from_torch(r3["mod"])).vl.feats,
                r3["out"],
            )

        # -- modulated sparse cross block (norm2 affine)
        var r4 = pyref.ref_mod_sparse_cross_block(seed)
        sd = r4["sd"]
        var b4 = ModulatedSparseTransformerCrossBlock(
            CH,
            ln_from(sd, "norm1", CH),
            ln_from(sd, "norm2", CH, affine=True),
            ln_from(sd, "norm3", CH),
            sparse_mha_from(sd, "self_attn", CH, HEADS),
            sparse_mha_from(sd, "cross_attn", CH, HEADS, is_cross=True),
            sparse_ffn_from(sd, "mlp"),
            modulation_from(sd, False),
        )
        check_tensor(
            "mod_sparse_cross",
            b4.forward(sparse_of(r4), tensor_from_torch(r4["mod"]), tensor_from_torch(r4["ctx"])).vl.feats,
            r4["out"],
        )

        # -- dense plain + cross
        var r5 = pyref.ref_dense_block(seed)
        sd = r5["sd"]
        var b5 = TransformerBlock(
            ln_from(sd, "norm1", CH, affine=True), ln_from(sd, "norm2", CH, affine=True),
            dense_mha_from(sd, "attn", CH, HEADS),
            dense_ffn_from(sd, "mlp"),
        )
        check_tensor("dense_block", b5.forward(tensor_from_torch(r5["x"])), r5["out"])

        var r6 = pyref.ref_dense_cross_block(seed)
        sd = r6["sd"]
        var b6 = TransformerCrossBlock(
            ln_from(sd, "norm1", CH, affine=True),
            ln_from(sd, "norm2", CH, affine=True),
            ln_from(sd, "norm3", CH, affine=True),
            dense_mha_from(sd, "self_attn", CH, HEADS),
            dense_mha_from(sd, "cross_attn", CH, HEADS, is_cross=True),
            dense_ffn_from(sd, "mlp"),
        )
        check_tensor(
            "dense_cross",
            b6.forward(tensor_from_torch(r6["x"]), tensor_from_torch(r6["ctx"])),
            r6["out"],
        )

        # -- modulated dense, adaLN and share_mod
        for share_i in range(2):
            var share = share_i == 1
            var r7 = pyref.ref_mod_dense_block(seed, share)
            sd = r7["sd"]
            var b7 = ModulatedTransformerBlock(
                CH,
                ln_from(sd, "norm1", CH), ln_from(sd, "norm2", CH),
                dense_mha_from(sd, "attn", CH, HEADS),
                dense_ffn_from(sd, "mlp"),
                modulation_from(sd, share),
            )
            check_tensor(
                "mod_dense(share=" + String(share) + ")",
                b7.forward(tensor_from_torch(r7["x"]), tensor_from_torch(r7["mod"])),
                r7["out"],
            )

        # -- modulated dense cross
        var r8 = pyref.ref_mod_dense_cross_block(seed)
        sd = r8["sd"]
        var b8 = ModulatedTransformerCrossBlock(
            CH,
            ln_from(sd, "norm1", CH),
            ln_from(sd, "norm2", CH, affine=True),
            ln_from(sd, "norm3", CH),
            dense_mha_from(sd, "self_attn", CH, HEADS),
            dense_mha_from(sd, "cross_attn", CH, HEADS, is_cross=True),
            dense_ffn_from(sd, "mlp"),
            modulation_from(sd, False),
        )
        check_tensor(
            "mod_dense_cross",
            b8.forward(
                tensor_from_torch(r8["x"]), tensor_from_torch(r8["mod"]), tensor_from_torch(r8["ctx"])
            ),
            r8["out"],
        )

        # -- absolute position embedder
        var r9 = pyref.ref_abs_pos(seed)
        var ape = AbsolutePositionEmbedder(CH)
        check_tensor("abs_pos", ape.forward(tensor_from_torch(r9["pos"])), r9["out"])

    print("wp7 parity vs torch: 3 seeds, 8 block variants + pos embedder passed")
