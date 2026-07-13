# WP5 parity: full/windowed attention kernels + SparseMultiHeadAttention +
# dense MultiHeadAttention vs the torch originals.
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_wp5_vs_torch.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.modules.nn import SparseLinear
from trellis2_mojo.modules.attention import MultiHeadAttention
from trellis2_mojo.sparse.attention.full_attn import (
    sparse_sdpa_qkv,
    sparse_sdpa_q_kv,
    sparse_sdpa_q_kv_dense,
    sparse_sdpa_q_k_v,
    dense_sdpa_qkv,
    dense_sdpa_q_kv,
    dense_sdpa_q_k_v,
)
from trellis2_mojo.sparse.attention.windowed_attn import sparse_windowed_sdpa_self
from trellis2_mojo.sparse.attention.modules import (
    SparseMultiHeadAttention,
    ATTN_MODE_FULL,
    ATTN_MODE_DOUBLE_WINDOWED,
)
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32
comptime HEADS = 2
comptime HDIM = 8
comptime CH = 16


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64 = 5e-5) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error("value mismatch: " + name + " @ " + String(i) + " diff " + String(d))


def lin(r: PythonObject, prefix: String) raises -> SparseLinear:
    return SparseLinear(tensor_from_torch(r[prefix + "_w"]), tensor_from_torch(r[prefix + "_b"]))


def dummy_lin() raises -> SparseLinear:
    return SparseLinear(Tensor[F32]([1, 1]), Tensor[F32]([1]))


def sparse_of(r: PythonObject, feats_key: String, coords_key: String) raises -> SparseTensor[F32]:
    return SparseTensor[F32](tensor_from_torch(r[feats_key]), intmatrix_from_torch(r[coords_key]))


def py_int_list(l: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for v in l:
        out.append(v)
    return out


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp5")

    for seed in range(3):
        # -- kernels: sparse variants
        var r1 = pyref.ref_full_qkv(seed)
        check_tensor("full_qkv", sparse_sdpa_qkv(sparse_of(r1, "qkv", "coords")).vl.feats, r1["out"])

        var r2 = pyref.ref_q_kv_sparse(seed)
        check_tensor(
            "q_kv_sparse",
            sparse_sdpa_q_kv(sparse_of(r2, "qf", "qc"), sparse_of(r2, "kvf", "kvc")).vl.feats,
            r2["out"],
        )

        var r3 = pyref.ref_q_kv_dense(seed)
        check_tensor(
            "q_kv_dense",
            sparse_sdpa_q_kv_dense(sparse_of(r3, "qf", "qc"), tensor_from_torch(r3["kv"])).vl.feats,
            r3["out"],
        )

        var r4 = pyref.ref_q_k_v_sparse(seed)
        check_tensor(
            "q_k_v_sparse",
            sparse_sdpa_q_k_v(
                sparse_of(r4, "qf", "qc"), sparse_of(r4, "kf", "kvc"), sparse_of(r4, "vf", "kvc")
            ).vl.feats,
            r4["out"],
        )

        # -- kernels: dense variants
        var r5 = pyref.ref_dense_sdpa(seed)
        check_tensor("dense_qkv", dense_sdpa_qkv(tensor_from_torch(r5["qkv"])), r5["out_qkv"])
        check_tensor(
            "dense_q_kv",
            dense_sdpa_q_kv(tensor_from_torch(r5["q"]), tensor_from_torch(r5["kv"])),
            r5["out_q_kv"],
        )
        check_tensor(
            "dense_q_k_v",
            dense_sdpa_q_k_v(
                tensor_from_torch(r5["q"]), tensor_from_torch(r5["k"]), tensor_from_torch(r5["v"])
            ),
            r5["out_q_k_v"],
        )

        # -- windowed kernel: plain and shifted, plus cache hit
        var shifts: List[List[Int]] = [[0, 0, 0], [1, 1, 1]]
        for si in range(2):
            var rw = pyref.ref_windowed(seed, 2, py_int_list(shifts[si]))
            var xw = sparse_of(rw, "qkv", "coords")
            check_tensor(
                "windowed(shift=" + String(si) + ")",
                sparse_windowed_sdpa_self(xw, 2, shifts[si]).vl.feats,
                rw["out"],
            )
            check_tensor(
                "windowed_cached(shift=" + String(si) + ")",
                sparse_windowed_sdpa_self(xw, 2, shifts[si]).vl.feats,
                rw["out"],
            )

        # -- SparseMultiHeadAttention: self full, plain / rms / rope / rms+rope
        for cfg in range(4):
            var use_rms = cfg == 1 or cfg == 3
            var use_rope = cfg == 2 or cfg == 3
            var rm = pyref.ref_sparse_mha_self(seed, use_rms, use_rope)
            var mha = SparseMultiHeadAttention(
                CH, HEADS, lin(rm, "to_qkv"), dummy_lin(), dummy_lin(), lin(rm, "to_out"),
                attn_mode=ATTN_MODE_FULL, use_rope=use_rope, qk_rms_norm=use_rms,
            )
            if use_rms:
                mha.q_rms_norm.gamma = tensor_from_torch(rm["gamma_q"])
                mha.k_rms_norm.gamma = tensor_from_torch(rm["gamma_k"])
            check_tensor(
                "mha_self(cfg=" + String(cfg) + ")",
                mha.forward(sparse_of(rm, "feats", "coords")).vl.feats,
                rm["out"],
            )

        # -- SparseMultiHeadAttention: double windowed
        var rdw = pyref.ref_sparse_mha_double_windowed(seed)
        var mha_dw = SparseMultiHeadAttention(
            CH, HEADS, lin(rdw, "to_qkv"), dummy_lin(), dummy_lin(), lin(rdw, "to_out"),
            attn_mode=ATTN_MODE_DOUBLE_WINDOWED, window_size=2,
        )
        check_tensor(
            "mha_double_windowed",
            mha_dw.forward(sparse_of(rdw, "feats", "coords")).vl.feats,
            rdw["out"],
        )

        # -- SparseMultiHeadAttention: cross with dense context
        for rms_i in range(2):
            var rc = pyref.ref_sparse_mha_cross(seed, rms_i == 1)
            var mha_c = SparseMultiHeadAttention(
                CH, HEADS, dummy_lin(), lin(rc, "to_q"), lin(rc, "to_kv"), lin(rc, "to_out"),
                qk_rms_norm=rms_i == 1,
            )
            if rms_i == 1:
                mha_c.q_rms_norm.gamma = tensor_from_torch(rc["gamma_q"])
                mha_c.k_rms_norm.gamma = tensor_from_torch(rc["gamma_k"])
            check_tensor(
                "mha_cross(rms=" + String(rms_i) + ")",
                mha_c.forward_cross(sparse_of(rc, "feats", "coords"), tensor_from_torch(rc["ctx"])).vl.feats,
                rc["out"],
            )

        # -- dense MultiHeadAttention: self and cross, with rms
        for rms_i in range(2):
            var rds = pyref.ref_dense_mha(seed, "self", rms_i == 1)
            var dmha = MultiHeadAttention(
                CH, HEADS, lin(rds, "to_qkv"), dummy_lin(), dummy_lin(), lin(rds, "to_out"),
                qk_rms_norm=rms_i == 1,
            )
            if rms_i == 1:
                dmha.q_rms_norm.gamma = tensor_from_torch(rds["gamma_q"])
                dmha.k_rms_norm.gamma = tensor_from_torch(rds["gamma_k"])
            check_tensor(
                "dense_mha_self(rms=" + String(rms_i) + ")",
                dmha.forward(tensor_from_torch(rds["x"])),
                rds["out"],
            )

            var rdc = pyref.ref_dense_mha(seed, "cross", rms_i == 1)
            var dmha_c = MultiHeadAttention(
                CH, HEADS, dummy_lin(), lin(rdc, "to_q"), lin(rdc, "to_kv"), lin(rdc, "to_out"),
                qk_rms_norm=rms_i == 1,
            )
            if rms_i == 1:
                dmha_c.q_rms_norm.gamma = tensor_from_torch(rdc["gamma_q"])
                dmha_c.k_rms_norm.gamma = tensor_from_torch(rdc["gamma_k"])
            check_tensor(
                "dense_mha_cross(rms=" + String(rms_i) + ")",
                dmha_c.forward_cross(tensor_from_torch(rdc["x"]), tensor_from_torch(rdc["ctx"])),
                rdc["out"],
            )

    print("wp5 attention parity vs torch: 3 seeds, all variants passed")
