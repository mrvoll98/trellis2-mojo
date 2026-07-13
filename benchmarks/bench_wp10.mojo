# WP10 benchmarks: the Mojo port vs the torch originals on the hot
# inference paths — sparse full/windowed attention, submanifold conv,
# modulated transformer block, and the full FlowEuler CFG-interval
# sampling loop with a small SLat flow model.
#
# Methodology: cases are generated (seeded) on the Python side; both sides
# get one warmup run (which also fills layout/partition/neighbor caches —
# steady-state hot-path numbers), then ITERS timed runs; minimums are
# reported. Case building/interop conversion is NOT timed. The torch
# baseline for full attention is the semantically correct per-batch SDPA
# loop (the original's padded naive fallback is incorrect but vectorized;
# its numbers are printed separately below the table).
#
# Run from repo root: pixi run bench   (mojo run -I . benchmarks/bench_wp10.mojo)

from std.python import Python, PythonObject
from std.time import perf_counter_ns

from trellis2_mojo.interop import tensor_from_torch, intmatrix_from_torch
from trellis2_mojo.loaders import ln_from, sparse_mha_from, sparse_ffn_from, modulation_from
from trellis2_mojo.models.structured_latent_flow import slat_flow_from
from trellis2_mojo.pipelines.image_to_3d import SlatFlowVelocity
from trellis2_mojo.samplers.flow_euler import FlowEulerSampler
from trellis2_mojo.sparse.attention.full_attn import sparse_sdpa_qkv
from trellis2_mojo.sparse.attention.windowed_attn import sparse_windowed_sdpa_self
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.conv import SparseConv3d
from trellis2_mojo.sparse.transformer.modulated import ModulatedSparseTransformerBlock
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32
comptime ITERS = 3
comptime GRID = 16

# sampler config (matches the torch side's constants)
comptime STEPS = 8
comptime RESCALE_T = 3.0
comptime CFG = 5.0
comptime LO = 0.2
comptime HI = 0.9


def fmt(v: Float64) raises -> String:
    if v >= 100.0:
        return String(Int(v + 0.5))
    return String(Float64(Int(v * 100.0 + 0.5)) / 100.0)


def py_min_ms(times: PythonObject) raises -> Float64:
    var n = Int(py=times.__len__())
    var best: Float64 = 1e30
    for i in range(n):
        var v = Float64(py=times[i]) * 1000.0
        if v < best:
            best = v
    return best


def row(name: String, size: String, torch_ms: Float64, mojo_ms: Float64) raises:
    print(
        "| " + name + " | " + size + " | " + fmt(torch_ms) + " | "
        + fmt(mojo_ms) + " | " + fmt(mojo_ms / torch_ms) + "x |"
    )


def sparse_of(c: PythonObject, feats_key: String) raises -> SparseTensor[F32]:
    return SparseTensor[F32](tensor_from_torch(c[feats_key]), intmatrix_from_torch(c["coords"]))


def bench_attn(x: SparseTensor[F32]) raises -> Float64:
    _ = sparse_sdpa_qkv(x)
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = sparse_sdpa_qkv(x)
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if dt < best:
            best = dt
    return best


def bench_windowed(x: SparseTensor[F32], window: Int) raises -> Float64:
    var shift: List[Int] = [0, 0, 0]
    _ = sparse_windowed_sdpa_self(x, window, shift)
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = sparse_windowed_sdpa_self(x, window, shift)
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if dt < best:
            best = dt
    return best


def bench_conv(conv: SparseConv3d, x: SparseTensor[F32]) raises -> Float64:
    _ = conv.forward(x)
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = conv.forward(x)
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if dt < best:
            best = dt
    return best


def bench_block(
    block: ModulatedSparseTransformerBlock, x: SparseTensor[F32], mod: Tensor[F32]
) raises -> Float64:
    _ = block.forward(x, mod)
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = block.forward(x, mod)
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if dt < best:
            best = dt
    return best


def bench_sampler(vel: SlatFlowVelocity, noise: Tensor[F32]) raises -> Float64:
    var sampler = FlowEulerSampler(1e-5)
    _ = sampler.sample_cfg_interval(vel, noise, STEPS, RESCALE_T, CFG, LO, HI)
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = sampler.sample_cfg_interval(vel, noise, STEPS, RESCALE_T, CFG, LO, HI)
        var dt = Float64(perf_counter_ns() - t0) / 1e6
        if dt < best:
            best = dt
    return best


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("benchmarks.bench_torch_ref")

    print("machine:", pyref.machine_info())
    print("iters:", ITERS, "(min reported), warmup 1, conversion untimed")
    print()
    print("| case | size | torch ms | mojo ms | mojo/torch |")
    print("|---|---|---:|---:|---:|")

    # -- sparse full self-attention, three sizes
    var attn_tokens: List[Int] = [256, 512, 1024]
    var attn_heads: List[Int] = [4, 4, 8]
    var attn_hdim: List[Int] = [32, 64, 64]
    var attn_names: List[String] = ["attn-full S", "attn-full M", "attn-full L"]
    var padded_note = String("")
    for i in range(3):
        var c = pyref.case_attn(0, attn_tokens[i], 2, attn_heads[i], attn_hdim[i], GRID)
        var torch_ms = py_min_ms(pyref.time_attn_correct(c, ITERS))
        var padded_ms = py_min_ms(pyref.time_attn_padded(c, ITERS))
        var x = sparse_of(c, "qkv")
        var size = (
            "2x" + String(attn_tokens[i]) + " tok, H" + String(attn_heads[i])
            + " D" + String(attn_hdim[i])
        )
        row(attn_names[i], size, torch_ms, bench_attn(x))
        padded_note += " " + attn_names[i] + ": " + fmt(padded_ms) + " ms;"

    # -- windowed self-attention (the model-used window path)
    var cw = pyref.case_attn(1, 2048, 2, 4, 32, GRID)
    var torch_w = py_min_ms(pyref.time_attn_windowed(cw, 4, ITERS))
    var xw = sparse_of(cw, "qkv")
    row("attn-windowed", "2x2048 tok, H4 D32, win 4", torch_w, bench_windowed(xw, 4))

    # -- submanifold sparse conv 3^3
    var cc = pyref.case_conv(2, 2048, 2, 32, 32, 3, GRID)
    var torch_c = py_min_ms(pyref.time_conv(cc, ITERS))
    var xc = sparse_of(cc, "feats")
    var conv = SparseConv3d(tensor_from_torch(cc["w"]), tensor_from_torch(cc["b"]), dilation=1)
    row("conv3d 3^3", "2x2048 tok, 32->32", torch_c, bench_conv(conv, xc))

    # -- modulated sparse transformer block (share_mod, mlp_ratio 4)
    comptime BCH = 128
    comptime BHEADS = 4
    var cb = pyref.case_block(3, 512, 2, BCH, BHEADS, GRID)
    var torch_b = py_min_ms(pyref.time_block(cb, ITERS))
    var sd = cb["sd"]
    var block = ModulatedSparseTransformerBlock(
        BCH,
        ln_from(sd, "norm1", BCH), ln_from(sd, "norm2", BCH),
        sparse_mha_from(sd, "attn", BCH, BHEADS),
        sparse_ffn_from(sd, "mlp"),
        modulation_from(sd, True),
    )
    var xb = sparse_of(cb, "feats")
    row(
        "mod-block", "2x512 tok, C128 H4, ffn x4",
        torch_b, bench_block(block, xb, tensor_from_torch(cb["mod"])),
    )

    # -- FlowEuler CFG-interval sampling loop with a small SLat flow model
    var cs = pyref.case_sampler(4, 128, 2, 8, 64, 2, 2, 5, 16, STEPS, GRID)
    var torch_s = py_min_ms(pyref.time_sampler(cs, ITERS))
    var model = slat_flow_from(cs["sd"], 8, 64, 16, 8, 2, 2, True, True, True, True)
    var cond = tensor_from_torch(cs["cond"])
    var neg_cond = Tensor[F32](cond.shape)
    var vel = SlatFlowVelocity(model^, intmatrix_from_torch(cs["coords"]), cond.copy(), neg_cond^)
    row(
        "slat-sampler", "2x128 tok, C64 x2 blk, 8 steps",
        torch_s, bench_sampler(vel, tensor_from_torch(cs["noise"])),
    )

    print()
    print("torch padded-naive attention (incorrect semantics, vectorized):" + padded_note)
