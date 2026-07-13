# Throwaway micro-benchmark: per-step timing of the mod-block forward
# (2x512 tok, C128 H4, ffn x4 — same shapes as the WP10 bench case) to
# attribute where the block time goes. Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix, OP_ADD, OP_MUL
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.modules.nn import LayerNorm32, SparseLinear, activation, ACT_GELU_TANH, linear
from trellis2_mojo.sparse.attention.full_attn import varlen_sdpa

comptime F32 = DType.float32
comptime ITERS = 5
comptime T = 1024
comptime C = 128
comptime H = 4
comptime D = 32


def fill(mut t: Tensor[F32], seed: Int) raises:
    for i in range(t.numel()):
        t.data[i] = Float32(((i * 31 + seed * 17) % 41) - 20) * 0.008


def best_ms(mut times: List[Float64]) raises -> Float64:
    var best: Float64 = 1e30
    for i in range(len(times)):
        if times[i] < best:
            best = times[i]
    return best


def main() raises:
    var feats = Tensor[F32]([T, C])
    fill(feats, 1)
    var coords = IntMatrix(T, 4)
    for i in range(T):
        coords.set(i, 0, 0 if i < T // 2 else 1)
        coords.set(i, 1, (i // 64) % 16)
        coords.set(i, 2, (i // 4) % 16)
        coords.set(i, 3, i % 16)
    var x = SparseTensor[F32](feats.copy(), coords^, 2)

    var w_qkv = Tensor[F32]([3 * C, C])
    var b_qkv = Tensor[F32]([3 * C])
    var w_out = Tensor[F32]([C, C])
    var b_out = Tensor[F32]([C])
    var w_f1 = Tensor[F32]([4 * C, C])
    var b_f1 = Tensor[F32]([4 * C])
    var w_f2 = Tensor[F32]([C, 4 * C])
    var b_f2 = Tensor[F32]([C])
    fill(w_qkv, 2)
    fill(b_qkv, 3)
    fill(w_out, 4)
    fill(b_out, 5)
    fill(w_f1, 6)
    fill(b_f1, 7)
    fill(w_f2, 8)
    fill(b_f2, 9)
    var ln = LayerNorm32(C)
    var scale_row = Tensor[F32]([2, C])
    fill(scale_row, 10)

    var t_ln = List[Float64]()
    var t_mod = List[Float64]()
    var t_qkv = List[Float64]()
    var t_reshape = List[Float64]()
    var t_unbind = List[Float64]()
    var t_sdpa = List[Float64]()
    var t_out = List[Float64]()
    var t_gate = List[Float64]()
    var t_add = List[Float64]()
    var t_ffn1 = List[Float64]()
    var t_act = List[Float64]()
    var t_ffn2 = List[Float64]()

    for _ in range(ITERS + 1):
        var t0 = perf_counter_ns()
        var n1 = ln.forward(x.vl.feats)
        var t1 = perf_counter_ns()
        var h = x.replace(n1^)
        h = h.elemwise_batch(scale_row._binop_scalar(1.0, OP_ADD), OP_MUL)
        h = h.elemwise_batch(scale_row, OP_ADD)
        var t2 = perf_counter_ns()
        var qkv = linear(h.vl.feats, w_qkv, b_qkv)
        var t3 = perf_counter_ns()
        var tail: List[Int] = [3, H, D]
        qkv = qkv.reshape_rows(tail)
        var t4 = perf_counter_ns()
        var parts = qkv.unbind(1)
        var t5 = perf_counter_ns()
        var att = varlen_sdpa(parts[0], parts[1], parts[2], x.vl.offsets, x.vl.offsets)
        var t6 = perf_counter_ns()
        var flat: List[Int] = [C]
        var o = linear(att.reshape_rows(flat), w_out, b_out)
        var t7 = perf_counter_ns()
        var g = x.replace(o^).elemwise_batch(scale_row, OP_MUL)
        var t8 = perf_counter_ns()
        var y = x + g
        var t9 = perf_counter_ns()
        var f1 = linear(y.vl.feats, w_f1, b_f1)
        var t10 = perf_counter_ns()
        var fa = activation(f1, ACT_GELU_TANH)
        var t11 = perf_counter_ns()
        var f2 = linear(fa, w_f2, b_f2)
        var t12 = perf_counter_ns()
        _ = f2
        _ = y
        t_ln.append(Float64(t1 - t0) / 1e6)
        t_mod.append(Float64(t2 - t1) / 1e6)
        t_qkv.append(Float64(t3 - t2) / 1e6)
        t_reshape.append(Float64(t4 - t3) / 1e6)
        t_unbind.append(Float64(t5 - t4) / 1e6)
        t_sdpa.append(Float64(t6 - t5) / 1e6)
        t_out.append(Float64(t7 - t6) / 1e6)
        t_gate.append(Float64(t8 - t7) / 1e6)
        t_add.append(Float64(t9 - t8) / 1e6)
        t_ffn1.append(Float64(t10 - t9) / 1e6)
        t_act.append(Float64(t11 - t10) / 1e6)
        t_ffn2.append(Float64(t12 - t11) / 1e6)

    print("layernorm      ", best_ms(t_ln), "ms")
    print("mod shift/scale", best_ms(t_mod), "ms")
    print("to_qkv linear  ", best_ms(t_qkv), "ms")
    print("reshape_rows   ", best_ms(t_reshape), "ms")
    print("unbind         ", best_ms(t_unbind), "ms")
    print("varlen_sdpa    ", best_ms(t_sdpa), "ms")
    print("to_out linear  ", best_ms(t_out), "ms")
    print("gate mul       ", best_ms(t_gate), "ms")
    print("residual add   ", best_ms(t_add), "ms")
    print("ffn lin1       ", best_ms(t_ffn1), "ms")
    print("gelu-tanh      ", best_ms(t_act), "ms")
    print("ffn lin2       ", best_ms(t_ffn2), "ms")
