# Throwaway micro-benchmark: per-op timing of ONE dense DiT cross-block
# forward at the REAL ss-flow shape ([1, 4096, 1024], H16 D64, ffn 4096,
# qk_rms + rope, cross-context [1, 1029, 1024]) — to attribute where the
# ~2 min/forward in the e2e ss stage actually goes. Not wired into pixi.

from std.time import perf_counter_ns

from trellis2_mojo.sparse.tensor import Tensor, OP_ADD, OP_MUL
from trellis2_mojo.modules.nn import LayerNorm32, activation, ACT_GELU_TANH, linear, modulate
from trellis2_mojo.modules.rope import apply_rotary_embedding
from trellis2_mojo.sparse.attention.modules import MultiHeadRMSNorm
from trellis2_mojo.sparse.attention.full_attn import dense_sdpa_q_k_v

comptime F32 = DType.float32
comptime ITERS = 3
comptime L = 4096
comptime LC = 1029
comptime C = 1024
comptime H = 16
comptime D = 64


def fill(mut t: Tensor[F32], seed: Int) raises:
    for i in range(t.numel()):
        t.data[i] = Float32(((i * 31 + seed * 17) % 41) - 20) * 0.008


def tick(mut acc: List[Float64], mut t0: UInt) raises:
    var t1 = perf_counter_ns()
    acc.append(Float64(t1 - t0) / 1e6)
    t0 = perf_counter_ns()


def report(name: String, acc: List[Float64]) raises:
    # min over iters (entries appended once per iter)
    var best: Float64 = 1e30
    for v in acc:
        if v < best:
            best = v
    print("  " + name + ": " + String(best) + " ms")


def main() raises:
    var x = Tensor[F32]([1, L, C])
    fill(x, 1)
    var ctx = Tensor[F32]([1, LC, C])
    fill(ctx, 2)
    var shift = Tensor[F32]([1, C])
    var scale = Tensor[F32]([1, C])
    fill(shift, 3)
    fill(scale, 4)
    var phases = Tensor[F32]([L, D // 2, 2])
    fill(phases, 5)
    var w_qkv = Tensor[F32]([3 * C, C])
    var b_qkv = Tensor[F32]([3 * C])
    var w_out = Tensor[F32]([C, C])
    var b_out = Tensor[F32]([C])
    var w_q = Tensor[F32]([C, C])
    var b_q = Tensor[F32]([C])
    var w_kv = Tensor[F32]([2 * C, C])
    var b_kv = Tensor[F32]([2 * C])
    var w_f1 = Tensor[F32]([4 * C, C])
    var b_f1 = Tensor[F32]([4 * C])
    var w_f2 = Tensor[F32]([C, 4 * C])
    var b_f2 = Tensor[F32]([C])
    fill(w_qkv, 6)
    fill(w_out, 7)
    fill(w_q, 8)
    fill(w_kv, 9)
    fill(w_f1, 10)
    fill(w_f2, 11)
    var ln = LayerNorm32(C)
    var rms = MultiHeadRMSNorm(D, H)

    var t_ln = List[Float64]()
    var t_mod = List[Float64]()
    var t_qkv = List[Float64]()
    var t_reshape = List[Float64]()
    var t_unbind = List[Float64]()
    var t_rms = List[Float64]()
    var t_rope = List[Float64]()
    var t_sdpa = List[Float64]()
    var t_out = List[Float64]()
    var t_add = List[Float64]()
    var t_cq = List[Float64]()
    var t_ckv = List[Float64]()
    var t_csdpa = List[Float64]()
    var t_ffn1 = List[Float64]()
    var t_act = List[Float64]()
    var t_ffn2 = List[Float64]()

    for it in range(ITERS + 1):
        var t0 = perf_counter_ns()
        var n1 = ln.forward(x)
        tick(t_ln, t0)
        var h = modulate(n1, shift, scale)
        tick(t_mod, t0)
        var qkv = linear(h, w_qkv, b_qkv)
        tick(t_qkv, t0)
        var qkv_shape: List[Int] = [1, L, 3, H, D]
        var qkv5 = Tensor[F32].from_values(qkv_shape, qkv.data)
        tick(t_reshape, t0)
        var parts = qkv5.unbind(2)
        tick(t_unbind, t0)
        var q = rms.forward(parts[0])
        var k = rms.forward(parts[1])
        tick(t_rms, t0)
        q = apply_rotary_embedding(q, phases)
        k = apply_rotary_embedding(k, phases)
        tick(t_rope, t0)
        var attn = dense_sdpa_q_k_v(q, k, parts[2])
        tick(t_sdpa, t0)
        var attn_shape: List[Int] = [1, L, C]
        var o = linear(Tensor[F32].from_values(attn_shape, attn.data), w_out, b_out)
        tick(t_out, t0)
        var y = x._binop_flat(o, OP_ADD)
        tick(t_add, t0)
        # cross-attention
        var cq_shape: List[Int] = [1, L, H, D]
        var cq = Tensor[F32].from_values(cq_shape, linear(ln.forward(y), w_q, b_q).data)
        tick(t_cq, t0)
        var ckv_raw = linear(ctx, w_kv, b_kv)
        var ckv_shape: List[Int] = [1, LC, 2, H, D]
        var ckv = Tensor[F32].from_values(ckv_shape, ckv_raw.data).unbind(2)
        tick(t_ckv, t0)
        var cattn = dense_sdpa_q_k_v(cq, ckv[0], ckv[1])
        tick(t_csdpa, t0)
        var f1 = linear(y, w_f1, b_f1)
        tick(t_ffn1, t0)
        var g = activation(f1, ACT_GELU_TANH)
        tick(t_act, t0)
        var f2 = linear(g, w_f2, b_f2)
        tick(t_ffn2, t0)
        _ = f2
        _ = cattn
        if it == 0:  # drop warmup: keep only timed iters
            t_ln.clear(); t_mod.clear(); t_qkv.clear(); t_reshape.clear()
            t_unbind.clear(); t_rms.clear(); t_rope.clear(); t_sdpa.clear()
            t_out.clear(); t_add.clear(); t_cq.clear(); t_ckv.clear()
            t_csdpa.clear(); t_ffn1.clear(); t_act.clear(); t_ffn2.clear()

    print("dense DiT cross-block @ [1, 4096, 1024], H16 D64, ffn 4096 (min of 3):")
    report("layernorm", t_ln)
    report("modulate", t_mod)
    report("qkv linear (1024->3072)", t_qkv)
    report("reshape", t_reshape)
    report("unbind", t_unbind)
    report("qk rms x2", t_rms)
    report("rope q+k", t_rope)
    report("self sdpa (4096 tok)", t_sdpa)
    report("out linear (1024->1024)", t_out)
    report("residual add", t_add)
    report("cross q (ln+linear)", t_cq)
    report("cross kv linear (1029 tok)", t_ckv)
    report("cross sdpa (4096 x 1029)", t_csdpa)
    report("ffn1 (1024->4096)", t_ffn1)
    report("gelu-tanh", t_act)
    report("ffn2 (4096->1024)", t_ffn2)
