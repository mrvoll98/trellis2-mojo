# Throwaway: per-segment timing of ONE real ModulatedTransformerCrossBlock
# forward at the REAL ss_flow geometry ([1, 4096, 1536], H12 D128,
# ffn 8192, ctx [1, 1029, 1024]) with TRELLIS2_GPU-style offload ACTIVE —
# attributes the remaining ss-stage time after WP11 step 7 (chained
# self-attention). WP11 step 10 adds the whole-block struct timing:
# per-op segments (chains with individual transfers + CPU glue) vs the
# device-resident block forward. Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.loaders import dense_mha_from, dense_ffn_from
from trellis2_mojo.modules.nn import LayerNorm32, modulate
from trellis2_mojo.modules.rope import RotaryPositionEmbedder
from trellis2_mojo.modules.transformer.modulated import ModulatedTransformerCrossBlock, _gate
from trellis2_mojo.sparse.transformer.modulated import Modulation
from trellis2_mojo.sparse.tensor import Tensor, OP_ADD

comptime F32 = DType.float32
comptime ITERS = 3


def fill(mut t: Tensor[F32], seed: Int, scale: Float32):
    var state = seed * 6364136223846793005 + 1442695040888963407
    for i in range(len(t.data)):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & ((1 << 20) - 1)) / Float32(1 << 19)
        t.data[i] = (u - 1.0) * scale


def report(name: String, acc: List[Float64]) raises:
    var best: Float64 = 1e30
    for v in acc:
        if v < best:
            best = v
    print("  " + name + ": " + String(best) + " ms")


def main() raises:
    var gpu = GpuContext()
    comptime L = 4096
    comptime LC = 1029
    comptime C = 1536
    comptime CTX = 1024
    comptime H = 12
    comptime D = 128
    comptime FFN = 8192

    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([3 * C, C])
    var bq = Tensor[F32]([3 * C])
    var wo = Tensor[F32]([C, C])
    var bo = Tensor[F32]([C])
    var gq = Tensor[F32]([H, D])
    var gk = Tensor[F32]([H, D])
    fill(wq, 1, 0.02)
    fill(bq, 2, 0.05)
    fill(wo, 3, 0.02)
    fill(bo, 4, 0.05)
    fill(gq, 5, 0.25)
    fill(gk, 6, 0.25)
    for i in range(len(gq.data)):
        gq.data[i] += 1.0
        gk.data[i] += 1.0
    d["self_attn.to_qkv.weight"] = wq^
    d["self_attn.to_qkv.bias"] = bq^
    d["self_attn.to_out.weight"] = wo^
    d["self_attn.to_out.bias"] = bo^
    d["self_attn.q_rms_norm.gamma"] = gq^
    d["self_attn.k_rms_norm.gamma"] = gk^
    var cwq = Tensor[F32]([C, C])
    var cbq = Tensor[F32]([C])
    var cwkv = Tensor[F32]([2 * C, CTX])
    var cbkv = Tensor[F32]([2 * C])
    var cwo = Tensor[F32]([C, C])
    var cbo = Tensor[F32]([C])
    var cgq = Tensor[F32]([H, D])
    var cgk = Tensor[F32]([H, D])
    fill(cwq, 7, 0.02)
    fill(cbq, 8, 0.05)
    fill(cwkv, 9, 0.02)
    fill(cbkv, 10, 0.05)
    fill(cwo, 11, 0.02)
    fill(cbo, 12, 0.05)
    fill(cgq, 13, 0.25)
    fill(cgk, 14, 0.25)
    for i in range(len(cgq.data)):
        cgq.data[i] += 1.0
        cgk.data[i] += 1.0
    d["cross_attn.to_q.weight"] = cwq^
    d["cross_attn.to_q.bias"] = cbq^
    d["cross_attn.to_kv.weight"] = cwkv^
    d["cross_attn.to_kv.bias"] = cbkv^
    d["cross_attn.to_out.weight"] = cwo^
    d["cross_attn.to_out.bias"] = cbo^
    d["cross_attn.q_rms_norm.gamma"] = cgq^
    d["cross_attn.k_rms_norm.gamma"] = cgk^
    var wf1 = Tensor[F32]([FFN, C])
    var bf1 = Tensor[F32]([FFN])
    var wf2 = Tensor[F32]([C, FFN])
    var bf2 = Tensor[F32]([C])
    fill(wf1, 15, 0.02)
    fill(bf1, 16, 0.05)
    fill(wf2, 17, 0.02)
    fill(bf2, 18, 0.05)
    d["mlp.mlp.0.weight"] = wf1^
    d["mlp.mlp.0.bias"] = bf1^
    d["mlp.mlp.2.weight"] = wf2^
    d["mlp.mlp.2.bias"] = bf2^

    var sd = StateDict(d^)
    sd.gpu = gpu.copy()
    var self_attn = dense_mha_from(sd, "self_attn", C, H, qk_rms_norm=True)
    var cross_attn = dense_mha_from(sd, "cross_attn", C, H, is_cross=True, qk_rms_norm=True)
    var mlp = dense_ffn_from(sd, "mlp")
    if not self_attn.chain:
        raise Error("bench: no self chain")

    var ln1 = LayerNorm32(C, 1e-6, False)
    var ln2 = LayerNorm32(C, 1e-6, True)
    var ln3 = LayerNorm32(C, 1e-6, False)
    var emb = RotaryPositionEmbedder(D)
    var pos = Tensor[F32]([L, 3])
    for r in range(L):
        pos.data[r * 3] = Float32(r % 16)
        pos.data[r * 3 + 1] = Float32((r // 16) % 16)
        pos.data[r * 3 + 2] = Float32(r // 256)
    var phases = emb.forward(pos)

    var x = Tensor[F32]([1, L, C])
    fill(x, 20, 1.0)
    var ctx = Tensor[F32]([1, LC, CTX])
    fill(ctx, 21, 1.0)
    var shift = Tensor[F32]([1, C])
    var scale = Tensor[F32]([1, C])
    var gate = Tensor[F32]([1, C])
    fill(shift, 22, 0.1)
    fill(scale, 23, 0.1)
    fill(gate, 24, 0.5)

    var t_glue1 = List[Float64]()
    var t_self = List[Float64]()
    var t_glue2 = List[Float64]()
    var t_cross = List[Float64]()
    var t_glue3 = List[Float64]()
    var t_mlp = List[Float64]()
    var t_glue4 = List[Float64]()
    var t_total = List[Float64]()

    for it in range(ITERS + 1):
        var tb = perf_counter_ns()
        var t0 = perf_counter_ns()
        var h = modulate(ln1.forward(x), shift, scale)
        var t1 = perf_counter_ns()
        var a = self_attn.forward(h, phases)
        var t2 = perf_counter_ns()
        var y = x._binop_flat(_gate(a, gate), OP_ADD)
        var n2 = ln2.forward(y)
        var t3 = perf_counter_ns()
        var ca = cross_attn.forward_cross(n2, ctx)
        var t4 = perf_counter_ns()
        y = y._binop_flat(ca, OP_ADD)
        var h3 = modulate(ln3.forward(y), shift, scale)
        var t5 = perf_counter_ns()
        var f = mlp.forward(h3)
        var t6 = perf_counter_ns()
        y = y._binop_flat(_gate(f, gate), OP_ADD)
        var t7 = perf_counter_ns()
        _ = y.data[0]
        if it > 0:
            t_glue1.append(Float64(t1 - t0) / 1e6)
            t_self.append(Float64(t2 - t1) / 1e6)
            t_glue2.append(Float64(t3 - t2) / 1e6)
            t_cross.append(Float64(t4 - t3) / 1e6)
            t_glue3.append(Float64(t5 - t4) / 1e6)
            t_mlp.append(Float64(t6 - t5) / 1e6)
            t_glue4.append(Float64(t7 - t6) / 1e6)
            t_total.append(Float64(t7 - tb) / 1e6)

    print("ss_flow cross-block @ [1, 4096, 1536], H12 D128, ffn 8192, GPU on (min of 3):")
    report("norm1+modulate", t_glue1)
    report("self-attn (chained)", t_self)
    report("gate+add+norm2", t_glue2)
    report("cross-attn (q/out GPU, kv CPU, sdpa GPU)", t_cross)
    report("add+norm3+modulate", t_glue3)
    report("mlp (chained)", t_mlp)
    report("gate+add", t_glue4)
    report("TOTAL block", t_total)

    # WP11 step 10: the whole-block struct — device-resident vs per-op.
    # Rebuild the parts as a real block (share_mod modulation); the
    # GPU-built instance takes gpu_cross_block_forward, the per-op timing
    # above is the step-7/8 comparison baseline.
    var n2 = LayerNorm32(C, 1e-6, True)
    var n2w = Tensor[F32]([C])
    var n2b = Tensor[F32]([C])
    fill(n2w, 30, 0.25)
    fill(n2b, 31, 0.1)
    for i in range(C):
        n2w.data[i] += 1.0
    n2.weight = n2w^
    n2.bias = n2b^
    var mvec = Tensor[F32]([6 * C])
    fill(mvec, 32, 0.2)
    var blk = ModulatedTransformerCrossBlock(
        C,
        LayerNorm32(C, 1e-6, False), n2^, LayerNorm32(C, 1e-6, False),
        self_attn.copy(), cross_attn.copy(), mlp.copy(),
        Modulation(mvec.copy()),
    )
    if not blk._gpu_block_ok(1, L, LC):
        raise Error("bench: block gate rejected the ss geometry")
    var bmod = Tensor[F32]([1, 6 * C])
    fill(bmod, 33, 0.2)
    var best_blk: Float64 = 1e30
    for it in range(ITERS + 1):
        var t0 = perf_counter_ns()
        var y = blk.forward(x, bmod, ctx, phases)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_blk:
            best_blk = ms
    print("  WHOLE BLOCK device-resident (step 10): " + String(best_blk) + " ms")
