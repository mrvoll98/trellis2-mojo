# WP11 step 3: GPU dense SDPA (GEMM composition, incl. pack + transfers)
# vs the CPU varlen_sdpa (flash path) on the ss_flow DiT's real shapes.
# CPU references from pass 8: self 4096 H16 D64 ~197 ms, cross 4096x1029
# ~52 ms. WP11 step 7 adds whole-MHA timings (qkv+rms/rope+sdpa+out) for
# chained vs unchained-GPU vs CPU on the REAL model geometries (ss_flow:
# C 1536 H 12 D 128; slat: C 1024 H 16 D 64). Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.attention import gpu_dense_sdpa, gpu_varlen_sdpa_single
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.loaders import dense_mha_from, sparse_mha_from
from trellis2_mojo.modules.rope import RotaryPositionEmbedder
from trellis2_mojo.sparse.attention.full_attn import dense_sdpa_q_k_v, varlen_sdpa
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def fill(mut t: Tensor[F32], seed: Int, scale: Float32):
    var state = seed * 6364136223846793005 + 1442695040888963407
    for i in range(len(t.data)):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & ((1 << 20) - 1)) / Float32(1 << 19)
        t.data[i] = (u - 1.0) * scale


def bench(gpu: GpuContext, l: Int, lkv: Int, h: Int, d: Int) raises:
    var q = Tensor[F32]([1, l, h, d])
    var k = Tensor[F32]([1, lkv, h, d])
    var v = Tensor[F32]([1, lkv, h, d])
    fill(q, l + lkv, 1.0)
    fill(k, l + lkv + 1, 1.0)
    fill(v, l + lkv + 2, 1.0)

    var best_gpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gpu_dense_sdpa(gpu, q, k, v)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_gpu:
            best_gpu = ms

    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = dense_sdpa_q_k_v(q, k, v)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms

    print(
        "L", l, " Lkv", lkv, " H", h, " D", d, ": gpu", best_gpu,
        "ms  cpu", best_cpu, "ms  speedup", best_cpu / best_gpu,
    )


def bench_single(gpu: GpuContext, t: Int, tkv: Int, h: Int, d: Int) raises:
    """Single-segment varlen (WP11 step 4, q-padding) vs CPU varlen_sdpa."""
    var q = Tensor[F32]([t, h, d])
    var k = Tensor[F32]([tkv, h, d])
    var v = Tensor[F32]([tkv, h, d])
    fill(q, t + tkv, 1.0)
    fill(k, t + tkv + 1, 1.0)
    fill(v, t + tkv + 2, 1.0)
    var qo: List[Int] = [0, t]
    var ko: List[Int] = [0, tkv]

    var best_gpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gpu_varlen_sdpa_single(gpu, q, k, v)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_gpu:
            best_gpu = ms

    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = varlen_sdpa(q, k, v, qo, ko)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms

    print(
        "single-seg T", t, " Tkv", tkv, " H", h, " D", d, ": gpu", best_gpu,
        "ms  cpu", best_cpu, "ms  speedup", best_cpu / best_gpu,
    )


def mha_sd(c: Int, h: Int, seed: Int) raises -> StateDict:
    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([3 * c, c])
    var bq = Tensor[F32]([3 * c])
    var wo = Tensor[F32]([c, c])
    var bo = Tensor[F32]([c])
    var gq = Tensor[F32]([h, c // h])
    var gk = Tensor[F32]([h, c // h])
    fill(wq, seed, 0.05)
    fill(bq, seed + 1, 0.1)
    fill(wo, seed + 2, 0.05)
    fill(bo, seed + 3, 0.1)
    fill(gq, seed + 4, 0.25)
    fill(gk, seed + 5, 0.25)
    for i in range(len(gq.data)):
        gq.data[i] += 1.0
        gk.data[i] += 1.0
    d["attn.to_qkv.weight"] = wq^
    d["attn.to_qkv.bias"] = bq^
    d["attn.to_out.weight"] = wo^
    d["attn.to_out.bias"] = bo^
    d["attn.q_rms_norm.gamma"] = gq^
    d["attn.k_rms_norm.gamma"] = gk^
    return StateDict(d^)


def bench_chain_dense(gpu: GpuContext, l: Int, c: Int, h: Int) raises:
    """Whole-MHA (qkv + rms + rope + sdpa + out) on the dense flow path:
    chained (step 7) vs unchained GPU (steps 2+3) vs CPU."""
    var sd = mha_sd(c, h, l + c)
    var cpu_mha = dense_mha_from(sd, "attn", c, h, qk_rms_norm=True)
    sd.gpu = gpu.copy()
    var gpu_mha = dense_mha_from(sd, "attn", c, h, qk_rms_norm=True)
    var nochain = gpu_mha.copy()
    nochain.chain = None
    if not gpu_mha.chain:
        raise Error("bench: chain was not built")

    var emb = RotaryPositionEmbedder(c // h)
    var pos = Tensor[F32]([l, 3])
    for r in range(l):
        pos.data[r * 3] = Float32(r % 16)
        pos.data[r * 3 + 1] = Float32((r // 16) % 16)
        pos.data[r * 3 + 2] = Float32(r // 256)
    var phases = emb.forward(pos)
    var x = Tensor[F32]([1, l, c])
    fill(x, l + c + 9, 1.0)

    var best_chain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gpu_mha.forward(x, phases)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_chain:
            best_chain = ms
    var best_nochain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = nochain.forward(x, phases)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_nochain:
            best_nochain = ms
    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = cpu_mha.forward(x, phases)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms
    print(
        "dense MHA L", l, " C", c, " H", h, ": chain", best_chain,
        "ms  gpu-unchained", best_nochain, "ms  cpu", best_cpu,
        "ms  chain-speedup(vs unchained)", best_nochain / best_chain,
        " (vs cpu)", best_cpu / best_chain,
    )


def bench_chain_sparse(gpu: GpuContext, t: Int, c: Int, h: Int) raises:
    """Whole sparse MHA (single segment, rope from coords)."""
    var sd = mha_sd(c, h, t + c)
    var cpu_mha = sparse_mha_from(sd, "attn", c, h, use_rope=True, qk_rms_norm=True)
    sd.gpu = gpu.copy()
    var gpu_mha = sparse_mha_from(sd, "attn", c, h, use_rope=True, qk_rms_norm=True)
    var nochain = gpu_mha.copy()
    nochain.chain = None
    if not gpu_mha.chain:
        raise Error("bench: sparse chain was not built")

    var feats = Tensor[F32]([t, c])
    fill(feats, t + c + 9, 1.0)
    var coords = IntMatrix(t, 4)
    for r in range(t):
        coords.set(r, 1, r % 32)
        coords.set(r, 2, (r // 32) % 32)
        coords.set(r, 3, r // 1024)
    var x = SparseTensor[F32](feats.copy(), coords.copy(), 1)

    var best_chain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gpu_mha.forward(x)
        var t1 = perf_counter_ns()
        _ = y.vl.feats.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_chain:
            best_chain = ms
    var best_nochain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = nochain.forward(x)
        var t1 = perf_counter_ns()
        _ = y.vl.feats.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_nochain:
            best_nochain = ms
    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = cpu_mha.forward(x)
        var t1 = perf_counter_ns()
        _ = y.vl.feats.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms
    print(
        "sparse MHA T", t, " C", c, " H", h, ": chain", best_chain,
        "ms  gpu-unchained", best_nochain, "ms  cpu", best_cpu,
        "ms  chain-speedup(vs unchained)", best_nochain / best_chain,
        " (vs cpu)", best_cpu / best_chain,
    )


def cross_mha_sd(c: Int, h: Int, cctx: Int, seed: Int) raises -> StateDict:
    var d = Dict[String, Tensor[F32]]()
    var wq = Tensor[F32]([c, c])
    var bq = Tensor[F32]([c])
    var wkv = Tensor[F32]([2 * c, cctx])
    var bkv = Tensor[F32]([2 * c])
    var wo = Tensor[F32]([c, c])
    var bo = Tensor[F32]([c])
    var gq = Tensor[F32]([h, c // h])
    var gk = Tensor[F32]([h, c // h])
    fill(wq, seed, 0.05)
    fill(bq, seed + 1, 0.1)
    fill(wkv, seed + 2, 0.05)
    fill(bkv, seed + 3, 0.1)
    fill(wo, seed + 4, 0.05)
    fill(bo, seed + 5, 0.1)
    fill(gq, seed + 6, 0.25)
    fill(gk, seed + 7, 0.25)
    for i in range(len(gq.data)):
        gq.data[i] += 1.0
        gk.data[i] += 1.0
    d["attn.to_q.weight"] = wq^
    d["attn.to_q.bias"] = bq^
    d["attn.to_kv.weight"] = wkv^
    d["attn.to_kv.bias"] = bkv^
    d["attn.to_out.weight"] = wo^
    d["attn.to_out.bias"] = bo^
    d["attn.q_rms_norm.gamma"] = gq^
    d["attn.k_rms_norm.gamma"] = gk^
    return StateDict(d^)


def bench_chain_cross(gpu: GpuContext, l: Int, lkv: Int, c: Int, cctx: Int, h: Int) raises:
    """Whole cross-MHA (q + q-rms + sdpa + out; kv + k-rms on CPU both
    ways): chained (step 8) vs unchained GPU vs CPU."""
    var sd = cross_mha_sd(c, h, cctx, l + c)
    var cpu_mha = dense_mha_from(sd, "attn", c, h, is_cross=True, qk_rms_norm=True)
    sd.gpu = gpu.copy()
    var gpu_mha = dense_mha_from(sd, "attn", c, h, is_cross=True, qk_rms_norm=True)
    var nochain = gpu_mha.copy()
    nochain.cross_chain = None
    if not gpu_mha.cross_chain:
        raise Error("bench: cross chain was not built")

    var x = Tensor[F32]([1, l, c])
    var ctx = Tensor[F32]([1, lkv, cctx])
    fill(x, l + c + 9, 1.0)
    fill(ctx, l + c + 10, 1.0)

    var best_chain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gpu_mha.forward_cross(x, ctx)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_chain:
            best_chain = ms
    var best_nochain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = nochain.forward_cross(x, ctx)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_nochain:
            best_nochain = ms
    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = cpu_mha.forward_cross(x, ctx)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms
    print(
        "cross MHA L", l, " Lkv", lkv, " C", c, " H", h, ": chain", best_chain,
        "ms  gpu-unchained", best_nochain, "ms  cpu", best_cpu,
        "ms  chain-speedup(vs unchained)", best_nochain / best_chain,
        " (vs cpu)", best_cpu / best_chain,
    )


def main() raises:
    var gpu = GpuContext()
    print("GPU dense SDPA (pack+kernels+readback) vs CPU flash (min of 3):")
    bench(gpu, 4096, 4096, 16, 64)  # ss_flow self-attention
    bench(gpu, 4096, 1029, 16, 64)  # ss_flow cross-attention
    bench(gpu, 2048, 2048, 16, 64)
    bench_single(gpu, 2369, 2369, 16, 64)  # slat self (smoke token count)
    bench_single(gpu, 2369, 1029, 16, 64)  # slat cross
    print("gate-floor probe (golden run landed at 1857 < 2048):")
    bench_single(gpu, 1857, 1857, 12, 128)  # golden slat self (real geom)
    bench_single(gpu, 1857, 1029, 12, 128)  # golden slat cross
    bench_single(gpu, 1280, 1280, 12, 128)
    bench_single(gpu, 1024, 1024, 12, 128)
    print("WP11 step 7: whole-MHA chained vs unchained vs CPU (min of 3):")
    bench_chain_dense(gpu, 4096, 1536, 12)   # ss_flow real geometry (D 128)
    bench_chain_sparse(gpu, 2369, 1024, 16)  # slat real geometry (D 64)
    print("WP11 step 8: whole cross-MHA chained vs unchained vs CPU (min of 3):")
    bench_chain_cross(gpu, 4096, 1029, 1536, 1024, 12)  # ss_flow cross
    bench_chain_cross(gpu, 2369, 1029, 1024, 1024, 16)  # slat cross geometry
