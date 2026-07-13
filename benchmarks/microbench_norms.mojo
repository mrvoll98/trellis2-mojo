# Throwaway micro-benchmark: MultiHeadRMSNorm / rope / GroupNorm32 /
# ChannelLayerNorm32 at model-realistic shapes (the qk_rms+rope path real
# checkpoints use, and the SS-VAE decoder path). Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.modules.nn import GroupNorm32, ChannelLayerNorm32
from trellis2_mojo.sparse.attention.modules import MultiHeadRMSNorm
from trellis2_mojo.sparse.attention.rope import SparseRotaryPositionEmbedder

comptime F32 = DType.float32
comptime ITERS = 5


def fill(mut t: Tensor[F32], seed: Int) raises:
    for i in range(t.numel()):
        t.data[i] = Float32(((i * 31 + seed * 17) % 41) - 20) * 0.008


def main() raises:
    # qk_rms + rope path: 4096 tokens, H16 D64 (big-model-ish)
    comptime T = 4096
    comptime H = 16
    comptime D = 64
    var q = Tensor[F32]([T, H, D])
    fill(q, 1)
    var rms = MultiHeadRMSNorm(D, H)
    var coords = IntMatrix(T, 4)
    for i in range(T):
        coords.set(i, 0, 0 if i < T // 2 else 1)
        coords.set(i, 1, (i // 256) % 16)
        coords.set(i, 2, (i // 16) % 16)
        coords.set(i, 3, i % 16)
    var xs = SparseTensor[F32](q.copy(), coords^, 2)
    var rope = SparseRotaryPositionEmbedder(D)

    var best: Float64 = 1e30
    _ = rms.forward(q)
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = rms.forward(q)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
    print("rms [4096,16,64]              ", best, "ms")

    best = 1e30
    _ = rope.embed(xs, xs)  # warmup fills the phase cache
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = rope.embed(xs, xs)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
    print("rope embed q+k (cached phases)", best, "ms")

    # SS-VAE decoder shapes: [1, C, 16^3] and [1, C, 32^3]
    var g1 = Tensor[F32]([1, 512, 16, 16, 16])
    fill(g1, 2)
    var g2 = Tensor[F32]([1, 128, 32, 32, 32])
    fill(g2, 3)
    var gn1 = GroupNorm32(32, 512)
    var gn2 = GroupNorm32(32, 128)
    var cln1 = ChannelLayerNorm32(512)
    var cln2 = ChannelLayerNorm32(128)

    best = 1e30
    _ = gn1.forward(g1)
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = gn1.forward(g1)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
    print("groupnorm [1,512,16^3]        ", best, "ms")

    best = 1e30
    _ = gn2.forward(g2)
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = gn2.forward(g2)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
    print("groupnorm [1,128,32^3]        ", best, "ms")

    best = 1e30
    _ = cln1.forward(g1)
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = cln1.forward(g1)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
    print("chan-ln   [1,512,16^3]        ", best, "ms")

    best = 1e30
    _ = cln2.forward(g2)
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        _ = cln2.forward(g2)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
    print("chan-ln   [1,128,32^3]        ", best, "ms")
