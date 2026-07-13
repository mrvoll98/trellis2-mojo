# Throwaway micro-benchmark: varlen_sdpa at the real ss-flow self-attention
# shape (1 segment x 4096 tokens, H16 D64) and the cross shape (4096 q vs
# 1029 kv). Perf pass 8 attribution. Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.attention.full_attn import varlen_sdpa

comptime F32 = DType.float32
comptime ITERS = 3


def fill(mut t: Tensor[F32], seed: Int) raises:
    for i in range(t.numel()):
        t.data[i] = Float32(((i * 31 + seed * 17) % 41) - 20) * 0.008


def bench(name: String, tq: Int, tkv: Int, h: Int, d: Int) raises:
    var q_shape: List[Int] = [tq, h, d]
    var kv_shape: List[Int] = [tkv, h, d]
    var q = Tensor[F32](q_shape)
    var k = Tensor[F32](kv_shape)
    var v = Tensor[F32](kv_shape)
    fill(q, 1)
    fill(k, 2)
    fill(v, 3)
    var qo: List[Int] = [0, tq]
    var ko: List[Int] = [0, tkv]
    _ = varlen_sdpa(q, k, v, qo, ko)  # warmup
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        var y = varlen_sdpa(q, k, v, qo, ko)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
        _ = y
    var gf = Float64(2 * 2 * tq * tkv * d * h) / (best * 1e6)
    print("  " + name + ": " + String(best) + " ms  " + String(gf) + " GF/s (qk+av)")


def bench2(name: String, tq: Int, tkv: Int, h: Int, ci: Int, co: Int) raises:
    var q_shape: List[Int] = [tq, h, ci]
    var k_shape: List[Int] = [tkv, h, ci]
    var v_shape: List[Int] = [tkv, h, co]
    var q = Tensor[F32](q_shape)
    var k = Tensor[F32](k_shape)
    var v = Tensor[F32](v_shape)
    fill(q, 1)
    fill(k, 2)
    fill(v, 3)
    var qo: List[Int] = [0, tq]
    var ko: List[Int] = [0, tkv]
    _ = varlen_sdpa(q, k, v, qo, ko)
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        var y = varlen_sdpa(q, k, v, qo, ko)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
        _ = y
    print("  " + name + ": " + String(best) + " ms")


def main() raises:
    print("varlen_sdpa @ real shapes (min of 3):")
    bench("self 4096x4096 H16 D64", 4096, 4096, 16, 64)
    bench("cross 4096x1029 H16 D64", 4096, 1029, 16, 64)
    bench("dino self 1029x1029 H16 D64", 1029, 1029, 16, 64)
    print("phase attribution via degenerate dims (self case):")
    bench2("full ci64 co64", 4096, 4096, 16, 64, 64)
    bench2("qk-ish ci64 co1", 4096, 4096, 16, 64, 1)
    bench2("av-ish ci1 co64", 4096, 4096, 16, 1, 64)
    bench2("softmax-ish ci1 co1", 4096, 4096, 16, 1, 1)
