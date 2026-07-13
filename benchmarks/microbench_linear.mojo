# Throwaway micro-benchmark: linear() throughput on the mod-block and
# sampler shapes, to attribute where mod-block time goes. Not wired into
# pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.modules.nn import linear

comptime F32 = DType.float32
comptime ITERS = 5


def bench_shape(rows: Int, ci: Int, co: Int) raises:
    var x = Tensor[F32]([rows, ci])
    var w = Tensor[F32]([co, ci])
    var b = Tensor[F32]([co])
    for i in range(x.numel()):
        x.data[i] = Float32((i % 37) - 18) * 0.01
    for i in range(w.numel()):
        w.data[i] = Float32((i % 29) - 14) * 0.01
    for i in range(co):
        b.data[i] = Float32(i % 7) * 0.1
    _ = linear(x, w, b)  # warmup
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        var y = linear(x, w, b)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
        _ = y
    var gflops = Float64(2 * rows * ci * co) / (best * 1e6)
    print(rows, "x", ci, "->", co, ": ", best, "ms  ", gflops, "GF/s")


def main() raises:
    print("mod-block shapes (rows=1024):")
    bench_shape(1024, 128, 384)  # qkv
    bench_shape(1024, 128, 128)  # attn out
    bench_shape(1024, 128, 512)  # ffn1
    bench_shape(1024, 512, 128)  # ffn2
    print("sampler shapes (rows=256):")
    bench_shape(256, 64, 192)
    bench_shape(256, 64, 256)
    bench_shape(256, 256, 64)
    print("big square:")
    bench_shape(4096, 512, 512)
    print("real DiT shapes (ss-flow: 4096 tok, C=1024, ffn 4096):")
    bench_shape(4096, 1024, 3072)  # qkv
    bench_shape(4096, 1024, 1024)  # attn out
    bench_shape(4096, 1024, 4096)  # ffn1
    bench_shape(4096, 4096, 1024)  # ffn2
    print("real DINOv3 shapes (1029 tok, C=1024, ffn 4096):")
    bench_shape(1029, 1024, 1024)  # q/k/v/o
    bench_shape(1029, 1024, 4096)  # mlp up
    bench_shape(1029, 4096, 1024)  # mlp down
