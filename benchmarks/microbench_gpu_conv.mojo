# WP11 step 6: GPU sparse conv (gather kernel incl. CSR build + transfers)
# vs the CPU SparseConv3d on decode-realistic shapes (dense blobs — the
# real decode coords are surface-ish but similar edge counts per token at
# the coarse levels). Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.conv import GpuSparseConv
from trellis2_mojo.sparse.conv import SparseConv3d
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32


def fill(mut t: Tensor[F32], seed: Int, scale: Float32):
    var state = seed * 6364136223846793005 + 1442695040888963407
    for i in range(len(t.data)):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & ((1 << 20) - 1)) / Float32(1 << 19)
        t.data[i] = (u - 1.0) * scale


def blob(side: Int) raises -> IntMatrix:
    var n = side * side * side
    var coords = IntMatrix(n, 4)
    var r = 0
    for z in range(side):
        for y in range(side):
            for x in range(side):
                coords.set(r, 1, x + 4)
                coords.set(r, 2, y + 4)
                coords.set(r, 3, z + 4)
                r += 1
    return coords^


def bench(gpu: Optional[GpuContext], side: Int, ci: Int, co: Int) raises:
    var coords = blob(side)
    var n = coords.rows
    var feats = Tensor[F32]([n, ci])
    fill(feats, n, 1.0)
    var w = Tensor[F32]([co, 3, 3, 3, ci])
    var b = Tensor[F32]([co])
    fill(w, n + 1, 0.05)
    fill(b, n + 2, 0.1)
    # f16-quantize so the f16-storage path is measurable (WP11 step 15) —
    # the real decoder weights are fp16 checkpoints, so this IS the real
    # shape of the data; the f32 kernel is timed on the same values via
    # allow_16bit=False
    for i in range(len(w.data)):
        w.data[i] = w.data[i].cast[DType.float16]().cast[F32]()
    var conv_cpu = SparseConv3d(w.copy(), b.copy())
    var conv_gpu = SparseConv3d(w.copy(), b.copy())
    conv_gpu.gpu = GpuSparseConv.try_build(gpu, conv_gpu.weight, conv_gpu.bias, True)
    var conv_g32 = SparseConv3d(w.copy(), b.copy())
    conv_g32.gpu = GpuSparseConv.try_build(
        gpu, conv_g32.weight, conv_g32.bias, True, allow_16bit=False
    )
    var x = SparseTensor[F32](feats.copy(), coords.copy(), 1)
    _ = conv_cpu.forward(x)  # warm the neighbor-map cache for both

    var best_gpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = conv_gpu.forward(x)
        var t1 = perf_counter_ns()
        _ = y.vl.feats.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_gpu:
            best_gpu = ms

    var best_g32: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = conv_g32.forward(x)
        var t1 = perf_counter_ns()
        _ = y.vl.feats.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_g32:
            best_g32 = ms

    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = conv_cpu.forward(x)
        var t1 = perf_counter_ns()
        _ = y.vl.feats.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms

    print(
        "n", n, " ci", ci, " co", co, ": gpu-f16", best_gpu, "ms  gpu-f32",
        best_g32, "ms  cpu", best_cpu, "ms  f16-vs-f32", best_g32 / best_gpu,
        " vs-cpu", best_cpu / best_gpu,
    )


def main() raises:
    var gpu: Optional[GpuContext] = None
    gpu = GpuContext()
    print("GPU sparse conv (CSR+transfers+kernel, f16- vs f32-stored W) vs CPU (min of 3):")
    bench(gpu, 23, 512, 512)   # ~12k tokens — stage-1 convnext
    bench(gpu, 38, 256, 256)   # ~55k tokens — stage-2 convnext
    bench(gpu, 60, 128, 128)   # ~216k tokens — stage-3 convnext
    bench(gpu, 23, 512, 2048)  # up-block conv1 (C -> 8*C_out)
