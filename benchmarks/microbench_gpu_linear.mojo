# WP11 step 2: end-to-end GpuLinear.forward (pack + upload + kernel +
# readback) vs the CPU `linear` on the real DiT/DINOv3 shapes. This is the
# honest wiring-level number — the GEMM-only ceiling lives in
# microbench_gpu_gemm.mojo. Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.gpu.linear import GpuContext, GpuLinear, gpu_mlp_forward
from trellis2_mojo.modules.nn import linear, activation, ACT_GELU_TANH
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def fill(mut t: Tensor[F32], seed: Int, scale: Float32):
    var state = seed * 6364136223846793005 + 1442695040888963407
    for i in range(len(t.data)):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 33) & ((1 << 20) - 1)) / Float32(1 << 19)
        t.data[i] = (u - 1.0) * scale


def bench(gpu: Optional[GpuContext], rows: Int, co: Int, ci: Int) raises:
    var x = Tensor[F32]([rows, ci])
    var w = Tensor[F32]([co, ci])
    var b = Tensor[F32]([co])
    fill(x, rows + co, 1.0)
    fill(w, rows + co + 1, 0.03)
    fill(b, rows + co + 2, 0.5)
    var gl = GpuLinear.try_build(gpu, w, b, True)
    if not gl:
        raise Error("shape does not qualify")

    var best_gpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gl.value().forward(x)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_gpu:
            best_gpu = ms

    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = linear(x, w, b, True)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms

    var gf_gpu = Float64(2 * rows) * Float64(co) * Float64(ci) / (best_gpu * 1e6)
    var gf_cpu = Float64(2 * rows) * Float64(co) * Float64(ci) / (best_cpu * 1e6)
    print(
        rows, "x", co, "x", ci, ": gpu", best_gpu, "ms (", gf_gpu,
        "GF/s)  cpu", best_cpu, "ms (", gf_cpu, "GF/s)  speedup",
        best_cpu / best_gpu,
    )


def bench_mlp(gpu: Optional[GpuContext], rows: Int, ci: Int, hid: Int) raises:
    """Chained lin2(gelu(lin0(x))) on GPU (device-resident intermediate,
    WP11 step 5) vs unchained GPU vs the CPU pipeline."""
    var x = Tensor[F32]([rows, ci])
    var w0 = Tensor[F32]([hid, ci])
    var b0 = Tensor[F32]([hid])
    var w2 = Tensor[F32]([ci, hid])
    var b2 = Tensor[F32]([ci])
    fill(x, rows, 1.0)
    fill(w0, rows + 1, 0.03)
    fill(b0, rows + 2, 0.1)
    fill(w2, rows + 3, 0.015)
    fill(b2, rows + 4, 0.1)
    var gl0 = GpuLinear.try_build(gpu, w0, b0, True)
    var gl2 = GpuLinear.try_build(gpu, w2, b2, True)

    var best_chain: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gpu_mlp_forward(x, gl0.value(), gl2.value())
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_chain:
            best_chain = ms

    var best_unchained: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = gl2.value().forward(activation(gl0.value().forward(x), ACT_GELU_TANH))
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_unchained:
            best_unchained = ms

    var best_cpu: Float64 = 1e30
    for it in range(4):
        var t0 = perf_counter_ns()
        var y = linear(activation(linear(x, w0, b0, True), ACT_GELU_TANH), w2, b2, True)
        var t1 = perf_counter_ns()
        _ = y.data[0]
        var ms = Float64(t1 - t0) / 1e6
        if it > 0 and ms < best_cpu:
            best_cpu = ms

    print(
        "mlp ", rows, "x", ci, "->", hid, ": chain", best_chain,
        "ms  unchained-gpu", best_unchained, "ms  cpu", best_cpu,
        "ms  speedup vs cpu", best_cpu / best_chain,
    )


def main() raises:
    var gpu: Optional[GpuContext] = None
    gpu = GpuContext()
    print("GpuLinear.forward (pack+kernel+readback) vs CPU linear (min of 3):")
    bench(gpu, 4096, 3072, 1024)  # DiT qkv
    bench(gpu, 4096, 1024, 1024)  # DiT to_out
    bench(gpu, 4096, 4096, 1024)  # DiT mlp up
    bench(gpu, 4096, 1024, 4096)  # DiT mlp down
    bench(gpu, 1029, 3072, 1024)  # DINOv3-512 tokens, padded M
    bench(gpu, 2369, 4096, 1024)  # smoke-run slat tokens
    bench_mlp(gpu, 4096, 1536, 8192)  # ss_flow real mlp (WP11 step 5)
    bench_mlp(gpu, 2369, 1536, 8192)  # slat real mlp
