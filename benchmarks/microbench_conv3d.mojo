# Throwaway micro-benchmark: dense Conv3d at the real SS-VAE decoder
# shapes (perf pass 7). Not wired into pixi tasks.

from std.time import perf_counter_ns

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.modules.conv import Conv3d

comptime F32 = DType.float32
comptime ITERS = 3


def bench(name: String, n: Int, ci: Int, co: Int, g: Int, k: Int, p: Int) raises:
    var x_shape: List[Int] = [n, ci, g, g, g]
    var x = Tensor[F32](x_shape)
    for i in range(x.numel()):
        x.data[i] = Float32(((i * 31) % 41) - 20) * 0.008
    var w_shape: List[Int] = [co, ci, k, k, k]
    var w = Tensor[F32](w_shape)
    for i in range(w.numel()):
        w.data[i] = Float32(((i * 17) % 29) - 14) * 0.01
    var b = Tensor[F32]([co])
    var conv = Conv3d(w^, b^, 1, p)
    _ = conv.forward(x)  # warmup
    var best: Float64 = 1e30
    for _ in range(ITERS):
        var t0 = perf_counter_ns()
        var y = conv.forward(x)
        var t1 = perf_counter_ns()
        var ms = Float64(t1 - t0) / 1e6
        if ms < best:
            best = ms
        _ = y
    var gf = Float64(2 * n * co * ci * k * k * k * g * g * g) / (best * 1e6)
    print("  " + name + ": " + String(best) + " ms  " + String(gf) + " GF/s")


def main() raises:
    print("dense Conv3d @ SS-VAE decoder shapes (min of 3):")
    bench("input 8->512 @16^3 k3", 1, 8, 512, 16, 3, 1)
    bench("res 512->512 @16^3 k3", 1, 512, 512, 16, 3, 1)
    bench("upsample 512->1024 @16^3 k3", 1, 512, 1024, 16, 3, 1)
    bench("res 128->128 @32^3 k3", 1, 128, 128, 32, 3, 1)
    bench("upsample 128->256 @32^3 k3", 1, 128, 256, 32, 3, 1)
    bench("res 32->32 @64^3 k3", 1, 32, 32, 64, 3, 1)
    bench("out 32->1 @64^3 k3", 1, 32, 1, 64, 3, 1)
    bench("skip 512->128 @16^3 k1", 1, 512, 128, 16, 1, 0)
    print("pre-pass-7 baseline (single measurement):")
    naive_once(1, 512, 512, 16, 3, 1)


def naive_once(n: Int, ci: Int, co: Int, g: Int, k: Int, p: Int) raises:
    """The pre-pass-7 scalar single-thread loop, measured once."""
    var x_shape: List[Int] = [n, ci, g, g, g]
    var x = Tensor[F32](x_shape)
    for i in range(x.numel()):
        x.data[i] = Float32(((i * 31) % 41) - 20) * 0.008
    var w_shape: List[Int] = [co, ci, k, k, k]
    var wt = Tensor[F32](w_shape)
    for i in range(wt.numel()):
        wt.data[i] = Float32(((i * 17) % 29) - 14) * 0.01
    var bias = Tensor[F32]([co])
    var h = g
    var w = g
    var d = g
    var s = 1
    var oh = (h + 2 * p - k) // s + 1
    var ow = (w + 2 * p - k) // s + 1
    var od = (d + 2 * p - k) // s + 1
    var out_shape: List[Int] = [n, co, oh, ow, od]
    var out = Tensor[F32](out_shape)
    var t0 = perf_counter_ns()
    for b in range(n):
        for o in range(co):
            for zh in range(oh):
                for zw in range(ow):
                    for zd in range(od):
                        var acc: Float32 = bias.data[o]
                        for c in range(ci):
                            for kh in range(k):
                                var ih = zh * s - p + kh
                                if ih < 0 or ih >= h:
                                    continue
                                for kw in range(k):
                                    var iw = zw * s - p + kw
                                    if iw < 0 or iw >= w:
                                        continue
                                    for kd in range(k):
                                        var idd = zd * s - p + kd
                                        if idd < 0 or idd >= d:
                                            continue
                                        acc += (
                                            wt.data[(((o * ci + c) * k + kh) * k + kw) * k + kd]
                                            * x.data[(((b * ci + c) * h + ih) * w + iw) * d + idd]
                                        )
                        out.data[(((b * co + o) * oh + zh) * ow + zw) * od + zd] = acc
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1e6
    var gf = Float64(2 * n * co * ci * k * k * k * g * g * g) / (ms * 1e6)
    print("  naive res " + String(ci) + "->" + String(co) + " @" + String(g) + "^3: " + String(ms) + " ms  " + String(gf) + " GF/s")
