# WP11 step 6: GPU sparse conv (gather kernel over the CSR-sorted edge
# lists, gpu/conv.mojo) vs the CPU SparseConv3d. The GPU accumulates
# scalar-x-vec4 per channel where the CPU does a SIMD-tree dot per edge ->
# tolerance parity, never bit-equality. Also checks the try_build gate,
# the flops gate (small inputs stay on the bit-exact CPU path) and the
# sparse_conv3d_from + StateDict.gpu wiring.
#
# Runs on the Metal GPU; SKIPS loudly if no device is available.

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.conv import GpuSparseConv
from trellis2_mojo.gpu.linear import WFMT_F16, WFMT_F32
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.loaders import sparse_conv3d_from
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


def max_diff(a: Tensor[F32], b: Tensor[F32]) raises -> Float32:
    if len(a.data) != len(b.data):
        raise Error("shape mismatch in max_diff")
    var m: Float32 = 0
    for i in range(len(a.data)):
        var d = abs(a.data[i] - b.data[i])
        if d > m:
            m = d
    return m


def blob(side: Int) raises -> IntMatrix:
    """Dense side^3 block of coords (batch 0) — every interior voxel gets
    the full 27-neighbor edge set."""
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


def check_case(
    gpu: Optional[GpuContext], side: Int, ci: Int, co: Int, dilation: Int,
    seed: Int, atol: Float32
) raises:
    var coords = blob(side)
    var n = coords.rows
    var feats = Tensor[F32]([n, ci])
    fill(feats, seed, 1.0)
    var w = Tensor[F32]([co, 3, 3, 3, ci])
    var b = Tensor[F32]([co])
    fill(w, seed + 1, 0.05)
    fill(b, seed + 2, 0.1)

    var conv_cpu = SparseConv3d(w.copy(), b.copy(), dilation=dilation)
    var conv_gpu = SparseConv3d(w.copy(), b.copy(), dilation=dilation)
    conv_gpu.gpu = GpuSparseConv.try_build(gpu, conv_gpu.weight, conv_gpu.bias, True)
    if not conv_gpu.gpu:
        raise Error("try_build declined a qualifying conv")

    var x = SparseTensor[F32](feats.copy(), coords.copy(), 1)
    var y_cpu = conv_cpu.forward(x)
    var y_gpu = conv_gpu.forward(x)
    var d = max_diff(y_cpu.vl.feats, y_gpu.vl.feats)
    print(
        "  gpu vs cpu conv  n", n, " ci", ci, " co", co,
        " dil", dilation, " max|diff|:", d,
    )
    if d > atol:
        raise Error("gpu sparse conv parity failed")


def main() raises:
    var gpu: Optional[GpuContext] = None
    try:
        gpu = GpuContext()
    except:
        print("wp11 gpu conv: SKIPPED — no usable GPU device")
        return

    # decode-realistic channel combos on a dense blob (full 27-edge rows)
    check_case(gpu, 15, 256, 256, 1, 41, 2e-4)   # convnext mid level
    check_case(gpu, 12, 128, 512, 1, 42, 2e-4)   # rectangular (up conv1-ish)
    check_case(gpu, 12, 512, 256, 1, 43, 5e-4)   # wide ci
    check_case(gpu, 15, 256, 256, 2, 44, 2e-4)   # dilation 2

    # gates: co % 64 declines at build; small edge counts stay on CPU
    var wbad = Tensor[F32]([100, 3, 3, 3, 64])
    var bbad = Tensor[F32]([100])
    if GpuSparseConv.try_build(gpu, wbad, bbad, True):
        raise Error("try_build accepted co % 64 != 0")
    var wsmall = Tensor[F32]([64, 3, 3, 3, 64])
    var bsmall = Tensor[F32]([64])
    var gsmall = GpuSparseConv.try_build(gpu, wsmall, bsmall, True)
    if not gsmall:
        raise Error("try_build declined a tiling-ok conv")
    if gsmall.value().wants(1000 * 27):
        raise Error("wants accepted a transfer-bound edge count")

    # small conv through the dispatch must stay bit-identical CPU
    var coords_s = blob(6)
    var feats_s = Tensor[F32]([coords_s.rows, 64])
    fill(feats_s, 51, 1.0)
    var conv_a = SparseConv3d(wsmall.copy(), bsmall.copy())
    var conv_b = SparseConv3d(wsmall.copy(), bsmall.copy())
    conv_b.gpu = gsmall.copy()
    var xs = SparseTensor[F32](feats_s.copy(), coords_s.copy(), 1)
    if max_diff(conv_a.forward(xs).vl.feats, conv_b.forward(xs).vl.feats) != 0:
        raise Error("small conv did not take the exact CPU path")
    print("  small conv stays on CPU: bit-identical")

    # sparse_conv3d_from + StateDict.gpu ride-along
    var wl = Tensor[F32]([256, 3, 3, 3, 256])
    var bl = Tensor[F32]([256])
    fill(wl, 61, 0.05)
    fill(bl, 62, 0.1)
    var d = Dict[String, Tensor[F32]]()
    d["conv.weight"] = wl.copy()
    d["conv.bias"] = bl.copy()
    var sd = StateDict(d^)
    var loaded_cpu = sparse_conv3d_from(sd, "conv")
    sd.gpu = gpu.copy()
    var loaded_gpu = sparse_conv3d_from(sd, "conv")
    if not loaded_gpu.gpu:
        raise Error("sparse_conv3d_from did not attach the GPU conv")
    var coords_l = blob(15)
    var feats_l = Tensor[F32]([coords_l.rows, 256])
    fill(feats_l, 63, 1.0)
    var xl = SparseTensor[F32](feats_l.copy(), coords_l.copy(), 1)
    var dl = max_diff(loaded_cpu.forward(xl).vl.feats, loaded_gpu.forward(xl).vl.feats)
    print("  sparse_conv3d_from ride-along  max|diff|:", dl)
    if dl > 2e-4:
        raise Error("conv ride-along parity failed")

    # WP11 step 15: f16 weight storage. Exactly-f16 weights (the fp16
    # decoder checkpoints' shape) must select f16 storage and produce
    # BIT-identical results to f32 storage of the same values (the
    # hardware-cast expansion is exact); non-representable weights must
    # stay f32.
    var coords_q = blob(15)
    var feats_q = Tensor[F32]([coords_q.rows, 256])
    fill(feats_q, 71, 1.0)
    var wq = Tensor[F32]([256, 3, 3, 3, 256])
    var bq = Tensor[F32]([256])
    fill(wq, 72, 0.05)
    fill(bq, 73, 0.1)
    for i in range(len(wq.data)):
        wq.data[i] = wq.data[i].cast[DType.float16]().cast[F32]()
    var conv_f16 = SparseConv3d(wq.copy(), bq.copy())
    conv_f16.gpu = GpuSparseConv.try_build(gpu, conv_f16.weight, conv_f16.bias, True)
    if conv_f16.gpu.value().wfmt != WFMT_F16:
        raise Error("f16-exact conv weight did not select f16 storage")
    var conv_f32 = SparseConv3d(wq.copy(), bq.copy())
    conv_f32.gpu = GpuSparseConv.try_build(
        gpu, conv_f32.weight, conv_f32.bias, True, allow_16bit=False
    )
    if conv_f32.gpu.value().wfmt != WFMT_F32:
        raise Error("allow_16bit=False did not force f32 conv storage")
    var xq = SparseTensor[F32](feats_q.copy(), coords_q.copy(), 1)
    var y16 = conv_f16.forward(xq)
    if max_diff(y16.vl.feats, conv_f32.forward(xq).vl.feats) != 0:
        raise Error("f16 conv storage is not bit-identical to f32 storage")
    var conv_ref = SparseConv3d(wq.copy(), bq.copy())
    var dq = max_diff(y16.vl.feats, conv_ref.forward(xq).vl.feats)
    print("  f16-stored conv weight vs cpu  max|diff|:", dq, " (vs f32-stored: bit-identical)")
    if dq > 2e-4:
        raise Error("f16-stored conv parity failed")
    var wmix = wq.copy()
    wmix.data[999] = 0.1  # f32(0.1) is not f16-exact
    var gmix = GpuSparseConv.try_build(gpu, wmix, bq, True)
    if gmix.value().wfmt != WFMT_F32:
        raise Error("mixed conv weight did not fall back to f32 storage")
    print("  mixed conv weight falls back to f32 storage")

    print("wp11 gpu-conv parity vs cpu: 4 shapes + gates + CPU-fallback + ride-along + f16 storage passed")
