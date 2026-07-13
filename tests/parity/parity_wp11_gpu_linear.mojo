# WP11 step 2: GPU linear (tiled Metal GEMM behind SparseLinear) vs the CPU
# `linear` on DiT-realistic shapes. GPU accumulation order differs from both
# CPU paths (sequential over k vs SIMD-lane trees / packed panels), so this
# is a TOLERANCE parity — never bit-equality (pass 5/8 precedent).
#
# Also checks the wiring contract itself: try_build declines shapes the
# kernel can't take (co % 64, ci % 16, tiny weights), the SparseLinear
# dispatch leaves small row counts bit-identical on the CPU path, and the
# StateDict.gpu ride-along attaches device weights through lin_from.
#
# Runs on the Metal GPU; if no device is available the test SKIPS loudly
# (this project is pinned to the M4 Pro machine, so that should not happen).

from trellis2_mojo.gpu.linear import (
    GpuContext,
    GpuLinear,
    GPU_MIN_ROWS,
    WFMT_BF16,
    WFMT_F16,
    WFMT_F16SH,
    WFMT_F32,
    gpu_mlp_forward,
    gpu_mlp_wants,
)
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.loaders import lin_from, sparse_ffn_from
from trellis2_mojo.modules.nn import SparseLinear, linear, activation, ACT_GELU_TANH
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def fill(mut t: Tensor[F32], seed: Int, scale: Float32):
    """Deterministic pseudo-random fill, roughly N(0, scale)-shaped."""
    var state = seed * 6364136223846793005 + 1442695040888963407
    for i in range(len(t.data)):
        state = state * 6364136223846793005 + 1442695040888963407
        # 20 mantissa-ish bits -> [-1, 1) * scale
        var u = Float32((state >> 33) & ((1 << 20) - 1)) / Float32(1 << 19)
        t.data[i] = (u - 1.0) * scale


def make_case(rows: Int, co: Int, ci: Int, seed: Int) raises -> Tuple[Tensor[F32], Tensor[F32], Tensor[F32]]:
    var x = Tensor[F32]([rows, ci])
    var w = Tensor[F32]([co, ci])
    var b = Tensor[F32]([co])
    fill(x, seed, 1.0)
    fill(w, seed + 1, 0.03)  # ~1/sqrt(1024), keeps sums O(1)
    fill(b, seed + 2, 0.5)
    return (x^, w^, b^)


def quant_bf16(mut t: Tensor[F32]):
    """Truncate every value to its bf16-representable form (zero the low
    16 mantissa bits) — the exact shape of a bf16-loaded checkpoint."""
    for i in range(len(t.data)):
        var bits = t.data[i].to_bits[DType.uint32]() & 0xFFFF0000
        t.data[i] = SIMD[F32, 1](from_bits=bits)


def quant_f16(mut t: Tensor[F32]):
    """Round every value through f16 — the exact shape of an fp16-loaded
    checkpoint."""
    for i in range(len(t.data)):
        t.data[i] = t.data[i].cast[DType.float16]().cast[F32]()


def max_diff(a: Tensor[F32], b: Tensor[F32]) raises -> Float32:
    if len(a.data) != len(b.data):
        raise Error("shape mismatch in max_diff")
    var m: Float32 = 0
    for i in range(len(a.data)):
        var d = abs(a.data[i] - b.data[i])
        if d > m:
            m = d
    return m


def check_case(
    gpu: Optional[GpuContext], rows: Int, co: Int, ci: Int, has_bias: Bool, seed: Int, atol: Float32
) raises:
    var t = make_case(rows, co, ci, seed)
    var gl = GpuLinear.try_build(gpu, t[1], t[2], has_bias)
    if not gl:
        raise Error("try_build declined a qualifying shape")
    var cpu = linear(t[0], t[1], t[2], has_bias)
    var dev = gl.value().forward(t[0])
    var d = max_diff(cpu, dev)
    print(
        "  gpu vs cpu ", rows, "x", co, "x", ci,
        " bias:", has_bias, " max|diff|:", d,
    )
    if d > atol:
        raise Error("gpu linear parity failed")


def main() raises:
    var gpu: Optional[GpuContext] = None
    try:
        gpu = GpuContext()
    except:
        print("wp11 gpu linear: SKIPPED — no usable GPU device")
        return

    # DiT-realistic shapes: qkv/out/mlp on 4096 dense tokens, odd sparse
    # token counts (padding path), the adaLN 6144 head, k = 4096 mlp-down
    check_case(gpu, 4096, 3072, 1024, True, 11, 2e-4)
    check_case(gpu, 1029, 1024, 1024, True, 12, 2e-4)
    check_case(gpu, 2369, 4096, 1024, True, 13, 2e-4)
    check_case(gpu, 2369, 1024, 4096, True, 14, 4e-4)
    check_case(gpu, 600, 6144, 1024, False, 15, 2e-4)

    # 3-D x (dense DiT layout [1, L, C]) through SparseLinear dispatch;
    # 1500*3072*1024 clears the GPU_MIN_PROXY flops threshold
    var t3 = make_case(1500, 3072, 1024, 21)
    var x3 = Tensor[F32].from_values([1, 1500, 1024], t3[0].data)
    var sl = SparseLinear(t3[1].copy(), t3[2].copy())
    sl.gpu = GpuLinear.try_build(gpu, sl.weight, sl.bias, sl.has_bias)
    if not sl.gpu:
        raise Error("try_build declined the 3-D case")
    var out3 = sl.forward(x3)
    if out3.shape[0] != 1 or out3.shape[1] != 1500 or out3.shape[2] != 3072:
        raise Error("3-D output shape wrong")
    var d3 = max_diff(out3, linear(x3, t3[1], t3[2], True))
    print("  [1,1500,1024] -> 3072 via SparseLinear dispatch  max|diff|:", d3)
    if d3 > 2e-4:
        raise Error("3-D dispatch parity failed")

    # shapes the kernel can't take must decline (CPU fallback)
    var bad1 = make_case(8, 1000, 1024, 31)  # co % 64 != 0
    if GpuLinear.try_build(gpu, bad1[1], bad1[2], True):
        raise Error("try_build accepted co % 64 != 0")
    var bad2 = make_case(8, 1024, 1000, 32)  # ci % 16 != 0
    if GpuLinear.try_build(gpu, bad2[1], bad2[2], True):
        raise Error("try_build accepted ci % 16 != 0")
    var bad3 = make_case(8, 64, 64, 33)  # tiny weight
    if GpuLinear.try_build(gpu, bad3[1], bad3[2], True):
        raise Error("try_build accepted a tiny weight")

    # small row counts stay on the CPU path bit-identically
    var ts = make_case(GPU_MIN_ROWS - 1, 1024, 1024, 41)
    var sls = SparseLinear(ts[1].copy(), ts[2].copy())
    sls.gpu = GpuLinear.try_build(gpu, sls.weight, sls.bias, sls.has_bias)
    var small_gpu = sls.forward(ts[0])
    var small_cpu = linear(ts[0], ts[1], ts[2], True)
    if max_diff(small_gpu, small_cpu) != 0:
        raise Error("small-rows dispatch did not take the exact CPU path")
    print("  small rows (", GPU_MIN_ROWS - 1, ") stay on CPU: bit-identical")

    # StateDict ride-along: lin_from attaches device weights when sd.gpu set
    # (1029*4096*1024 clears the flops threshold, so forward runs on GPU)
    var tl = make_case(1029, 4096, 1024, 51)
    var d = Dict[String, Tensor[F32]]()
    d["blk.weight"] = tl[1].copy()
    d["blk.bias"] = tl[2].copy()
    var sd = StateDict(d^)
    sd.gpu = gpu.copy()
    var sll = lin_from(sd, "blk")
    if not sll.gpu:
        raise Error("lin_from did not attach GPU weights")
    var dl = max_diff(sll.forward(tl[0]), linear(tl[0], tl[1], tl[2], True))
    print("  lin_from + StateDict.gpu ride-along  max|diff|:", dl)
    if dl > 2e-4:
        raise Error("lin_from GPU parity failed")

    # WP11 step 5: chained mlp (lin2(gelu_tanh(lin0(x))) with the hidden
    # intermediate device-resident). GPU gelu computes tanh through exp;
    # tolerance covers that + two chained GEMM accumulations.
    var rows = 1500
    comptime CI = 1024
    comptime HID = 4096
    var xm = Tensor[F32]([rows, CI])
    var w0 = Tensor[F32]([HID, CI])
    var b0 = Tensor[F32]([HID])
    var w2 = Tensor[F32]([CI, HID])
    var b2 = Tensor[F32]([CI])
    fill(xm, 91, 1.0)
    fill(w0, 92, 0.03)
    fill(b0, 93, 0.1)
    fill(w2, 94, 0.015)
    fill(b2, 95, 0.1)
    var gl0 = GpuLinear.try_build(gpu, w0, b0, True)
    var gl2 = GpuLinear.try_build(gpu, w2, b2, True)
    if not gl0 or not gl2:
        raise Error("mlp chain: try_build declined")
    if not gpu_mlp_wants(gl0.value(), gl2.value(), rows):
        raise Error("mlp chain: wants rejected the qualifying case")
    var cpu_mid = activation(linear(xm, w0, b0, True), ACT_GELU_TANH)
    var cpu_mlp = linear(cpu_mid, w2, b2, True)
    var dev_mlp = gpu_mlp_forward(xm, gl0.value(), gl2.value())
    var dm = max_diff(cpu_mlp, dev_mlp)
    print("  mlp chain ", rows, "x", CI, "->", HID, "->", CI, "  max|diff|:", dm)
    if dm > 5e-4:
        raise Error("mlp chain parity failed")

    # dispatch integration: sparse_ffn_from + sd.gpu
    var df = Dict[String, Tensor[F32]]()
    df["mlp.mlp.0.weight"] = w0.copy()
    df["mlp.mlp.0.bias"] = b0.copy()
    df["mlp.mlp.2.weight"] = w2.copy()
    df["mlp.mlp.2.bias"] = b2.copy()
    var fsd = StateDict(df^)
    var ffn_cpu = sparse_ffn_from(fsd, "mlp")
    fsd.gpu = gpu.copy()
    var ffn_gpu = sparse_ffn_from(fsd, "mlp")
    if not ffn_gpu.lin1.gpu or not ffn_gpu.lin2.gpu:
        raise Error("sparse_ffn_from did not attach GPU weights")
    var dff = max_diff(ffn_cpu.forward_dense(xm), ffn_gpu.forward_dense(xm))
    print("  SparseFeedForwardNet dispatch  max|diff|:", dff)
    if dff > 5e-4:
        raise Error("ffn dispatch parity failed")

    # WP11 step 14: 16-bit device weight storage. Exactly-representable
    # weights must select the 16-bit format and produce BIT-identical
    # results to f32 storage of the same values (the expansion on the
    # shared-memory fill is exact); mixed weights must stay f32.
    var tq = make_case(1029, 4096, 1024, 61)
    var wbf = tq[1].copy()
    quant_bf16(wbf)
    var gl_bf = GpuLinear.try_build(gpu, wbf, tq[2], True)
    if not gl_bf or gl_bf.value().wfmt != WFMT_BF16:
        raise Error("bf16-exact weight did not select bf16 storage")
    var gl_bf32 = GpuLinear.try_build(gpu, wbf, tq[2], True, allow_16bit=False)
    if gl_bf32.value().wfmt != WFMT_F32:
        raise Error("allow_16bit=False did not force f32 storage")
    var y_bf = gl_bf.value().forward(tq[0])
    if max_diff(y_bf, gl_bf32.value().forward(tq[0])) != 0:
        raise Error("bf16 storage is not bit-identical to f32 storage")
    var d_bf = max_diff(y_bf, linear(tq[0], wbf, tq[2], True))
    print("  bf16-stored W 1029x4096x1024 vs cpu  max|diff|:", d_bf, " (vs f32-stored: bit-identical)")
    if d_bf > 2e-4:
        raise Error("bf16-stored parity failed")

    var wf16 = tq[1].copy()
    quant_f16(wf16)
    var gl_f16 = GpuLinear.try_build(gpu, wf16, tq[2], True)
    if not gl_f16 or gl_f16.value().wfmt != WFMT_F16:
        raise Error("f16-exact weight did not select f16 storage")
    var gl_h32 = GpuLinear.try_build(gpu, wf16, tq[2], True, allow_16bit=False)
    var y_f16 = gl_f16.value().forward(tq[0])
    if max_diff(y_f16, gl_h32.value().forward(tq[0])) != 0:
        raise Error("f16 storage is not bit-identical to f32 storage")
    var d_f16 = max_diff(y_f16, linear(tq[0], wf16, tq[2], True))
    print("  f16-stored W 1029x4096x1024 vs cpu  max|diff|:", d_f16, " (vs f32-stored: bit-identical)")
    if d_f16 > 2e-4:
        raise Error("f16-stored parity failed")

    # a single non-representable element must force f32 storage
    var wmix = wbf.copy()
    wmix.data[12345] = 0.1  # f32(0.1) has low mantissa bits set
    var gl_mix = GpuLinear.try_build(gpu, wmix, tq[2], True)
    if gl_mix.value().wfmt != WFMT_F32:
        raise Error("mixed weight did not fall back to f32 storage")
    print("  mixed weight falls back to f32 storage")

    # WP19 (TRELLIS2_GPU_F16): f16-shared-kjernen — 16-bits-kvalifiserte
    # vekter velger WFMT_F16SH når kontekst-flagget er satt; utgangen er
    # IKKE bit-eksakt (A castes til f16 på shared-fyllet) men innenfor
    # ulp-klassen (~1e-4 målt på probe-datene; toleranse 2e-3)
    var g16 = gpu.value().copy()
    g16.f16 = True
    var gpu16 = Optional[GpuContext](g16^)
    var gl_sh = GpuLinear.try_build(gpu16, wbf, tq[2], True)
    if not gl_sh or gl_sh.value().wfmt != WFMT_F16SH:
        raise Error("f16-flagget valgte ikke WFMT_F16SH for bf16-vekt")
    var y_sh = gl_sh.value().forward(tq[0])
    var d_sh = max_diff(y_sh, linear(tq[0], wbf, tq[2], True))
    print("  f16-shared W (WP19-flagg) vs cpu  max|diff|:", d_sh)
    if d_sh > 2e-3:
        raise Error("f16-shared parity failed")
    if max_diff(y_sh, y_bf) == 0:
        raise Error("f16-shared burde avvike fra f32-stien (A-cast) — dispatch inaktiv?")
    # flagget av -> uendret klassifisering
    var gl_off = GpuLinear.try_build(gpu, wbf, tq[2], True)
    if gl_off.value().wfmt != WFMT_BF16:
        raise Error("uten flagget må bf16-vekten fortsatt velge WFMT_BF16")
    print("  f16-shared: opt-in via kontekst-flagget, av som default")

    # the chained-enqueue path (mlp; the attention chains share the same
    # enqueue_gemm dispatch) with bf16-stored weights: bit-identical to
    # the f32-stored chain on the same quantized values
    var w0q = w0.copy()
    var w2q = w2.copy()
    quant_bf16(w0q)
    quant_bf16(w2q)
    var gl0q = GpuLinear.try_build(gpu, w0q, b0, True)
    var gl2q = GpuLinear.try_build(gpu, w2q, b2, True)
    if gl0q.value().wfmt != WFMT_BF16 or gl2q.value().wfmt != WFMT_BF16:
        raise Error("mlp chain: quantized weights did not select bf16")
    var gl0q32 = GpuLinear.try_build(gpu, w0q, b0, True, allow_16bit=False)
    var gl2q32 = GpuLinear.try_build(gpu, w2q, b2, True, allow_16bit=False)
    var mlp_bf = gpu_mlp_forward(xm, gl0q.value(), gl2q.value())
    if max_diff(mlp_bf, gpu_mlp_forward(xm, gl0q32.value(), gl2q32.value())) != 0:
        raise Error("bf16 mlp chain is not bit-identical to f32 storage")
    var cpu_midq = activation(linear(xm, w0q, b0, True), ACT_GELU_TANH)
    var dmq = max_diff(linear(cpu_midq, w2q, b2, True), mlp_bf)
    print("  bf16-stored mlp chain vs cpu  max|diff|:", dmq, " (vs f32-stored: bit-identical)")
    if dmq > 5e-4:
        raise Error("bf16 mlp chain parity failed")

    print("wp11 gpu-linear parity vs cpu: 5 shapes + 3-D dispatch + declines + ride-along + mlp chain + 16-bit storage passed")
