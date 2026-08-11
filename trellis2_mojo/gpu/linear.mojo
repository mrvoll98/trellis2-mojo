# WP11 step 2: the tiled Metal GEMM (benchmarks/microbench_gpu_gemm.mojo)
# wired behind `linear`. One shared DeviceContext + scratch per run
# (GpuContext, created from the TRELLIS2_GPU=1 env flag and passed into the
# checkpoint loaders); per-weight W^T device buffers uploaded once at model
# load (GpuLinear, attached to SparseLinear by lin_from/_lin_from). The
# bias is added on the CPU during readback (fused into the staging->out
# pass) — folding it into the GEMM cost a per-row repack of x, and the
# argument-marshalling limit below rules out a separate bias pointer.
# Step 14: W^T is stored in 16 bits (bf16/f16 bits, expanded on the
# shared-memory fill) whenever that is bit-exact for the given weights —
# see the WFMT_* note below; every weight-GEMM call site dispatches
# through GpuLinear.enqueue_gemm.
#
# Empirical 1.0.0b2-Metal laws (all found the hard way — probes 2026-07-10):
#  * Argument marshalling breaks above 4 total bindings, where ALL scalar
#    args together count as one binding: 3 pointers + scalars works,
#    4 pointers + 0 scalars works, 4 pointers + scalars or 5+ pointers
#    hands the kernel garbage addresses.
#  * Nothing commits until a map_to_host of a HOST-WRITTEN buffer:
#    ctx.synchronize() does NOT commit pending work (a kernel enqueued and
#    "synchronized" has not necessarily run), and mapping a buffer that was
#    never host-written returns stale data forever. The 1-element fence
#    buffer below (host-written at creation and re-written inside every
#    barrier map) is the commit+wait primitive; it also orders
#    enqueue_copy d2h transfers (verified).
#  * The mapped host pointer is write-combined (uncached) memory: WRITES
#    are fine (~8 GB/s serial), single-threaded READS are catastrophic
#    (~2.2-2.6 GB/s — and enqueue_copy d2h uses the same slow reads
#    internally, synchronously). Map enter/exit themselves are ~1-2 ms.
#    Mitigation: PARALLEL reads scale ~4.2x (9.2 GB/s at 8+ chunks), so
#    the readback is a chunked-parallel fused bias+copy straight off the
#    mapped pointer — no staging, no second copy.
#  * enqueue_copy(Span, buf)/(buf, Span) exist BUT the Span length is
#    IGNORED — both directions always move the full device-buffer length.
#    Never point them at an allocation smaller than the buffer (heap
#    corruption d2h, page-fault overread h2d).
#  * Bulk writes through h.unsafe_ptr()/memcpy inside map_to_host are fine.
#  * DeviceContext/DeviceBuffer are copyable handles; buffers created on
#    one handle-copy are usable from another.
#  * (6th law, probed 2026-07-11) Kernel writes past a 4 GiB byte offset
#    within ONE buffer binding are SILENTLY LOST (allocation succeeds,
#    reads below the line are fine) — no single binding may exceed
#    2^30 f32 elements. The sdpa scores buffer is head-group-chunked for
#    this reason (gpu/attention.mojo, WP17).
#
# Numerics: each output element accumulates sequentially over k on the GPU
# (bit-identical to a naive serial CPU dot; the bias lands at the end like
# both CPU paths), which is a DIFFERENT order than the SIMD-tree/packed
# CPU paths in modules/nn.mojo -> parity on tolerance, never bit-equality
# (pass 5/8 precedent). M is padded to 64 (pad rows are computed but never
# read back; the GEMM is row-independent); requires co % 64 == 0 and
# ci % 16 == 0 — anything else stays on the CPU path.

from max.algorithm import parallelize
from std.gpu import thread_idx, block_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import  AddressSpace
from std.memory import stack_allocation, unsafe_memcpy

# re-exported so existing `from trellis2_mojo.gpu.linear import GpuContext,
# gpu_context_from_env` call sites keep compiling (context/scratches moved
# to gpu/context.mojo when the attention path arrived — WP11 step 3)
from trellis2_mojo.gpu.context import GpuContext, GpuScratch, gpu_context_from_env
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32
comptime U16 = DType.uint16
comptime U32 = DType.uint32

comptime BM = 64
comptime BN = 64
comptime BK = 16
comptime TM = 4
comptime TN = 4

# Device W^T storage formats (WP11 step 14). 16-bit storage is used ONLY
# when every f32 weight is bit-exactly representable in the 16-bit form
# (bf16: low 16 mantissa bits zero — true for every bf16-loaded checkpoint,
# i.e. all the DiTs; f16: exact round-trip — the fp16 unet-decoder
# checkpoints), so the GEMM expands to IDENTICAL f32 values and the result
# is bit-identical to f32 storage. Anything else (random parity-test
# weights, f32 checkpoints) stays f32. Measured FLAT on the DiT shapes
# (B tiles are L2-cached — step 11 negatives): this buys device memory
# (half the W^T footprint on unified memory) and upload time, not speed.
comptime WFMT_F32 = 0
comptime WFMT_BF16 = 1
comptime WFMT_F16 = 2
# WP19 (TRELLIS2_GPU_F16=1): W^T lagres som f16-bits OG shared-flisene
# holdes i f16 (A castes på shared-fyllet) — probene i
# microbench_gpu_gemm målte +32-40 % (kjernen var shared-båndbredde-
# bundet; halv-MULTIPLIKASJON og dobbel-buffring var derimot målt døde).
# Numerikk: A-casten er ~4.9e-4 rel per element, målt ~2-4e-5 på
# akkumulert utgang; bf16-vekter mister kun subnormaler < 6e-8
# (absoluttfeil, neglisjerbar). IKKE bit-eksakt -> egen toleranseklasse.
comptime WFMT_F16SH = 3

# below this many rows the fixed overhead (~0.1-0.3 ms enqueue + copies)
# beats the GPU win over the 830 GF/s CPU packed GEMM
comptime GPU_MIN_ROWS = 512
# rows*co*ci below this is transfer-bound and loses to the CPU (measured:
# 4096x1024x1024 = 2^32 is the break-even, 1.06x; 1029x3072x1024 = 0.87x)
comptime GPU_MIN_PROXY = 1 << 32
# weights smaller than this never see rows that qualify; skip the upload
comptime GPU_MIN_WEIGHT = 1 << 19


def gemm_tiled(
    a: Pointer[Scalar[F32], ImmutAnyOrigin],
    b: Pointer[Scalar[F32], ImmutAnyOrigin],
    c: Pointer[Scalar[F32], MutAnyOrigin],
    m_dev: Int32, n_dev: Int32, kdim_dev: Int32,
):
    """C[m, n] = A[m, kdim] @ B[kdim, n]; 64x64 threadgroup tiles in shared
    memory, 4x4 register block per thread, cooperative 4-wide loads.
    Requires m % 64 == n % 64 == kdim % 16 == 0."""
    var m = Int(m_dev)
    var n = Int(n_dev)
    var kdim = Int(kdim_dev)

    var As = stack_allocation[
        BM * BK, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[F32, TN](0)
    var acc1 = SIMD[F32, TN](0)
    var acc2 = SIMD[F32, TN](0)
    var acc3 = SIMD[F32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.unsafe_store(ia, a.unsafe_load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        Bs.unsafe_store(ib, b.unsafe_load[width=4](b_src))
        barrier()
        for kk in range(BK):
            var bv = Bs.unsafe_load[width=TN](kk * BN + tx * TN)
            acc0 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 3) * BK + kk]) * bv
        barrier()
    c.unsafe_store((row0 + 0) * n + col0, acc0)
    c.unsafe_store((row0 + 1) * n + col0, acc1)
    c.unsafe_store((row0 + 2) * n + col0, acc2)
    c.unsafe_store((row0 + 3) * n + col0, acc3)


def wfmt_scan(w: Tensor[F32]) raises -> Tuple[Bool, Bool]:
    """(bf16_exact, f16_exact) for the weight tensor — parallel SIMD
    or-reduce; a format qualifies only when EVERY element's 16-bit form
    expands back to the identical f32 bit pattern. One cached-memory read
    pass, cheaper than the WC pack writes it lets callers halve. Shared by
    GpuLinear (bf16 preferred, then f16) and GpuSparseConv (f16 only)."""
    var wp = w.data.unsafe_ptr()
    var n = len(w.data)
    comptime SCH = 1 << 20
    var nsc = (n + SCH - 1) // SCH
    var bf_bad = List[Int](length=nsc, fill=0)
    var f16_bad = List[Int](length=nsc, fill=0)

    @parameter
    def scan(sc: Int):
        var lo = sc * SCH
        var hi = min(lo + SCH, n)
        comptime W = 8
        var accb = SIMD[U32, W](0)
        var acch = SIMD[U32, W](0)
        var i = lo
        while i + W <= hi:
            var v = wp.unsafe_load[width=W](i)
            var bits = v.to_bits[U32]()
            accb |= bits & 0xFFFF
            acch |= bits ^ v.cast[DType.float16]().cast[F32]().to_bits[U32]()
            i += W
        var sb: UInt32 = 0
        var sh: UInt32 = 0
        for l in range(W):
            sb |= accb[l]
            sh |= acch[l]
        while i < hi:
            var b1 = wp[unsafe_offset=i].to_bits[U32]()
            sb |= b1 & 0xFFFF
            sh |= b1 ^ wp[unsafe_offset=i].cast[DType.float16]().cast[F32]().to_bits[U32]()
            i += 1
        if sb != 0:
            bf_bad[sc] = 1
        if sh != 0:
            f16_bad[sc] = 1

    parallelize[scan](nsc)
    var all_bf = True
    var all_f16 = True
    for sc in range(nsc):
        if bf_bad[sc] != 0:
            all_bf = False
        if f16_bad[sc] != 0:
            all_f16 = False
    return (all_bf, all_f16)


def gemm_tiled_bf16(
    a: Pointer[Scalar[F32], ImmutAnyOrigin],
    b: Pointer[Scalar[U16], ImmutAnyOrigin],
    c: Pointer[Scalar[F32], MutAnyOrigin],
    m_dev: Int32, n_dev: Int32, kdim_dev: Int32,
):
    """gemm_tiled with B stored as bf16 bits, expanded u16 << 16 -> f32 on
    the shared-memory fill. The expansion is bit-exact for bf16-loaded
    weights, so the accumulation is bit-identical to the f32 kernel on the
    same values (benchmarks/microbench_gpu_gemm.mojo measured it FLAT)."""
    var m = Int(m_dev)
    var n = Int(n_dev)
    var kdim = Int(kdim_dev)

    var As = stack_allocation[
        BM * BK, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[F32, TN](0)
    var acc1 = SIMD[F32, TN](0)
    var acc2 = SIMD[F32, TN](0)
    var acc3 = SIMD[F32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.unsafe_store(ia, a.unsafe_load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.unsafe_load[width=4](b_src).cast[U32]() << 16
        Bs.unsafe_store(ib, SIMD[F32, 4](from_bits=raw))
        barrier()
        for kk in range(BK):
            var bv = Bs.unsafe_load[width=TN](kk * BN + tx * TN)
            acc0 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 3) * BK + kk]) * bv
        barrier()
    c.unsafe_store((row0 + 0) * n + col0, acc0)
    c.unsafe_store((row0 + 1) * n + col0, acc1)
    c.unsafe_store((row0 + 2) * n + col0, acc2)
    c.unsafe_store((row0 + 3) * n + col0, acc3)


def gemm_tiled_f16(
    a: Pointer[Scalar[F32], ImmutAnyOrigin],
    b: Pointer[Scalar[U16], ImmutAnyOrigin],
    c: Pointer[Scalar[F32], MutAnyOrigin],
    m_dev: Int32, n_dev: Int32, kdim_dev: Int32,
):
    """gemm_tiled with B stored as f16 bits (hardware cast -> f32 on the
    shared-memory fill) — for weights that round-trip f32 -> f16 exactly,
    i.e. the fp16 checkpoints. Bit-identical to f32 storage."""
    var m = Int(m_dev)
    var n = Int(n_dev)
    var kdim = Int(kdim_dev)

    var As = stack_allocation[
        BM * BK, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[F32], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[F32, TN](0)
    var acc1 = SIMD[F32, TN](0)
    var acc2 = SIMD[F32, TN](0)
    var acc3 = SIMD[F32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.unsafe_store(ia, a.unsafe_load[width=4](a_src))
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.unsafe_load[width=4](b_src)
        Bs.unsafe_store(ib, SIMD[DType.float16, 4](from_bits=raw).cast[F32]())
        barrier()
        for kk in range(BK):
            var bv = Bs.unsafe_load[width=TN](kk * BN + tx * TN)
            acc0 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 0) * BK + kk]) * bv
            acc1 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 1) * BK + kk]) * bv
            acc2 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 2) * BK + kk]) * bv
            acc3 += SIMD[F32, TN](As[unsafe_offset=(ty * TM + 3) * BK + kk]) * bv
        barrier()
    c.unsafe_store((row0 + 0) * n + col0, acc0)
    c.unsafe_store((row0 + 1) * n + col0, acc1)
    c.unsafe_store((row0 + 2) * n + col0, acc2)
    c.unsafe_store((row0 + 3) * n + col0, acc3)


def gemm_tiled_f16sh(
    a: Pointer[Scalar[F32], ImmutAnyOrigin],
    b: Pointer[Scalar[U16], ImmutAnyOrigin],
    c: Pointer[Scalar[F32], MutAnyOrigin],
    m_dev: Int32, n_dev: Int32, kdim_dev: Int32,
):
    """WP19: B lagret som f16-bits, BEGGE shared-flisene i f16 (A castes
    på fyllet, tilbake til f32 i registerlasten — mattematikken og
    akkumulatorene er f32). +32-40 % målt: kjernen var shared-
    båndbredde-bundet, og halverte fliser dobler effektiv båndbredde."""
    var m = Int(m_dev)
    var n = Int(n_dev)
    var kdim = Int(kdim_dev)

    var As = stack_allocation[
        BM * BK, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        BK * BN, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * 16 + tx
    var row0 = Int(block_idx.y) * BM + ty * TM
    var col0 = Int(block_idx.x) * BN + tx * TN
    var acc0 = SIMD[F32, TN](0)
    var acc1 = SIMD[F32, TN](0)
    var acc2 = SIMD[F32, TN](0)
    var acc3 = SIMD[F32, TN](0)
    var n_kb = kdim // BK
    for kb in range(n_kb):
        var ia = tid * 4
        var ar = ia // BK
        var ac = ia % BK
        var a_src = (Int(block_idx.y) * BM + ar) * kdim + kb * BK + ac
        As.unsafe_store(ia, a.unsafe_load[width=4](a_src).cast[DType.float16]())
        var ib = tid * 4
        var br = ib // BN
        var bc = ib % BN
        var b_src = (kb * BK + br) * n + Int(block_idx.x) * BN + bc
        var raw = b.unsafe_load[width=4](b_src)
        Bs.unsafe_store(ib, SIMD[DType.float16, 4](from_bits=raw))
        barrier()
        for kk in range(BK):
            var bv = Bs.unsafe_load[width=TN](kk * BN + tx * TN).cast[F32]()
            acc0 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 0) * BK + kk].cast[F32]()
            ) * bv
            acc1 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 1) * BK + kk].cast[F32]()
            ) * bv
            acc2 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 2) * BK + kk].cast[F32]()
            ) * bv
            acc3 += SIMD[F32, TN](
                As[unsafe_offset=(ty * TM + 3) * BK + kk].cast[F32]()
            ) * bv
        barrier()
    c.unsafe_store((row0 + 0) * n + col0, acc0)
    c.unsafe_store((row0 + 1) * n + col0, acc1)
    c.unsafe_store((row0 + 2) * n + col0, acc2)
    c.unsafe_store((row0 + 3) * n + col0, acc3)


struct GpuLinear(Copyable, Movable):
    """Per-SparseLinear device weights: W^T packed [ci, co] on the device
    (f32, or 16-bit bits when the weights are bit-exactly representable —
    WP11 step 14), a host copy of the bias for the fused readback add, and
    a device copy for the chained-mlp path (where the bias must land
    BEFORE the gelu, on the GPU)."""

    var g: GpuContext
    var wfmt: Int
    var bt: Optional[DeviceBuffer[F32]]  # W^T when wfmt == WFMT_F32
    var bt16: Optional[DeviceBuffer[U16]]  # W^T bf16/f16 bits otherwise
    var bias_host: List[Float32]
    var bias_dev: DeviceBuffer[F32]
    var has_bias: Bool
    var co: Int
    var ci: Int

    def __init__(
        out self,
        var g: GpuContext,
        wfmt: Int,
        var bt: Optional[DeviceBuffer[F32]],
        var bt16: Optional[DeviceBuffer[U16]],
        var bias_host: List[Float32],
        var bias_dev: DeviceBuffer[F32],
        has_bias: Bool,
        co: Int,
        ci: Int,
    ):
        self.g = g^
        self.wfmt = wfmt
        self.bt = bt^
        self.bt16 = bt16^
        self.bias_host = bias_host^
        self.bias_dev = bias_dev^
        self.has_bias = has_bias
        self.co = co
        self.ci = ci

    @staticmethod
    def try_build(
        gpu: Optional[GpuContext],
        weight: Tensor[F32],
        bias: Tensor[F32],
        has_bias: Bool,
        allow_16bit: Bool = True,
    ) raises -> Optional[GpuLinear]:
        """Upload W^T once at model load; None when the GPU is off or the
        kernel's shape constraints don't hold. Picks 16-bit storage when
        the weights allow it bit-exactly (see the WFMT_* note above);
        allow_16bit=False forces f32 storage (tests/debugging)."""
        if not gpu:
            return None
        var co = weight.shape[0]
        var ci = weight.shape[1]
        if co % BN != 0 or ci % BK != 0 or co * ci < GPU_MIN_WEIGHT:
            return None
        var g = gpu.value().copy()
        var n = ci * co
        var wp = weight.data.unsafe_ptr()

        # classify the storage format (wfmt_scan above): 16-bit only when
        # EVERY element expands back to the identical f32 bit pattern
        var wfmt = WFMT_F32
        if allow_16bit:
            var q = wfmt_scan(weight)
            if g.f16 and (q[0] or q[1]):
                # WP19-flagget: f16-shared-kjernen for alle 16-bits-
                # kvalifiserte vekter (bf16-vekter mister kun
                # subnormaler < 6e-8 i f16-castet — absoluttfeil)
                wfmt = WFMT_F16SH
            elif q[0]:
                wfmt = WFMT_BF16
            elif q[1]:
                wfmt = WFMT_F16

        # transpose pack, parallel over k-chunks (disjoint WC writes; this
        # runs once per weight at model load — ~170 weights per DiT make
        # the serial version a visible slice of load time)
        comptime KCH = 64
        var nch = (ci + KCH - 1) // KCH
        var bt: Optional[DeviceBuffer[F32]] = None
        var bt16: Optional[DeviceBuffer[U16]] = None
        if wfmt == WFMT_F32:
            bt = g.ctx.enqueue_create_buffer[F32](n)
            with bt.value().map_to_host() as h:
                var hp = h.unsafe_ptr()

                @parameter
                def pack(kc: Int):
                    var k1 = min((kc + 1) * KCH, ci)
                    for k in range(kc * KCH, k1):
                        for j in range(co):
                            hp[unsafe_offset=k * co + j] = wp[unsafe_offset=j * ci + k]

                parallelize[pack](nch)
        else:
            var is_bf = wfmt == WFMT_BF16
            bt16 = g.ctx.enqueue_create_buffer[U16](n)
            with bt16.value().map_to_host() as h:
                var hp = h.unsafe_ptr()

                @parameter
                def pack16(kc: Int):
                    var k1 = min((kc + 1) * KCH, ci)
                    for k in range(kc * KCH, k1):
                        for j in range(co):
                            var v = wp[unsafe_offset=j * ci + k]
                            if is_bf:
                                hp[unsafe_offset=k * co + j] = (v.to_bits[U32]() >> 16).cast[U16]()
                            else:
                                hp[unsafe_offset=k * co + j] = v.cast[DType.float16]().to_bits[U16]()

                parallelize[pack16](nch)
        var bias_host = List[Float32]()
        if has_bias:
            bias_host = List[Float32](length=co, fill=0)
            unsafe_memcpy(dest=bias_host.unsafe_ptr(), src=bias.data.unsafe_ptr(), count=co)
        var bias_dev = g.ctx.enqueue_create_buffer[F32](co)
        with bias_dev.map_to_host() as h:
            var bp = h.unsafe_ptr()
            if has_bias:
                unsafe_memcpy(dest=bp, src=bias.data.unsafe_ptr(), count=co)
            else:
                for j in range(co):
                    bp[unsafe_offset=j] = 0
        return GpuLinear(
            g^, wfmt, bt^, bt16^, bias_host^, bias_dev^, has_bias, co, ci
        )

    def wants(self, rows: Int) -> Bool:
        return rows >= GPU_MIN_ROWS and rows * self.co * self.ci >= GPU_MIN_PROXY

    def enqueue_gemm(
        self,
        a: Pointer[Scalar[F32], _],
        c: Pointer[Scalar[F32], _],
        m_pad: Int,
    ) raises:
        """Enqueue weight GEMM c[unsafe_offset=m_pad, co] = a[unsafe_offset=m_pad, ci] @ W^T."""
        # Nightly: DeviceBuffer pointers carry concrete origins; rebind for kernels.
        var a_k = rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](a)
        var c_k = rebind[Pointer[Scalar[F32], MutAnyOrigin]](c)
        if self.wfmt == WFMT_F16SH:
            var b_k_f16sh = rebind[Pointer[Scalar[U16], ImmutAnyOrigin]](self.bt16.value().unsafe_ptr())
            self.g.ctx.enqueue_function[gemm_tiled_f16sh](
                a_k,
                b_k_f16sh,
                c_k,
                Int32(m_pad),
                Int32(self.co),
                Int32(self.ci),
                grid_dim=(self.co // BN, m_pad // BM),
                block_dim=(16, 16),
            )
        elif self.wfmt == WFMT_BF16:
            var b_k_bf16 = rebind[Pointer[Scalar[U16], ImmutAnyOrigin]](self.bt16.value().unsafe_ptr())
            self.g.ctx.enqueue_function[gemm_tiled_bf16](
                a_k,
                b_k_bf16,
                c_k,
                Int32(m_pad),
                Int32(self.co),
                Int32(self.ci),
                grid_dim=(self.co // BN, m_pad // BM),
                block_dim=(16, 16),
            )
        elif self.wfmt == WFMT_F16:
            var b_k_f16 = rebind[Pointer[Scalar[U16], ImmutAnyOrigin]](self.bt16.value().unsafe_ptr())
            self.g.ctx.enqueue_function[gemm_tiled_f16](
                a_k,
                b_k_f16,
                c_k,
                Int32(m_pad),
                Int32(self.co),
                Int32(self.ci),
                grid_dim=(self.co // BN, m_pad // BM),
                block_dim=(16, 16),
            )
        else:
            var b_k = rebind[Pointer[Scalar[F32], ImmutAnyOrigin]](self.bt.value().unsafe_ptr())
            self.g.ctx.enqueue_function[gemm_tiled](
                a_k,
                b_k,
                c_k,
                Int32(m_pad),
                Int32(self.co),
                Int32(self.ci),
                grid_dim=(self.co // BN, m_pad // BM),
                block_dim=(16, 16),
            )

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        """y = x @ W^T + bias over the last dim, GEMM on the GPU, bias in
        the readback pass. Same contract as modules/nn.linear."""
        var ci = x.shape[len(x.shape) - 1]
        if ci != self.ci:
            raise Error("GpuLinear: in_features mismatch")
        var rows = x.numel() // ci
        var m_pad = ((rows + BM - 1) // BM) * BM
        var out_shape = x.shape.copy()
        out_shape[len(out_shape) - 1] = self.co
        var out = Tensor[F32](out_shape)

        var s = self.g.scratch
        if s[].a_cap < m_pad * ci:
            s[].a = self.g.ctx.enqueue_create_buffer[F32](m_pad * ci)
            s[].a_cap = m_pad * ci
        if s[].c_cap < m_pad * self.co:
            s[].c = self.g.ctx.enqueue_create_buffer[F32](m_pad * self.co)
            s[].c_cap = m_pad * self.co

        # upload: map enter is free for A (no device-side writes), chunked
        # parallel memcpy; pad rows are computed but never read back
        comptime NCHUNK = 16
        with s[].a.value().map_to_host() as h:
            var ap = h.unsafe_ptr()
            var xp = x.data.unsafe_ptr()
            var total = rows * ci
            var chunk = (total + NCHUNK - 1) // NCHUNK

            @parameter
            def copy_in(i: Int):
                var lo = i * chunk
                var n = min(chunk, total - lo)
                if n > 0:
                    unsafe_memcpy(dest=ap.unsafe_offset(lo), src=xp.unsafe_offset(lo), count=n)

            parallelize[copy_in](NCHUNK)

        self.enqueue_gemm(
            s[].a.value().unsafe_ptr(), s[].c.value().unsafe_ptr(), m_pad
        )
        # fence commits + waits (blit + kernel); the readback is a chunked
        # PARALLEL fused bias+copy straight off the mapped pointer (WC
        # memory: serial reads 2.2 GB/s, 8+ parallel chunks 9.2 GB/s).
        # Disjoint row ranges -> bit-identical to a serial pass.
        self.g.barrier()

        var op = out.data.unsafe_ptr()
        var co = self.co
        var has_bias = self.has_bias
        var bp = self.bias_host.unsafe_ptr()
        with s[].c.value().map_to_host() as h:
            var hp = h.unsafe_ptr()
            var rchunk = (rows + NCHUNK - 1) // NCHUNK

            @parameter
            def copy_out(i: Int):
                var r0 = i * rchunk
                var r1 = min(r0 + rchunk, rows)
                if r1 <= r0:
                    return
                if has_bias:
                    comptime W = 8
                    var co_main = (co // W) * W
                    for r in range(r0, r1):
                        var base = r * co
                        var j = 0
                        while j < co_main:
                            op.unsafe_store(base + j, hp.unsafe_load[width=W](base + j) + bp.unsafe_load[width=W](j))
                            j += W
                        while j < co:
                            op[unsafe_offset=base + j] = hp[unsafe_offset=base + j] + bp[unsafe_offset=j]
                            j += 1
                else:
                    unsafe_memcpy(dest=op.unsafe_offset(r0 * co), src=hp.unsafe_offset(r0 * co), count=(r1 - r0) * co)

            parallelize[copy_out](NCHUNK)
        return out^


def gelu_tanh_bias(
    c: Pointer[Scalar[F32], MutAnyOrigin],
    bias: Pointer[Scalar[F32], ImmutAnyOrigin],
    co_dev: Int32, total_dev: Int32,
):
    """In place on c [rows, co]: gelu-tanh(c + bias[unsafe_offset=col]). tanh is computed
    through exp — the GPU library tanh is a fast approximation (~2e-3)
    while exp is precise (softmax parity 2.3e-7)."""
    var co = Int(co_dev)
    var total = Int(total_dev)

    from std.gpu import thread_idx, block_idx, block_dim
    from std.math import exp
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < total:
        var v = c[unsafe_offset=i] + bias[unsafe_offset=i % co]
        var inner = Float32(0.7978845608028654) * (v + Float32(0.044715) * v * v * v)
        var ax = abs(inner)
        var t = exp(-2.0 * ax)
        var th = (1.0 - t) / (1.0 + t)
        if inner < 0:
            th = -th
        c[unsafe_offset=i] = 0.5 * v * (1.0 + th)


def gpu_mlp_wants(lin0: GpuLinear, lin2: GpuLinear, rows: Int) -> Bool:
    return lin0.wants(rows) and lin2.wants(rows) and lin0.co == lin2.ci


def gpu_mlp_forward(
    x: Tensor[F32], lin0: GpuLinear, lin2: GpuLinear
) raises -> Tensor[F32]:
    """Chained lin2(gelu_tanh(lin0(x))) with the [rows, lin0.co]
    intermediate device-resident (WP11 step 5) — for the DiT mlp that
    round-trip is 134 MB/block through WC memory. lin0's bias lands on
    the GPU before the gelu (bias_dev); lin2's bias uses the normal fused
    CPU readback. The A scratch doubles as the final output buffer (free
    after the first GEMM; sized to max of both roles). Pad rows may go
    NaN through the gelu on garbage — they stay in pad rows (every stage
    is row-independent) and are never read back."""
    var ci = x.shape[len(x.shape) - 1]
    if ci != lin0.ci:
        raise Error("gpu_mlp_forward: in_features mismatch")
    if lin0.co != lin2.ci:
        raise Error("gpu_mlp_forward: hidden width mismatch")
    var rows = x.numel() // ci
    var m_pad = ((rows + BM - 1) // BM) * BM
    var out_shape = x.shape.copy()
    out_shape[len(out_shape) - 1] = lin2.co
    var out = Tensor[F32](out_shape)

    var g = lin0.g.copy()
    var s = g.scratch
    var a_need = m_pad * max(ci, lin2.co)
    if s[].a_cap < a_need:
        s[].a = g.ctx.enqueue_create_buffer[F32](a_need)
        s[].a_cap = a_need
    if s[].c_cap < m_pad * lin0.co:
        s[].c = g.ctx.enqueue_create_buffer[F32](m_pad * lin0.co)
        s[].c_cap = m_pad * lin0.co

    comptime NCHUNK = 16
    with s[].a.value().map_to_host() as h:
        var ap = h.unsafe_ptr()
        var xp = x.data.unsafe_ptr()
        var total = rows * ci
        var chunk = (total + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_in(i: Int):
            var lo = i * chunk
            var n = min(chunk, total - lo)
            if n > 0:
                unsafe_memcpy(dest=ap.unsafe_offset(lo), src=xp.unsafe_offset(lo), count=n)

        parallelize[copy_in](NCHUNK)

    gpu_mlp_enqueue(g, s[].a.value(), rows, lin0, lin2, s[].a.value())
    g.barrier()

    # readback with lin2's bias fused (same pattern as GpuLinear.forward)
    var op = out.data.unsafe_ptr()
    var co2 = lin2.co
    var has_bias2 = lin2.has_bias
    var b2p = lin2.bias_host.unsafe_ptr()
    with s[].a.value().map_to_host() as h:
        var hp = h.unsafe_ptr()
        var rchunk = (rows + NCHUNK - 1) // NCHUNK

        @parameter
        def copy_out(i: Int):
            var r0 = i * rchunk
            var r1 = min(r0 + rchunk, rows)
            if r1 <= r0:
                return
            if has_bias2:
                comptime W = 8
                var co_main = (co2 // W) * W
                for r in range(r0, r1):
                    var base = r * co2
                    var j = 0
                    while j < co_main:
                        op.unsafe_store(base + j, hp.unsafe_load[width=W](base + j) + b2p.unsafe_load[width=W](j))
                        j += W
                    while j < co2:
                        op[unsafe_offset=base + j] = hp[unsafe_offset=base + j] + b2p[unsafe_offset=j]
                        j += 1
            else:
                unsafe_memcpy(dest=op.unsafe_offset(r0 * co2), src=hp.unsafe_offset(r0 * co2), count=(r1 - r0) * co2)

        parallelize[copy_out](NCHUNK)
    return out^


def gpu_mlp_enqueue(
    g: GpuContext,
    in_buf: DeviceBuffer[F32],
    rows: Int,
    lin0: GpuLinear,
    lin2: GpuLinear,
    out_buf: DeviceBuffer[F32],
) raises:
    """Enqueue-only mlp chain (WP11 step 10) from a device-resident input
    [m_pad, ci] to a device-resident output [m_pad, co] (in_buf == out_buf
    is fine — in_buf is free after the first GEMM). lin0's bias + gelu
    land on the GPU as before; lin2's bias is NOT added (host wrapper:
    readback; block path: gate_add). No transfers, no barrier."""
    var m_pad = ((rows + BM - 1) // BM) * BM
    var s = g.scratch
    if s[].c_cap < m_pad * lin0.co:
        s[].c = g.ctx.enqueue_create_buffer[F32](m_pad * lin0.co)
        s[].c_cap = m_pad * lin0.co
    lin0.enqueue_gemm(
        in_buf.unsafe_ptr(), s[].c.value().unsafe_ptr(), m_pad
    )
    var total_c = m_pad * lin0.co
    g.ctx.enqueue_function[gelu_tanh_bias](
        s[].c.value().unsafe_ptr(),
 lin0.bias_dev.unsafe_ptr(),

        Int32(lin0.co),
 Int32(total_c),

        grid_dim=((total_c + 255) // 256,), block_dim=(256,),
    )
    lin2.enqueue_gemm(
        s[].c.value().unsafe_ptr(), out_buf.unsafe_ptr(), m_pad
    )
