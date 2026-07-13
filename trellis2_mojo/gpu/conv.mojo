# WP11 step 6: submanifold sparse conv on the Metal GPU. The decode stage
# is conv-dominated (~35 s of the 43 s smoke decode: 23 s in the ConvNeXt
# convs + most of the 17 s upsample blocks), and the CPU kernel is
# gather-bound at ~150-215 GF/s.
#
# Formulation: the CPU edge lists (src, tgt, kidx from SparseConv3d's
# cached neighbor map) are counting-sorted by target into CSR on the host
# — since WP11 step 11 the sorted CSR is spatial-CACHED per coords/kernel
# by SparseConv3d.forward (the sort only depends on the edges; every conv
# on the same coords used to re-sort per call) — then ONE kernel computes
# each output row by walking its edge range:
# out[t, co0:co0+4] = sum_e x[src_e] . W^T[kidx_e][:, co0:co0+4]. Threads
# tile (row, co-lane-4); threads sharing a row broadcast the same x
# scalars and adjacent co-lanes read coalesced weight lines. The weight is
# uploaded ONCE per model load as [K, Ci, Co] (GpuSparseConv.try_build,
# same pattern as GpuLinear); x/edges upload and out readback per call.
#
# Argument marshalling (see gpu/linear.mojo laws): the kernel needs
# x, w, edges, out = 4 pointers, so it can take NO scalars — every
# dimension rides in the edges buffer header:
#   int32 [0..3] = n, ci, co, e_total
#   [4 .. 4+n]           row_start (n+1, CSR by target)
#   [5+n .. 5+n+e)       src
#   [5+n+e .. 5+n+2e)    kidx
#
# Numerics: within a row, edges keep the CPU's (kernel-offset-major)
# order (stable counting sort), but the GPU accumulates scalar-x-vec4
# per channel where the CPU does a SIMD-tree dot per edge -> tolerance
# parity, never bit-equality (established GPU precedent). The bias is
# added on the CPU in the parallel readback.
#
# WP11 step 15: the weight is stored as f16 bits (hardware cast on each
# line load) when every weight is exactly f16-representable — true for
# the fp16 decoder checkpoints, i.e. every sparse conv in the pipeline.
# The expansion is bit-exact, so the f16 kernel is bit-identical to the
# f32 kernel on the same values (step 14 precedent in gpu/linear.mojo).

from std.algorithm import parallelize
from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceBuffer
from std.memory import memcpy

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.linear import WFMT_F16, WFMT_F32, wfmt_scan
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32
comptime F16 = DType.float16
comptime I32 = DType.int32
comptime U16 = DType.uint16

# below this edge-flops proxy (E * ci * co) the transfer overhead beats
# the GPU win over the ~200 GF/s CPU gather kernel
comptime GPU_CONV_MIN_PROXY = 1 << 31


def sparse_conv_gather(
    x: UnsafePointer[Scalar[F32], MutAnyOrigin],
    w: UnsafePointer[Scalar[F32], MutAnyOrigin],
    edges: UnsafePointer[Scalar[I32], MutAnyOrigin],
    dst: UnsafePointer[Scalar[F32], MutAnyOrigin],
):
    """WP11 step 9: TWO target rows x 8 co-lanes per thread. The rows'
    edge lists are merge-walked on kidx — ascending within a row because
    the kidx-major build order survives the stable counting sort — so
    kernel offsets present in BOTH rows load each weight line ONCE for
    two rows' accumulation (the [ci, co] weight-plane slices are the
    dominant traffic; most decode rows share most of the 27 offsets).
    Per-row edge order and per-accumulator math are unchanged ->
    bit-identical to the single-row kernel. Dims live in the edges
    header (4 pointers leave no scalar binding); co % 64 == 0."""
    var n = Int(edges[0])
    var ci = Int(edges[1])
    var co = Int(edges[2])
    var et = Int(edges[3])
    var row = (Int(block_idx.y) * 16 + Int(thread_idx.y)) * 2
    var co0 = (Int(block_idx.x) * 8 + Int(thread_idx.x)) * 8
    if row >= n:
        return
    var sbase = 5 + n
    var kbase = 5 + n + et
    var a0 = Int(edges[4 + row])
    var a1 = Int(edges[4 + row + 1])
    var b1 = a1
    if row + 1 < n:
        b1 = Int(edges[4 + row + 2])
    var acc_a0 = SIMD[F32, 4](0)
    var acc_a1 = SIMD[F32, 4](0)
    var acc_b0 = SIMD[F32, 4](0)
    var acc_b1 = SIMD[F32, 4](0)
    var ea = a0
    var eb = a1
    comptime K_END = 1 << 30
    while ea < a1 or eb < b1:
        var ka = K_END
        if ea < a1:
            ka = Int(edges[kbase + ea])
        var kb = K_END
        if eb < b1:
            kb = Int(edges[kbase + eb])
        if ka == kb:
            var xa = Int(edges[sbase + ea]) * ci
            var xb = Int(edges[sbase + eb]) * ci
            var wb = ka * ci * co + co0
            for c in range(ci):
                var wrow = wb + c * co
                var wv0 = w.load[width=4](wrow)
                var wv1 = w.load[width=4](wrow + 4)
                var xsa = SIMD[F32, 4](x[xa + c])
                var xsb = SIMD[F32, 4](x[xb + c])
                acc_a0 += xsa * wv0
                acc_a1 += xsa * wv1
                acc_b0 += xsb * wv0
                acc_b1 += xsb * wv1
            ea += 1
            eb += 1
        elif ka < kb:
            var xa = Int(edges[sbase + ea]) * ci
            var wb = ka * ci * co + co0
            for c in range(ci):
                var wrow = wb + c * co
                var xsa = SIMD[F32, 4](x[xa + c])
                acc_a0 += xsa * w.load[width=4](wrow)
                acc_a1 += xsa * w.load[width=4](wrow + 4)
            ea += 1
        else:
            var xb = Int(edges[sbase + eb]) * ci
            var wb = kb * ci * co + co0
            for c in range(ci):
                var wrow = wb + c * co
                var xsb = SIMD[F32, 4](x[xb + c])
                acc_b0 += xsb * w.load[width=4](wrow)
                acc_b1 += xsb * w.load[width=4](wrow + 4)
            eb += 1
    dst.store(row * co + co0, acc_a0)
    dst.store(row * co + co0 + 4, acc_a1)
    if row + 1 < n:
        dst.store((row + 1) * co + co0, acc_b0)
        dst.store((row + 1) * co + co0 + 4, acc_b1)


def sparse_conv_gather_f16(
    x: UnsafePointer[Scalar[F32], MutAnyOrigin],
    w: UnsafePointer[Scalar[U16], MutAnyOrigin],
    edges: UnsafePointer[Scalar[I32], MutAnyOrigin],
    dst: UnsafePointer[Scalar[F32], MutAnyOrigin],
):
    """sparse_conv_gather with the weight stored as f16 bits (hardware
    cast -> f32 on each line load) — WP11 step 15. Only used when every
    weight round-trips f32 -> f16 exactly (the fp16 decoder checkpoints),
    so the accumulation sees identical values and the result is
    bit-identical to the f32-stored kernel. bf16 is deliberately NOT
    supported here: no sparse-conv checkpoint is bf16 (the DiTs have no
    sparse conv), and the 4-pointer marshalling limit leaves no scalar
    binding for a format flag — two kernels, host-side dispatch."""
    var n = Int(edges[0])
    var ci = Int(edges[1])
    var co = Int(edges[2])
    var et = Int(edges[3])
    var row = (Int(block_idx.y) * 16 + Int(thread_idx.y)) * 2
    var co0 = (Int(block_idx.x) * 8 + Int(thread_idx.x)) * 8
    if row >= n:
        return
    var sbase = 5 + n
    var kbase = 5 + n + et
    var a0 = Int(edges[4 + row])
    var a1 = Int(edges[4 + row + 1])
    var b1 = a1
    if row + 1 < n:
        b1 = Int(edges[4 + row + 2])
    var acc_a0 = SIMD[F32, 4](0)
    var acc_a1 = SIMD[F32, 4](0)
    var acc_b0 = SIMD[F32, 4](0)
    var acc_b1 = SIMD[F32, 4](0)
    var ea = a0
    var eb = a1
    comptime K_END = 1 << 30
    while ea < a1 or eb < b1:
        var ka = K_END
        if ea < a1:
            ka = Int(edges[kbase + ea])
        var kb = K_END
        if eb < b1:
            kb = Int(edges[kbase + eb])
        if ka == kb:
            var xa = Int(edges[sbase + ea]) * ci
            var xb = Int(edges[sbase + eb]) * ci
            var wb = ka * ci * co + co0
            for c in range(ci):
                var wrow = wb + c * co
                var wv0 = SIMD[F16, 4](from_bits=w.load[width=4](wrow)).cast[F32]()
                var wv1 = SIMD[F16, 4](from_bits=w.load[width=4](wrow + 4)).cast[F32]()
                var xsa = SIMD[F32, 4](x[xa + c])
                var xsb = SIMD[F32, 4](x[xb + c])
                acc_a0 += xsa * wv0
                acc_a1 += xsa * wv1
                acc_b0 += xsb * wv0
                acc_b1 += xsb * wv1
            ea += 1
            eb += 1
        elif ka < kb:
            var xa = Int(edges[sbase + ea]) * ci
            var wb = ka * ci * co + co0
            for c in range(ci):
                var wrow = wb + c * co
                var xsa = SIMD[F32, 4](x[xa + c])
                acc_a0 += xsa * SIMD[F16, 4](from_bits=w.load[width=4](wrow)).cast[F32]()
                acc_a1 += xsa * SIMD[F16, 4](from_bits=w.load[width=4](wrow + 4)).cast[F32]()
            ea += 1
        else:
            var xb = Int(edges[sbase + eb]) * ci
            var wb = kb * ci * co + co0
            for c in range(ci):
                var wrow = wb + c * co
                var xsb = SIMD[F32, 4](x[xb + c])
                acc_b0 += xsb * SIMD[F16, 4](from_bits=w.load[width=4](wrow)).cast[F32]()
                acc_b1 += xsb * SIMD[F16, 4](from_bits=w.load[width=4](wrow + 4)).cast[F32]()
            eb += 1
    dst.store(row * co + co0, acc_a0)
    dst.store(row * co + co0 + 4, acc_a1)
    if row + 1 < n:
        dst.store((row + 1) * co + co0, acc_b0)
        dst.store((row + 1) * co + co0 + 4, acc_b1)


struct GpuSparseConv(Copyable, Movable):
    """Per-SparseConv3d device weights: [K, Ci, Co] (transposed from the
    conv_none [Co, Kd, Kh, Kw, Ci] layout — f32, or f16 bits when every
    weight is exactly f16-representable, WP11 step 15) plus the host bias
    for the readback add."""

    var g: GpuContext
    var wfmt: Int
    var wt: Optional[DeviceBuffer[F32]]  # [K, Ci, Co] when wfmt == WFMT_F32
    var wt16: Optional[DeviceBuffer[U16]]  # f16 bits otherwise
    var bias_host: List[Float32]
    var has_bias: Bool
    var co: Int
    var ci: Int
    var ksize: Int

    def __init__(
        out self,
        var g: GpuContext,
        wfmt: Int,
        var wt: Optional[DeviceBuffer[F32]],
        var wt16: Optional[DeviceBuffer[U16]],
        var bias_host: List[Float32],
        has_bias: Bool,
        co: Int,
        ci: Int,
        ksize: Int,
    ):
        self.g = g^
        self.wfmt = wfmt
        self.wt = wt^
        self.wt16 = wt16^
        self.bias_host = bias_host^
        self.has_bias = has_bias
        self.co = co
        self.ci = ci
        self.ksize = ksize

    @staticmethod
    def try_build(
        gpu: Optional[GpuContext],
        weight: Tensor[F32],
        bias: Tensor[F32],
        has_bias: Bool,
        allow_16bit: Bool = True,
    ) raises -> Optional[GpuSparseConv]:
        """Upload W as [K, Ci, Co] once at model load; None when the GPU is
        off or co doesn't tile. Stores f16 bits when every weight is
        exactly f16-representable (the fp16 decoder checkpoints — the only
        source of sparse-conv weights; bf16 deliberately unsupported, see
        the f16 kernel's docstring); allow_16bit=False forces f32."""
        if not gpu:
            return None
        var co = weight.shape[0]
        var ksize = weight.shape[1] * weight.shape[2] * weight.shape[3]
        var ci = weight.shape[4]
        if co % 64 != 0:
            return None
        var g = gpu.value().copy()
        var wp = weight.data.unsafe_ptr()
        var wfmt = WFMT_F32
        if allow_16bit:
            var q = wfmt_scan(weight)
            if q[1]:
                wfmt = WFMT_F16
        var wt: Optional[DeviceBuffer[F32]] = None
        var wt16: Optional[DeviceBuffer[U16]] = None
        if wfmt == WFMT_F32:
            wt = g.ctx.enqueue_create_buffer[F32](ksize * ci * co)
            with wt.value().map_to_host() as h:
                var hp = h.unsafe_ptr()

                @parameter
                def pack(k: Int):
                    for c in range(ci):
                        var dst = (k * ci + c) * co
                        for o in range(co):
                            hp[dst + o] = wp[(o * ksize + k) * ci + c]

                parallelize[pack](ksize)
        else:
            wt16 = g.ctx.enqueue_create_buffer[U16](ksize * ci * co)
            with wt16.value().map_to_host() as h:
                var hp = h.unsafe_ptr()

                @parameter
                def pack16(k: Int):
                    for c in range(ci):
                        var dst = (k * ci + c) * co
                        for o in range(co):
                            hp[dst + o] = (
                                wp[(o * ksize + k) * ci + c]
                                .cast[F16]()
                                .to_bits[U16]()
                            )

                parallelize[pack16](ksize)
        var bias_host = List[Float32]()
        if has_bias:
            bias_host = List[Float32](length=co, fill=0)
            memcpy(dest=bias_host.unsafe_ptr(), src=bias.data.unsafe_ptr(), count=co)
        return GpuSparseConv(
            g^, wfmt, wt^, wt16^, bias_host^, has_bias, co, ci, ksize
        )

    def wants(self, n_edges: Int) -> Bool:
        return n_edges * self.ci * self.co >= GPU_CONV_MIN_PROXY

    def forward(
        self,
        x: Tensor[F32],
        row_start: List[Int],
        src_s: List[Int],
        kidx_s: List[Int],
    ) raises -> Tensor[F32]:
        """x [N, Ci] + the CSR-SORTED edge lists (counting-sorted by
        target with kidx order preserved per row — built and spatial-cached
        by SparseConv3d.forward since WP11 step 11) -> [N, Co]. The int32
        pack fill is chunk-parallel; only the per-conv header differs
        between convs sharing coords."""
        var n = x.shape[0]
        var ci = x.shape[1]
        if ci != self.ci:
            raise Error("GpuSparseConv: in_features mismatch")
        if len(row_start) != n + 1:
            raise Error("GpuSparseConv: row_start length mismatch")
        var e_total = len(src_s)
        var out_shape: List[Int] = [n, self.co]
        var out = Tensor[F32](out_shape)

        var pack_len = 5 + n + 2 * e_total
        var epack = List[Int32](length=pack_len, fill=0)
        var ep = epack.unsafe_ptr()
        ep[0] = Int32(n)
        ep[1] = Int32(ci)
        ep[2] = Int32(self.co)
        ep[3] = Int32(e_total)
        var rp = row_start.unsafe_ptr()
        var sp = src_s.unsafe_ptr()
        var kp = kidx_s.unsafe_ptr()
        var sbase = 5 + n
        var kbase = 5 + n + e_total
        comptime FCH = 16
        var rchunk = (n + 1 + FCH - 1) // FCH
        var echunk = (e_total + FCH - 1) // FCH

        @parameter
        def fill_pack(i: Int):
            var r0 = i * rchunk
            var r1 = min(r0 + rchunk, n + 1)
            for r in range(r0, r1):
                ep[4 + r] = Int32(rp[r])
            var e0 = i * echunk
            var e1 = min(e0 + echunk, e_total)
            for e in range(e0, e1):
                ep[sbase + e] = Int32(sp[e])
                ep[kbase + e] = Int32(kp[e])

        parallelize[fill_pack](FCH)

        var g = self.g.copy()
        var s = g.scratch
        if s[].a_cap < n * ci:
            s[].a = g.ctx.enqueue_create_buffer[F32](n * ci)
            s[].a_cap = n * ci
        if s[].c_cap < n * self.co:
            s[].c = g.ctx.enqueue_create_buffer[F32](n * self.co)
            s[].c_cap = n * self.co
        if s[].e_cap < pack_len:
            s[].e = g.ctx.enqueue_create_buffer[I32](pack_len)
            s[].e_cap = pack_len

        comptime NCHUNK = 16
        with s[].a.value().map_to_host() as h:
            var ap = h.unsafe_ptr()
            var xp = x.data.unsafe_ptr()
            var total = n * ci
            var chunk = (total + NCHUNK - 1) // NCHUNK

            @parameter
            def copy_x(i: Int):
                var lo = i * chunk
                var cnt = min(chunk, total - lo)
                if cnt > 0:
                    memcpy(dest=ap + lo, src=xp + lo, count=cnt)

            parallelize[copy_x](NCHUNK)
        with s[].e.value().map_to_host() as h:
            var dp = h.unsafe_ptr()
            var chunk = (pack_len + NCHUNK - 1) // NCHUNK

            @parameter
            def copy_e(i: Int):
                var lo = i * chunk
                var cnt = min(chunk, pack_len - lo)
                if cnt > 0:
                    memcpy(dest=dp + lo, src=ep + lo, count=cnt)

            parallelize[copy_e](NCHUNK)

        if self.wfmt == WFMT_F16:
            g.ctx.enqueue_function[sparse_conv_gather_f16](
                s[].a.value().unsafe_ptr(), self.wt16.value().unsafe_ptr(),
                s[].e.value().unsafe_ptr(), s[].c.value().unsafe_ptr(),
                grid_dim=(self.co // 64, (n + 31) // 32), block_dim=(8, 16),
            )
        else:
            g.ctx.enqueue_function[sparse_conv_gather](
                s[].a.value().unsafe_ptr(), self.wt.value().unsafe_ptr(),
                s[].e.value().unsafe_ptr(), s[].c.value().unsafe_ptr(),
                grid_dim=(self.co // 64, (n + 31) // 32), block_dim=(8, 16),
            )
        g.barrier()

        # readback with the bias fused (chunked parallel — WC reads)
        var op = out.data.unsafe_ptr()
        var co = self.co
        var has_bias = self.has_bias
        var bp = self.bias_host.unsafe_ptr()
        with s[].c.value().map_to_host() as h:
            var hp = h.unsafe_ptr()
            var rchunk = (n + NCHUNK - 1) // NCHUNK

            @parameter
            def copy_out(i: Int):
                var r0 = i * rchunk
                var r1 = min(r0 + rchunk, n)
                if r1 <= r0:
                    return
                if has_bias:
                    comptime W = 8
                    var co_main = (co // W) * W
                    for r in range(r0, r1):
                        var base = r * co
                        var j = 0
                        while j < co_main:
                            op.store(base + j, hp.load[width=W](base + j) + bp.load[width=W](j))
                            j += W
                        while j < co:
                            op[base + j] = hp[base + j] + bp[j]
                            j += 1
                else:
                    memcpy(dest=op + r0 * co, src=hp + r0 * co, count=(r1 - r0) * co)

            parallelize[copy_out](NCHUNK)
        return out^
