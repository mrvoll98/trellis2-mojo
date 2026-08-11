# Dense nn.Conv3d (cross-correlation) on [N, C, H, W, D] — needed by the
# sparse-structure VAE decoder (kernel 3 padding 1, and the 1x1 skip path).
# Cubic kernels and uniform stride/padding only, which is all the models use.
#
# Perf pass 7 (the WP10 recipe): stride-1 convs vectorize over the
# innermost output dim (each SIMD lane is one zd position; x taps are
# contiguous loads at s=1) for the interior zd range where every kd tap is
# in bounds, with the original scalar loop for edge positions, strided
# convs and small inputs. Per-element accumulation order (bias, then
# c/kh/kw/kd) is IDENTICAL in every path -> bit-identical results.
# (b, o) items write disjoint output regions and are parallelized above a
# flops-proxy threshold. This kernel is the SS-VAE decoder's entire cost:
# naive it ran at ~2 GF/s single-threaded and dominated the e2e ss stage.

from max.algorithm import parallelize

from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


struct Conv3d(Copyable, Movable):
    var weight: Tensor[F32]  # [Co, Ci, k, k, k]
    var bias: Tensor[F32]    # [Co]
    var stride: Int
    var padding: Int

    def __init__(out self, var weight: Tensor[F32], var bias: Tensor[F32], stride: Int = 1, padding: Int = 0):
        self.weight = weight^
        self.bias = bias^
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        if len(x.shape) != 5:
            raise Error("Conv3d: expected [N, C, H, W, D]")
        var n = x.shape[0]
        var ci = x.shape[1]
        var h = x.shape[2]
        var w = x.shape[3]
        var d = x.shape[4]
        if self.weight.shape[1] != ci:
            raise Error("Conv3d: in_channels mismatch")
        var co = self.weight.shape[0]
        var k = self.weight.shape[2]
        var s = self.stride
        var p = self.padding
        var oh = (h + 2 * p - k) // s + 1
        var ow = (w + 2 * p - k) // s + 1
        var od = (d + 2 * p - k) // s + 1
        var out_shape: List[Int] = [n, co, oh, ow, od]
        var out = Tensor[F32](out_shape)
        comptime W = 8
        var xp = x.data.unsafe_ptr()
        var wp = self.weight.data.unsafe_ptr()
        var bp = self.bias.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()

        # interior zd range for stride 1: every kd tap in bounds
        # (zd - p + kd in [0, d) for kd 0..k-1  <=>  zd in [p, d + p - k])
        var zd_lo = p
        var zd_hi = d + p - k + 1  # exclusive
        if zd_lo < 0:
            zd_lo = 0
        if zd_hi > od:
            zd_hi = od

        comptime OU = 4
        var n_ob = (co + OU - 1) // OU

        @parameter
        def item(bi: Int):
            var b = bi // n_ob
            var o0 = (bi % n_ob) * OU
            var ou = min(OU, co - o0)
            var o_stride = oh * ow * od
            for zh in range(oh):
                for zw in range(ow):
                    var o_base = (((b * co + o0) * oh + zh) * ow + zw) * od
                    # vectorized interior span (stride 1 only): SIMD lanes
                    # are zd positions, every kd tap in bounds. Full OU=4
                    # out-channel blocks share every x load (register
                    # blocking, pass-4 recipe); W8 chunks then a W4 chunk
                    # so small grids (16^3 interior = 14) stay vectorized.
                    var vec_start = 0
                    var vec_end = 0
                    if s == 1 and ou == OU and zd_hi - zd_lo >= 4:
                        vec_start = zd_lo
                        var zd = zd_lo
                        while zd + W <= zd_hi:
                            var a0 = SIMD[F32, W](bp[unsafe_offset=o0])
                            var a1 = SIMD[F32, W](bp[unsafe_offset=o0 + 1])
                            var a2 = SIMD[F32, W](bp[unsafe_offset=o0 + 2])
                            var a3 = SIMD[F32, W](bp[unsafe_offset=o0 + 3])
                            for c in range(ci):
                                for kh in range(k):
                                    var ih = zh - p + kh
                                    if ih < 0 or ih >= h:
                                        continue
                                    for kw in range(k):
                                        var iw = zw - p + kw
                                        if iw < 0 or iw >= w:
                                            continue
                                        var x_base = (((b * ci + c) * h + ih) * w + iw) * d + zd - p
                                        var w_base = (((o0 * ci + c) * k + kh) * k + kw) * k
                                        var w_step = ci * k * k * k
                                        for kd in range(k):
                                            var xv = xp.unsafe_load[width=W](x_base + kd)
                                            a0 += SIMD[F32, W](wp[unsafe_offset=w_base + kd]) * xv
                                            a1 += SIMD[F32, W](wp[unsafe_offset=w_base + w_step + kd]) * xv
                                            a2 += SIMD[F32, W](wp[unsafe_offset=w_base + 2 * w_step + kd]) * xv
                                            a3 += SIMD[F32, W](wp[unsafe_offset=w_base + 3 * w_step + kd]) * xv
                            op.unsafe_store(o_base + zd, a0)
                            op.unsafe_store(o_base + o_stride + zd, a1)
                            op.unsafe_store(o_base + 2 * o_stride + zd, a2)
                            op.unsafe_store(o_base + 3 * o_stride + zd, a3)
                            zd += W
                        if zd + 4 <= zd_hi:
                            var a0 = SIMD[F32, 4](bp[unsafe_offset=o0])
                            var a1 = SIMD[F32, 4](bp[unsafe_offset=o0 + 1])
                            var a2 = SIMD[F32, 4](bp[unsafe_offset=o0 + 2])
                            var a3 = SIMD[F32, 4](bp[unsafe_offset=o0 + 3])
                            for c in range(ci):
                                for kh in range(k):
                                    var ih = zh - p + kh
                                    if ih < 0 or ih >= h:
                                        continue
                                    for kw in range(k):
                                        var iw = zw - p + kw
                                        if iw < 0 or iw >= w:
                                            continue
                                        var x_base = (((b * ci + c) * h + ih) * w + iw) * d + zd - p
                                        var w_base = (((o0 * ci + c) * k + kh) * k + kw) * k
                                        var w_step = ci * k * k * k
                                        for kd in range(k):
                                            var xv = xp.unsafe_load[width=4](x_base + kd)
                                            a0 += SIMD[F32, 4](wp[unsafe_offset=w_base + kd]) * xv
                                            a1 += SIMD[F32, 4](wp[unsafe_offset=w_base + w_step + kd]) * xv
                                            a2 += SIMD[F32, 4](wp[unsafe_offset=w_base + 2 * w_step + kd]) * xv
                                            a3 += SIMD[F32, 4](wp[unsafe_offset=w_base + 3 * w_step + kd]) * xv
                            op.unsafe_store(o_base + zd, a0)
                            op.unsafe_store(o_base + o_stride + zd, a1)
                            op.unsafe_store(o_base + 2 * o_stride + zd, a2)
                            op.unsafe_store(o_base + 3 * o_stride + zd, a3)
                            zd += 4
                        vec_end = zd
                    # partial o-blocks (co % OU, incl. co < OU) still get a
                    # per-channel W8 interior rung; then the scalar pass:
                    # edges, interior tail and strided convs — the original
                    # loop body, same per-element accumulation order
                    for oo in range(o0, o0 + ou):
                        var oo_base = (((b * co + oo) * oh + zh) * ow + zw) * od
                        var vs = 0
                        var ve = 0
                        if s == 1 and ou != OU and zd_hi - zd_lo >= W:
                            vs = zd_lo
                            var zd = zd_lo
                            while zd + W <= zd_hi:
                                var acc = SIMD[F32, W](bp[unsafe_offset=oo])
                                for c in range(ci):
                                    for kh in range(k):
                                        var ih = zh - p + kh
                                        if ih < 0 or ih >= h:
                                            continue
                                        for kw in range(k):
                                            var iw = zw - p + kw
                                            if iw < 0 or iw >= w:
                                                continue
                                            var x_base = (((b * ci + c) * h + ih) * w + iw) * d + zd - p
                                            var w_base = (((oo * ci + c) * k + kh) * k + kw) * k
                                            for kd in range(k):
                                                acc += SIMD[F32, W](wp[unsafe_offset=w_base + kd]) * xp.unsafe_load[width=W](x_base + kd)
                                op.unsafe_store(oo_base + zd, acc)
                                zd += W
                            ve = zd
                        for zd in range(od):
                            if (zd >= vec_start and zd < vec_end) or (zd >= vs and zd < ve):
                                continue
                            var acc: Float32 = bp[unsafe_offset=oo]
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
                                                wp[unsafe_offset=(((oo * ci + c) * k + kh) * k + kw) * k + kd]
                                                * xp[unsafe_offset=(((b * ci + c) * h + ih) * w + iw) * d + idd]
                                            )
                            op[unsafe_offset=oo_base + zd] = acc

        var n_items = n * n_ob
        var flops = n * co * oh * ow * od * ci * k * k * k
        if flops < 1 << 21:
            for bi in range(n_items):
                item(bi)
        else:
            parallelize[item](n_items)
        return out^
