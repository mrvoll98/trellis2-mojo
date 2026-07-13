# Mojo port of the dense building blocks the inference path actually uses:
#   modules/norm.py      -> LayerNorm32, GroupNorm32, ChannelLayerNorm32
#   modules/utils.py     -> modulate (zero/scale/convert_module are torch-only)
#   modules/sparse/linear.py       -> SparseLinear (+ dense linear core)
#   modules/sparse/nonlinearity.py -> relu/silu/gelu (+ Sparse wrappers)
#
# The "32" suffix means "compute in float32 under autocast" in the original;
# v1 runs everything in float32 so the cast is a no-op — the names are kept
# so the mapping to the source stays 1:1.
#
# modules/sparse/norm.py (SparseGroupNorm/SparseLayerNorm) is NOT ported:
# no inference model uses it, and SparseLayerNorm's permute/reshape is
# incompatible with nn.LayerNorm's shape contract (it would crash if called).

from std.algorithm import parallelize
from std.math import erf, exp, sqrt, tanh

from trellis2_mojo.gpu.linear import GpuLinear
from trellis2_mojo.sparse.tensor import Tensor, OP_ADD, OP_MUL
from trellis2_mojo.sparse.basic import VarLenTensor, SparseTensor

comptime F32 = DType.float32


# -- linear -------------------------------------------------------------------

def linear(x: Tensor[F32], weight: Tensor[F32], bias: Tensor[F32], has_bias: Bool = True) raises -> Tensor[F32]:
    """y = x @ weight.T + bias over the last dim (torch F.linear).
    x: [..., Ci], weight: [Co, Ci], bias: [Co]. SIMD dot over Ci (WP10) —
    the x row and the weight row are both contiguous over i. Register
    tiling (pass 4): RU rows x OU out features per block share the x/w
    loads across 8 independent accumulator chains; per-(row, out) math is
    identical to the single-pair path, so results stay bit-identical.
    Row blocks are parallelized (disjoint out regions -> bit-identical to
    the serial path, which small inputs take to skip thread-spawn
    overhead).

    Large inputs take a packed-GEMM path instead (pass 5): the weight is
    packed into [k][NR]-major panels and each x row block into [k][MR],
    so the MR x NR output tile stays in 8 SIMD registers through the
    whole k loop (outer-product formulation, 3 loads per 8 vector FMAs).
    Each output element is a plain sequential sum over k — a different
    (deterministic) accumulation order than the dot path, within parity
    tolerance of it; serial and parallel runs are identical either way.
    Row/column tails that don't fill a tile fall back to the dot path."""
    comptime W = 8
    comptime RU = 4
    comptime OU = 2
    comptime MR = 4
    comptime NR = 16
    var ci = x.shape[len(x.shape) - 1]
    if weight.shape[1] != ci:
        raise Error("linear: in_features mismatch")
    var co = weight.shape[0]
    var rows = x.numel() // ci
    var out_shape = x.shape.copy()
    out_shape[len(out_shape) - 1] = co
    var out = Tensor[F32](out_shape)
    var xp = x.data.unsafe_ptr()
    var wp = weight.data.unsafe_ptr()
    var bp = bias.data.unsafe_ptr()
    var op = out.data.unsafe_ptr()

    @parameter
    def row_block(rb: Int):
        var r0 = rb * RU
        var ru = min(RU, rows - r0)
        var o = 0
        while o < co:
            var ou = min(OU, co - o)
            if ru == 4 and ou == 2:
                var x_base0 = r0 * ci
                var x_base1 = x_base0 + ci
                var x_base2 = x_base1 + ci
                var x_base3 = x_base2 + ci
                var w_base0 = o * ci
                var w_base1 = w_base0 + ci
                var a00 = SIMD[F32, W](0)
                var a01 = SIMD[F32, W](0)
                var a10 = SIMD[F32, W](0)
                var a11 = SIMD[F32, W](0)
                var a20 = SIMD[F32, W](0)
                var a21 = SIMD[F32, W](0)
                var a30 = SIMD[F32, W](0)
                var a31 = SIMD[F32, W](0)
                var i = 0
                while i + W <= ci:
                    var wv0 = wp.load[width=W](w_base0 + i)
                    var wv1 = wp.load[width=W](w_base1 + i)
                    var xv0 = xp.load[width=W](x_base0 + i)
                    var xv1 = xp.load[width=W](x_base1 + i)
                    var xv2 = xp.load[width=W](x_base2 + i)
                    var xv3 = xp.load[width=W](x_base3 + i)
                    a00 += xv0 * wv0
                    a01 += xv0 * wv1
                    a10 += xv1 * wv0
                    a11 += xv1 * wv1
                    a20 += xv2 * wv0
                    a21 += xv2 * wv1
                    a30 += xv3 * wv0
                    a31 += xv3 * wv1
                    i += W
                var s00 = a00.reduce_add()
                var s01 = a01.reduce_add()
                var s10 = a10.reduce_add()
                var s11 = a11.reduce_add()
                var s20 = a20.reduce_add()
                var s21 = a21.reduce_add()
                var s30 = a30.reduce_add()
                var s31 = a31.reduce_add()
                while i < ci:
                    var w0i = wp[w_base0 + i]
                    var w1i = wp[w_base1 + i]
                    s00 += xp[x_base0 + i] * w0i
                    s01 += xp[x_base0 + i] * w1i
                    s10 += xp[x_base1 + i] * w0i
                    s11 += xp[x_base1 + i] * w1i
                    s20 += xp[x_base2 + i] * w0i
                    s21 += xp[x_base2 + i] * w1i
                    s30 += xp[x_base3 + i] * w0i
                    s31 += xp[x_base3 + i] * w1i
                    i += 1
                if has_bias:
                    var b0 = bp[o]
                    var b1 = bp[o + 1]
                    s00 += b0
                    s01 += b1
                    s10 += b0
                    s11 += b1
                    s20 += b0
                    s21 += b1
                    s30 += b0
                    s31 += b1
                var o_base = r0 * co + o
                op[o_base] = s00
                op[o_base + 1] = s01
                op[o_base + co] = s10
                op[o_base + co + 1] = s11
                op[o_base + 2 * co] = s20
                op[o_base + 2 * co + 1] = s21
                op[o_base + 3 * co] = s30
                op[o_base + 3 * co + 1] = s31
            else:
                for oo in range(o, o + ou):
                    var w_base = oo * ci
                    for rr in range(r0, r0 + ru):
                        var x_base = rr * ci
                        var accv = SIMD[F32, W](0)
                        var i = 0
                        while i + W <= ci:
                            accv += xp.load[width=W](x_base + i) * wp.load[width=W](w_base + i)
                            i += W
                        var acc = accv.reduce_add()
                        while i < ci:
                            acc += xp[x_base + i] * wp[w_base + i]
                            i += 1
                        if has_bias:
                            acc += bp[oo]
                        op[rr * co + oo] = acc
            o += ou

    # threshold tuned on the WP10 sampler case: many small linear calls per
    # forward make spawn/join overhead dominate below ~2M flops-proxy
    var n_blocks = (rows + RU - 1) // RU
    if rows * co * ci < 1 << 21:
        for rb in range(n_blocks):
            row_block(rb)
        return out^
    if rows < MR or co < NR:
        parallelize[row_block](n_blocks)
        return out^

    # packed-GEMM path: pack the weight once per call into [k][NR]-major
    # panels (the pack cost is one pass over the weight, amortized over
    # all row blocks)
    var n_panels = co // NR
    var co_main = n_panels * NR
    var bpack = List[Float32](length=n_panels * ci * NR, fill=0)
    var bpkp = bpack.unsafe_ptr()
    for p in range(n_panels):
        var n0 = p * NR
        var base = p * ci * NR
        for kk in range(ci):
            for j in range(NR):
                bpkp[base + kk * NR + j] = wp[(n0 + j) * ci + kk]

    var n_gblocks = rows // MR

    @parameter
    def gemm_block(rb: Int):
        var r0 = rb * MR
        # pack the x block as [k][MR] so one width-MR load feeds all rows
        var apack = List[Float32](length=ci * MR, fill=0)
        var app = apack.unsafe_ptr()
        for m in range(MR):
            var xb = (r0 + m) * ci
            for kk in range(ci):
                app[kk * MR + m] = xp[xb + kk]
        for p in range(n_panels):
            var base = p * ci * NR
            var n0 = p * NR
            var c00 = SIMD[F32, W](0)
            var c01 = SIMD[F32, W](0)
            var c10 = SIMD[F32, W](0)
            var c11 = SIMD[F32, W](0)
            var c20 = SIMD[F32, W](0)
            var c21 = SIMD[F32, W](0)
            var c30 = SIMD[F32, W](0)
            var c31 = SIMD[F32, W](0)
            for kk in range(ci):
                var bv0 = bpkp.load[width=W](base + kk * NR)
                var bv1 = bpkp.load[width=W](base + kk * NR + W)
                var av = app.load[width=MR](kk * MR)
                c00 += bv0 * SIMD[F32, W](av[0])
                c01 += bv1 * SIMD[F32, W](av[0])
                c10 += bv0 * SIMD[F32, W](av[1])
                c11 += bv1 * SIMD[F32, W](av[1])
                c20 += bv0 * SIMD[F32, W](av[2])
                c21 += bv1 * SIMD[F32, W](av[2])
                c30 += bv0 * SIMD[F32, W](av[3])
                c31 += bv1 * SIMD[F32, W](av[3])
            if has_bias:
                var bb0 = bp.load[width=W](n0)
                var bb1 = bp.load[width=W](n0 + W)
                c00 += bb0
                c01 += bb1
                c10 += bb0
                c11 += bb1
                c20 += bb0
                c21 += bb1
                c30 += bb0
                c31 += bb1
            var ob = r0 * co + n0
            op.store(ob, c00)
            op.store(ob + W, c01)
            op.store(ob + co, c10)
            op.store(ob + co + W, c11)
            op.store(ob + 2 * co, c20)
            op.store(ob + 2 * co + W, c21)
            op.store(ob + 3 * co, c30)
            op.store(ob + 3 * co + W, c31)
        # column tail (co % NR) for these rows via the dot path
        for oo in range(co_main, co):
            var w_base = oo * ci
            for rr in range(r0, r0 + MR):
                var x_base = rr * ci
                var accv = SIMD[F32, W](0)
                var i = 0
                while i + W <= ci:
                    accv += xp.load[width=W](x_base + i) * wp.load[width=W](w_base + i)
                    i += W
                var acc = accv.reduce_add()
                while i < ci:
                    acc += xp[x_base + i] * wp[w_base + i]
                    i += 1
                if has_bias:
                    acc += bp[oo]
                op[rr * co + oo] = acc

    parallelize[gemm_block](n_gblocks)
    # row tail (rows % MR): with RU == MR the last (partial) dot block is
    # exactly the leftover rows, over all co
    if n_gblocks * MR < rows:
        row_block(n_blocks - 1)
    return out^


struct SparseLinear(Copyable, Movable):
    var weight: Tensor[F32]  # [Co, Ci]
    var bias: Tensor[F32]    # [Co]
    var has_bias: Bool
    # WP11: device copy of the weight, attached by lin_from at model load
    # when TRELLIS2_GPU=1 and the shape qualifies; None -> pure CPU
    var gpu: Optional[GpuLinear]

    def __init__(out self, var weight: Tensor[F32], var bias: Tensor[F32], has_bias: Bool = True):
        self.weight = weight^
        self.bias = bias^
        self.has_bias = has_bias
        self.gpu = None

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        if self.gpu:
            if self.gpu.value().wants(x.numel() // x.shape[len(x.shape) - 1]):
                return self.gpu.value().forward(x)
        return linear(x, self.weight, self.bias, self.has_bias)

    def forward(self, x: VarLenTensor[F32]) raises -> VarLenTensor[F32]:
        return x.replace(self.forward(x.feats))

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        return x.replace(self.forward(x.vl.feats))


# -- norms --------------------------------------------------------------------

struct LayerNorm32(Copyable, Movable):
    """nn.LayerNorm over the last dim. Used directly on sparse feats [T, C]
    and dense [N, L, C] by every transformer block."""

    var num_channels: Int
    var eps: Float64
    var affine: Bool
    var weight: Tensor[F32]  # [C], unused when affine == False
    var bias: Tensor[F32]

    def __init__(out self, num_channels: Int, eps: Float64 = 1e-5, affine: Bool = True) raises:
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.weight = Tensor[F32]([num_channels], 1)
        self.bias = Tensor[F32]([num_channels], 0)

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        comptime W = 8
        var c = self.num_channels
        if x.shape[len(x.shape) - 1] != c:
            raise Error("LayerNorm32: last dim mismatch")
        var rows = x.numel() // c
        var out = Tensor[F32](x.shape)
        var xp = x.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()
        var wp = self.weight.data.unsafe_ptr()
        var bp = self.bias.data.unsafe_ptr()
        for r in range(rows):
            var base = r * c
            var sv = SIMD[F32, W](0)
            var i = 0
            while i + W <= c:
                sv += xp.load[width=W](base + i)
                i += W
            var mean = sv.reduce_add()
            while i < c:
                mean += xp[base + i]
                i += 1
            mean /= Float32(c)
            var vv = SIMD[F32, W](0)
            i = 0
            while i + W <= c:
                var dvec = xp.load[width=W](base + i) - mean
                vv += dvec * dvec
                i += W
            var variance = vv.reduce_add()
            while i < c:
                var d = xp[base + i] - mean
                variance += d * d
                i += 1
            variance /= Float32(c)  # biased, as torch layer_norm
            var inv_std = 1.0 / (variance + Float32(self.eps)) ** 0.5
            i = 0
            if self.affine:
                while i + W <= c:
                    op.store(
                        base + i,
                        (xp.load[width=W](base + i) - mean) * inv_std * wp.load[width=W](i)
                        + bp.load[width=W](i),
                    )
                    i += W
                while i < c:
                    op[base + i] = (xp[base + i] - mean) * inv_std * wp[i] + bp[i]
                    i += 1
            else:
                while i + W <= c:
                    op.store(base + i, (xp.load[width=W](base + i) - mean) * inv_std)
                    i += W
                while i < c:
                    op[base + i] = (xp[base + i] - mean) * inv_std
                    i += 1
        return out^

    def forward(self, x: VarLenTensor[F32]) raises -> VarLenTensor[F32]:
        return x.replace(self.forward(x.feats))

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        return x.replace(self.forward(x.vl.feats))


struct GroupNorm32(Copyable, Movable):
    """nn.GroupNorm on dense [N, C, *spatial] (used by the SS VAE)."""

    var num_groups: Int
    var num_channels: Int
    var eps: Float64
    var affine: Bool
    var weight: Tensor[F32]  # [C]
    var bias: Tensor[F32]

    def __init__(out self, num_groups: Int, num_channels: Int, eps: Float64 = 1e-5, affine: Bool = True) raises:
        if num_channels % num_groups != 0:
            raise Error("GroupNorm32: channels not divisible by groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.weight = Tensor[F32]([num_channels], 1)
        self.bias = Tensor[F32]([num_channels], 0)

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        """The (batch, group) region is one contiguous span, so mean and
        variance are SIMD reductions over it (WP10 pass 6; W-lane
        accumulation + reduce_add reorders the sums relative to the scalar
        loop — within parity tolerance) and the normalize pass is SIMD per
        channel; (batch, group) items are parallelized for large inputs."""
        comptime W = 8
        var n = x.shape[0]
        var c = x.shape[1]
        if c != self.num_channels:
            raise Error("GroupNorm32: channel mismatch")
        var sp = x.numel() // (n * c)
        var cg = c // self.num_groups
        var out = Tensor[F32](x.shape)
        var xp = x.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()
        var wp = self.weight.data.unsafe_ptr()
        var bp = self.bias.data.unsafe_ptr()
        var groups = self.num_groups
        var affine = self.affine
        var eps = Float32(self.eps)

        @parameter
        def item(w: Int):
            var b = w // groups
            var g = w % groups
            var start = (b * c + g * cg) * sp
            var cnt = cg * sp
            var sv = SIMD[F32, W](0)
            var i = 0
            while i + W <= cnt:
                sv += xp.load[width=W](start + i)
                i += W
            var mean = sv.reduce_add()
            while i < cnt:
                mean += xp[start + i]
                i += 1
            mean /= Float32(cnt)
            var vv = SIMD[F32, W](0)
            i = 0
            while i + W <= cnt:
                var dvec = xp.load[width=W](start + i) - mean
                vv += dvec * dvec
                i += W
            var variance = vv.reduce_add()
            while i < cnt:
                var dd = xp[start + i] - mean
                variance += dd * dd
                i += 1
            variance /= Float32(cnt)
            var inv_std = 1.0 / (variance + eps) ** 0.5
            var mv = SIMD[F32, W](mean)
            var isv = SIMD[F32, W](inv_std)
            for ci in range(g * cg, (g + 1) * cg):
                var cbase = (b * c + ci) * sp
                var s = 0
                if affine:
                    var wv = SIMD[F32, W](wp[ci])
                    var bv = SIMD[F32, W](bp[ci])
                    while s + W <= sp:
                        op.store(cbase + s, (xp.load[width=W](cbase + s) - mv) * isv * wv + bv)
                        s += W
                    while s < sp:
                        op[cbase + s] = (xp[cbase + s] - mean) * inv_std * wp[ci] + bp[ci]
                        s += 1
                else:
                    while s + W <= sp:
                        op.store(cbase + s, (xp.load[width=W](cbase + s) - mv) * isv)
                        s += W
                    while s < sp:
                        op[cbase + s] = (xp[cbase + s] - mean) * inv_std
                        s += 1

        var n_items = n * groups
        if n * c * sp < 1 << 17:
            for w in range(n_items):
                item(w)
        else:
            parallelize[item](n_items)
        return out^


struct ChannelLayerNorm32(Copyable, Movable):
    """LayerNorm over the channel dim of dense [N, C, *spatial]
    (permute -> LayerNorm32 -> permute back in the original)."""

    var inner: LayerNorm32

    def __init__(out self, num_channels: Int, eps: Float64 = 1e-5, affine: Bool = True) raises:
        self.inner = LayerNorm32(num_channels, eps, affine)

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        """Vectorized over spatial positions (WP10 pass 6): each SIMD lane
        is one position, accumulating over channels in the same order as
        the scalar loop (the strided per-position walk was the hot spot);
        inv_std uses SIMD sqrt instead of `** 0.5` (ULP-level difference at
        most, within parity tolerance). Spatial chunks are parallelized
        for large inputs."""
        comptime W = 8
        comptime CH = 512
        var n = x.shape[0]
        var c = x.shape[1]
        var sp = x.numel() // (n * c)
        var out = Tensor[F32](x.shape)
        var xp = x.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()
        var wp = self.inner.weight.data.unsafe_ptr()
        var bp = self.inner.bias.data.unsafe_ptr()
        var affine = self.inner.affine
        var eps = Float32(self.inner.eps)
        var chunks_per_b = (sp + CH - 1) // CH

        @parameter
        def item(w: Int):
            var b = w // chunks_per_b
            var s0 = (w % chunks_per_b) * CH
            var s1 = min(s0 + CH, sp)
            var bbase = b * c * sp
            var s = s0
            while s + W <= s1:
                var mv = SIMD[F32, W](0)
                for ci in range(c):
                    mv += xp.load[width=W](bbase + ci * sp + s)
                mv /= Float32(c)
                var vv = SIMD[F32, W](0)
                for ci in range(c):
                    var dv = xp.load[width=W](bbase + ci * sp + s) - mv
                    vv += dv * dv
                vv /= Float32(c)
                var inv = 1.0 / sqrt(vv + eps)
                for ci in range(c):
                    var o = (xp.load[width=W](bbase + ci * sp + s) - mv) * inv
                    if affine:
                        o = o * SIMD[F32, W](wp[ci]) + SIMD[F32, W](bp[ci])
                    op.store(bbase + ci * sp + s, o)
                s += W
            while s < s1:
                var mean: Float32 = 0
                for ci in range(c):
                    mean += xp[bbase + ci * sp + s]
                mean /= Float32(c)
                var variance: Float32 = 0
                for ci in range(c):
                    var d = xp[bbase + ci * sp + s] - mean
                    variance += d * d
                variance /= Float32(c)
                var inv_std = 1.0 / (variance + eps) ** 0.5
                for ci in range(c):
                    var v = (xp[bbase + ci * sp + s] - mean) * inv_std
                    if affine:
                        v = v * wp[ci] + bp[ci]
                    op[bbase + ci * sp + s] = v
                s += 1

        var n_items = n * chunks_per_b
        if n * c * sp < 1 << 17:
            for w in range(n_items):
                item(w)
        else:
            parallelize[item](n_items)
        return out^


# -- nonlinearities -----------------------------------------------------------

def _relu(v: Float32) raises -> Float32:
    return v if v > 0 else 0


def _silu(v: Float32) raises -> Float32:
    return v / (1.0 + exp(-v))


def _gelu(v: Float32) raises -> Float32:
    # torch nn.GELU default (approximate='none'): 0.5 x (1 + erf(x / sqrt(2)))
    return 0.5 * v * (1.0 + erf(v * 0.70710678118654752440))


def _gelu_tanh(v: Float32) raises -> Float32:
    # torch nn.GELU(approximate='tanh'), used by the FFNs
    var inner = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    return 0.5 * v * (1.0 + tanh(inner))


comptime ACT_RELU = 0
comptime ACT_SILU = 1
comptime ACT_GELU = 2
comptime ACT_GELU_TANH = 3


def activation(x: Tensor[F32], kind: Int) raises -> Tensor[F32]:
    """Elementwise, SIMD over the flat buffer with a scalar tail (WP10);
    the vector lanes apply the exact same formulas as the scalar helpers.
    Large buffers are split into W-aligned chunks and parallelized (pass 5)
    — per-element values are unchanged, so results stay bit-identical."""
    comptime W = 8
    comptime CH = 1 << 15
    var out = Tensor[F32](x.shape)
    var n = x.numel()
    var xp = x.data.unsafe_ptr()
    var op = out.data.unsafe_ptr()
    if kind != ACT_RELU and kind != ACT_SILU and kind != ACT_GELU and kind != ACT_GELU_TANH:
        raise Error("unknown activation")

    @parameter
    def chunk(c: Int):
        var i = c * CH
        var hi = min(i + CH, n)
        if kind == ACT_RELU:
            var zero = SIMD[F32, W](0)
            while i + W <= hi:
                op.store(i, max(xp.load[width=W](i), zero))
                i += W
            while i < hi:
                op[i] = xp[i] if xp[i] > 0 else 0
                i += 1
        elif kind == ACT_SILU:
            while i + W <= hi:
                var v = xp.load[width=W](i)
                op.store(i, v / (1.0 + exp(-v)))
                i += W
            while i < hi:
                op[i] = xp[i] / (1.0 + exp(-xp[i]))
                i += 1
        elif kind == ACT_GELU:
            while i + W <= hi:
                var v = xp.load[width=W](i)
                op.store(i, 0.5 * v * (1.0 + erf(v * 0.70710678118654752440)))
                i += W
            while i < hi:
                op[i] = 0.5 * xp[i] * (1.0 + erf(xp[i] * 0.70710678118654752440))
                i += 1
        else:
            while i + W <= hi:
                var v = xp.load[width=W](i)
                var inner = 0.7978845608028654 * (v + 0.044715 * v * v * v)
                op.store(i, 0.5 * v * (1.0 + tanh(inner)))
                i += W
            while i < hi:
                var s = 0.7978845608028654 * (xp[i] + 0.044715 * xp[i] * xp[i] * xp[i])
                op[i] = 0.5 * xp[i] * (1.0 + tanh(s))
                i += 1

    var n_chunks = (n + CH - 1) // CH
    if n < 1 << 17:
        for c in range(n_chunks):
            chunk(c)
    else:
        parallelize[chunk](n_chunks)
    return out^


def activation(x: VarLenTensor[F32], kind: Int) raises -> VarLenTensor[F32]:
    return x.replace(activation(x.feats, kind))


def activation(x: SparseTensor[F32], kind: Int) raises -> SparseTensor[F32]:
    return x.replace(activation(x.vl.feats, kind))


# -- modulation ---------------------------------------------------------------

def modulate(x: Tensor[F32], shift: Tensor[F32], scale: Tensor[F32]) raises -> Tensor[F32]:
    """modules/utils.py: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1).
    x: [N, L, C], shift/scale: [N, C]."""
    var n = x.shape[0]
    var l = x.shape[1]
    var c = x.shape[2]
    if shift.numel() != n * c or scale.numel() != n * c:
        raise Error("modulate: shift/scale shape mismatch")
    comptime W = 8
    var out = Tensor[F32](x.shape)
    var xp = x.data.unsafe_ptr()
    var shp = shift.data.unsafe_ptr()
    var scp = scale.data.unsafe_ptr()
    var op = out.data.unsafe_ptr()
    for b in range(n):
        var mb = b * c
        for j in range(l):
            var xb = (b * l + j) * c
            var ci = 0
            while ci + W <= c:
                op.store(
                    xb + ci,
                    xp.load[width=W](xb + ci) * (1.0 + scp.load[width=W](mb + ci))
                    + shp.load[width=W](mb + ci),
                )
                ci += W
            while ci < c:
                op[xb + ci] = xp[xb + ci] * (1.0 + scp[mb + ci]) + shp[mb + ci]
                ci += 1
    return out^
