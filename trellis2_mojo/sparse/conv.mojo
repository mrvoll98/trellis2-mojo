# Mojo port of modules/sparse/conv/conv_none.py: submanifold sparse 3D
# convolution (stride 1) via gather-multiply-scatter over a coordinate hash.
# This is the portable backend; spconv/flex_gemm/torchsparse stay Python.
#
# Weight layout follows flex_gemm/conv_none: [Co, Kd, Kh, Kw, Ci], where the
# kernel depth/height/width axes offset coordinate columns 1/2/3. The
# neighbor map is cached in the spatial cache keyed by kernel and dilation.
# SparseInverseConv3d is unimplemented in the original 'none' backend too.

from max.algorithm import parallelize

from trellis2_mojo.gpu.conv import GpuSparseConv
from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.basic import SparseTensor, CacheValue

comptime F32 = DType.float32


struct SparseConv3d(Copyable, Movable):
    var in_channels: Int
    var out_channels: Int
    var kernel_size: List[Int]  # (Kd, Kh, Kw)
    var dilation: List[Int]
    var weight: Tensor[F32]     # [Co, Kd, Kh, Kw, Ci]
    var bias: Tensor[F32]       # [Co]
    var has_bias: Bool
    # WP11 step 6: set by sparse_conv3d_from when TRELLIS2_GPU=1 — big
    # convs run the GPU gather kernel on the same edge lists
    var gpu: Optional[GpuSparseConv]

    def __init__(
        out self,
        var weight: Tensor[F32],
        var bias: Tensor[F32],
        has_bias: Bool = True,
        dilation: Int = 1,
    ) raises:
        if weight.ndim() != 5:
            raise Error("SparseConv3d: weight must be [Co, Kd, Kh, Kw, Ci]")
        self.out_channels = weight.shape[0]
        self.in_channels = weight.shape[4]
        self.kernel_size = [weight.shape[1], weight.shape[2], weight.shape[3]]
        self.dilation = [dilation, dilation, dilation]
        self.weight = weight^
        self.bias = bias^
        self.has_bias = has_bias
        self.gpu = None

    def _cache_name(self) raises -> String:
        return (
            "SubMConv3d_naive_neighbor_" + String(self.kernel_size[2]) + "x"
            + String(self.kernel_size[1]) + "x" + String(self.kernel_size[0])
            + "_dilation" + String(self.dilation[0])
        )

    def _neighbor_map(self, x: SparseTensor[F32]) raises -> Tuple[List[Int], List[Int], List[Int]]:
        """(src, tgt, kernel_idx) edge lists; cached per coords + kernel."""
        var cache_name = self._cache_name()
        var cached = x.get_spatial_cache(cache_name)
        if cached:
            var v = cached.value().copy()
            return (v.ints[0].copy(), v.ints[1].copy(), v.ints[2].copy())

        # coordinate hash: coords are in [0, 1024); offsets shift by +512 to
        # keep packed keys positive and collision-free
        var coord_to_idx = Dict[Int, Int]()
        for r in range(x.coords.rows):
            var key = ((x.coords.at(r, 0) * 2048 + x.coords.at(r, 1) + 512) * 2048
                       + x.coords.at(r, 2) + 512) * 2048 + x.coords.at(r, 3) + 512
            coord_to_idx[key] = r

        var kd = self.kernel_size[0]
        var kh = self.kernel_size[1]
        var kw = self.kernel_size[2]
        var src = List[Int]()
        var tgt = List[Int]()
        var kidx = List[Int]()
        for kz in range(kd):
            for ky in range(kh):
                for kx in range(kw):
                    var oz = (kz - kd // 2) * self.dilation[0]
                    var oy = (ky - kh // 2) * self.dilation[1]
                    var ox = (kx - kw // 2) * self.dilation[2]
                    var k = kz * kh * kw + ky * kw + kx
                    for r in range(x.coords.rows):
                        var key = ((x.coords.at(r, 0) * 2048 + x.coords.at(r, 1) + oz + 512) * 2048
                                   + x.coords.at(r, 2) + oy + 512) * 2048 + x.coords.at(r, 3) + ox + 512
                        var hit = coord_to_idx.get(key)
                        if hit:
                            src.append(hit.value())
                            tgt.append(r)
                            kidx.append(k)

        var cv = CacheValue.from_ints([src.copy(), tgt.copy(), kidx.copy()])
        x.register_spatial_cache(cache_name, cv^)
        return (src^, tgt^, kidx^)

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var edges = self._neighbor_map(x)
        var src = edges[0].copy()
        var tgt = edges[1].copy()
        var kidx = edges[2].copy()

        if self.gpu:
            if self.gpu.value().wants(len(src)):
                # WP11 step 11: the CSR sort depends ONLY on the edges, so
                # it is cached next to the neighbor map (keyed per coords +
                # kernel + dilation) — before this, every conv on the same
                # coords re-ran the counting sort per call
                var csr_name = self._cache_name() + "_csr"
                var csr_cached = x.get_spatial_cache(csr_name)
                if csr_cached:
                    var v = csr_cached.value().copy()
                    return x.replace(self.gpu.value().forward(
                        x.vl.feats, v.ints[0], v.ints[1], v.ints[2]
                    ))
                var n_rows = x.coords.rows
                var e_total = len(src)
                var row_start = List[Int](length=n_rows + 1, fill=0)
                var src_s = List[Int](length=e_total, fill=0)
                var kidx_s = List[Int](length=e_total, fill=0)
                # stable counting sort by target (the kidx-major build
                # order is preserved within each row — the GPU kernel's
                # merge walk relies on it)
                var counts = List[Int](length=n_rows, fill=0)
                for e in range(e_total):
                    counts[tgt[e]] += 1
                var run = 0
                for r in range(n_rows):
                    row_start[r] = run
                    run += counts[r]
                row_start[n_rows] = run
                var cur = List[Int](length=n_rows, fill=0)
                for r in range(n_rows):
                    cur[r] = row_start[r]
                for e in range(e_total):
                    var t = tgt[e]
                    var p = cur[t]
                    src_s[p] = src[e]
                    kidx_s[p] = kidx[e]
                    cur[t] = (p + 1)
                x.register_spatial_cache(csr_name, CacheValue.from_ints(
                    [row_start.copy(), src_s.copy(), kidx_s.copy()]
                ))
                return x.replace(self.gpu.value().forward(
                    x.vl.feats, row_start, src_s, kidx_s
                ))

        var n = x.coords.rows
        var ci = self.in_channels
        var co = self.out_channels
        var ksize = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        var shape: List[Int] = [n, co]
        var out = Tensor[F32](shape)
        var n_edges = len(src)
        var xp = x.vl.feats.data.unsafe_ptr()
        var wp = self.weight.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()
        var srcp = src.unsafe_ptr()
        var tgtp = tgt.unsafe_ptr()
        var kidxp = kidx.unsafe_ptr()

        # SIMD dot over Ci; parallel work items are output-channel chunks so
        # every item keeps the full edge order on disjoint out columns ->
        # bit-identical to the serial path (WP10).
        comptime W = 8
        comptime O_CHUNK = 8
        var n_work = (co + O_CHUNK - 1) // O_CHUNK

        @parameter
        def chunk(wk: Int):
            var o_lo = wk * O_CHUNK
            var o_hi = min(o_lo + O_CHUNK, co)
            for e in range(n_edges):
                var x_base = srcp[unsafe_offset=e] * ci
                var t_base = tgtp[unsafe_offset=e] * co
                for o in range(o_lo, o_hi):
                    var w_base = (o * ksize + kidxp[unsafe_offset=e]) * ci
                    var accv = SIMD[F32, W](0)
                    var i = 0
                    while i + W <= ci:
                        accv += xp.unsafe_load[width=W](x_base + i) * wp.unsafe_load[width=W](w_base + i)
                        i += W
                    var acc = accv.reduce_add()
                    while i < ci:
                        acc += xp[unsafe_offset=x_base + i] * wp[unsafe_offset=w_base + i]
                        i += 1
                    op[unsafe_offset=t_base + o] += acc

        if n_work == 1 or n_edges * co * ci < 1 << 16:
            for wk in range(n_work):
                chunk(wk)
        else:
            parallelize[chunk](n_work)
        if self.has_bias:
            for r in range(n):
                for o in range(co):
                    out.data[r * co + o] += self.bias.data[o]
        return x.replace(out^)
