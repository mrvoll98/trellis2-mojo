# Mojo port of modules/sparse/spatial/basic.py: SparseDownsample (mean/max
# pooling into the coarser grid) and SparseUpsample (nearest-neighbor via
# the cached down->up index map, or an explicit subdivision tensor).
#
# The down/upsample pair communicates through the scale-keyed spatial cache
# exactly like the original: downsample registers 'upsample_<f>' on the
# coarse tensor so a later upsample can invert it. The training-only
# 'subdivision' cache is not ported (inference scope).
#
# Not ported (phantom exports in sparse/__init__.py, no implementation
# exists upstream): SparseSubdivide, sparse_nearest/trilinear_interpolate.

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix, stable_argsort
from trellis2_mojo.sparse.basic import SparseTensor, CacheValue

comptime F32 = DType.float32

comptime POOL_MEAN = 0
comptime POOL_MAX = 1


struct DownsampleCodes(Copyable, Movable):
    """Shared machinery for SparseDownsample and SparseSpatial2Channel:
    unique coarse-cell codes in sorted order + per-row inverse index."""

    var idx: List[Int]         # row -> index of its coarse cell
    var new_coords: IntMatrix  # [U, 4] coarse coords, batch-contiguous
    var max_dims: List[Int]    # coarse spatial shape (MAX in the original)

    def __init__(out self, x: SparseTensor[F32], factor: Int) raises:
        var dim = x.coords.cols - 1
        var sp = x.spatial_shape()
        self.max_dims = List[Int]()
        for i in range(dim):
            self.max_dims.append((sp[i] + factor - 1) // factor)
        var codes = List[Int]()
        for r in range(x.coords.rows):
            var code = x.coords.at(r, 0)
            for i in range(dim):
                code = code * self.max_dims[i] + x.coords.at(r, i + 1) // factor
            codes.append(code)
        var order = stable_argsort(codes)
        # count unique codes, then decode them into coarse coords
        var n = len(order)
        self.idx = List[Int](length=n, fill=0)
        var uniq = List[Int]()
        for s in range(n):
            if s == 0 or codes[order[s]] != codes[order[s - 1]]:
                uniq.append(codes[order[s]])
            self.idx[order[s]] = len(uniq) - 1
        self.new_coords = IntMatrix(len(uniq), dim + 1)
        for u in range(len(uniq)):
            var code = uniq[u]
            for i in range(dim - 1, -1, -1):
                self.new_coords.set(u, i + 1, code % self.max_dims[i])
                code //= self.max_dims[i]
            self.new_coords.set(u, 0, code)


struct SparseDownsample(Copyable, Movable):
    var factor: Int
    var mode: Int

    def __init__(out self, factor: Int, mode: Int = POOL_MEAN):
        self.factor = factor
        self.mode = mode

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var cache_name = "downsample_" + String(self.factor)
        var new_coords: IntMatrix
        var idx: List[Int]
        var max_dims = List[Int]()
        var cached = x.get_spatial_cache(cache_name)
        if cached:
            var v = cached.value().copy()
            new_coords = v.mat.value().copy()
            idx = v.ints[0].copy()
        else:
            var dc = DownsampleCodes(x, self.factor)
            new_coords = dc.new_coords.copy()
            idx = dc.idx.copy()
            max_dims = dc.max_dims.copy()

        var c = x.vl.feats.shape[1]
        var u = new_coords.rows
        var shape: List[Int] = [u, c]
        var init: Float32 = 0 if self.mode == POOL_MEAN else -3.4e38
        var new_feats = Tensor[F32](shape, init)
        var counts = List[Int](length=u, fill=0)
        for r in range(x.coords.rows):
            counts[idx[r]] += 1
            for ci in range(c):
                var v = x.vl.feats.data[r * c + ci]
                var o = idx[r] * c + ci
                if self.mode == POOL_MEAN:
                    new_feats.data[o] += v
                elif v > new_feats.data[o]:
                    new_feats.data[o] = v
        if self.mode == POOL_MEAN:
            for uu in range(u):
                for ci in range(c):
                    new_feats.data[uu * c + ci] /= Float32(counts[uu])

        var out = SparseTensor[F32](new_feats^, new_coords.copy(), x.batch_size())
        out.scale = [
            x.scale[0].mul_int(self.factor),
            x.scale[1].mul_int(self.factor),
            x.scale[2].mul_int(self.factor),
        ]
        out.cache = x.cache

        if not cached:
            x.register_spatial_cache(cache_name, CacheValue.from_mat_idx(new_coords, idx))
            out.register_spatial_cache(
                "upsample_" + String(self.factor), CacheValue.from_mat_idx(x.coords, idx)
            )
            out.register_spatial_cache("shape", CacheValue.from_shape(max_dims))
        return out^


struct SparseUpsample(Copyable, Movable):
    var factor: Int

    def __init__(out self, factor: Int):
        self.factor = factor

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        """Cache path: requires a preceding SparseDownsample on the lineage."""
        var cached = x.get_spatial_cache("upsample_" + String(self.factor))
        if not cached:
            raise Error("SparseUpsample: cache not found; provide subdivision or pair with SparseDownsample")
        var v = cached.value().copy()
        var new_coords = v.mat.value().copy()
        var idx = v.ints[0].copy()
        var out = SparseTensor[F32](x.vl.feats.select_rows(idx), new_coords^, x.batch_size())
        out.scale = [
            x.scale[0].div_int(self.factor),
            x.scale[1].div_int(self.factor),
            x.scale[2].div_int(self.factor),
        ]
        out.cache = x.cache
        return out^

    def forward_subdivision(
        self, x: SparseTensor[F32], subdivision: SparseTensor[F32]
    ) raises -> SparseTensor[F32]:
        """Subdivision path: sub feats [N, factor^3] of 0/1 select which
        children of each coarse voxel exist. Fresh spatial cache (as the
        original: cache only kept on the cached path)."""
        var dim = x.coords.cols - 1
        var f3 = self.factor ** dim
        var sub = subdivision.vl.feats.copy()
        var n = x.coords.rows
        var idx = List[Int]()
        var subidx = List[Int]()
        for r in range(n):
            for c in range(f3):
                if sub.data[r * f3 + c] != 0:
                    idx.append(r)
                    subidx.append(c)
        var m = len(idx)
        var new_coords = IntMatrix(m, dim + 1)
        for e in range(m):
            new_coords.set(e, 0, x.coords.at(idx[e], 0))
            for i in range(dim):
                var child = subidx[e] // self.factor ** i % self.factor
                new_coords.set(e, i + 1, x.coords.at(idx[e], i + 1) * self.factor + child)
        var out = SparseTensor[F32](x.vl.feats.select_rows(idx), new_coords^, x.batch_size())
        out.scale = [
            x.scale[0].div_int(self.factor),
            x.scale[1].div_int(self.factor),
            x.scale[2].div_int(self.factor),
        ]
        return out^
