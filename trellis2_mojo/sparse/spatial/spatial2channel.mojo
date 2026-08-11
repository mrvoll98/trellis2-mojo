# Mojo port of modules/sparse/spatial/spatial2channel.py:
# SparseSpatial2Channel (fold factor^3 children into channels, zeros for
# missing children) and SparseChannel2Spatial (unfold back via the cached
# index map or an explicit subdivision tensor).

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix
from trellis2_mojo.sparse.basic import SparseTensor, CacheValue
from trellis2_mojo.sparse.spatial.basic import DownsampleCodes

comptime F32 = DType.float32


def _subidx_of(x: SparseTensor[F32], factor: Int) raises -> List[Int]:
    """Child slot per row: sum((coord[i+1] % f) * f^i) — axis x in low bits."""
    var dim = x.coords.cols - 1
    var out = List[Int]()
    for r in range(x.coords.rows):
        var s = 0
        for i in range(dim):
            s += (x.coords.at(r, i + 1) % factor) * factor ** i
        out.append(s)
    return out^


struct SparseSpatial2Channel(Copyable, Movable):
    var factor: Int

    def __init__(out self, factor: Int = 2):
        self.factor = factor

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var dim = x.coords.cols - 1
        var f3 = self.factor ** dim
        var cache_name = "spatial2channel_" + String(self.factor)
        var new_coords: IntMatrix
        var idx: List[Int]
        var subidx: List[Int]
        var max_dims = List[Int]()
        var cached = x.get_spatial_cache(cache_name)
        if cached:
            var v = cached.value().copy()
            new_coords = v.mat.value().copy()
            idx = v.ints[0].copy()
            subidx = v.ints[1].copy()
        else:
            var dc = DownsampleCodes(x, self.factor)
            new_coords = dc.new_coords.copy()
            idx = dc.idx.copy()
            max_dims = dc.max_dims.copy()
            subidx = _subidx_of(x, self.factor)

        var c = x.vl.feats.shape[1]
        var u = new_coords.rows
        var shape: List[Int] = [u, c * f3]
        var new_feats = Tensor[F32](shape)
        for r in range(x.coords.rows):
            var slot = idx[r] * f3 + subidx[r]
            for ci in range(c):
                new_feats.data[slot * c + ci] = x.vl.feats.data[r * c + ci]

        var out = SparseTensor[F32](new_feats^, new_coords.copy(), x.batch_size())
        out.scale = [
            x.scale[0].mul_int(self.factor),
            x.scale[1].mul_int(self.factor),
            x.scale[2].mul_int(self.factor),
        ]
        out.cache = x.cache

        if not cached:
            var cv = CacheValue.from_mat_idx(new_coords, idx)
            cv.ints.append(subidx.copy())
            x.register_spatial_cache(cache_name, cv^)
            var cv2 = CacheValue.from_mat_idx(x.coords, idx)
            cv2.ints.append(subidx.copy())
            out.register_spatial_cache("channel2spatial_" + String(self.factor), cv2^)
            out.register_spatial_cache("shape", CacheValue.from_shape(max_dims))
        return out^


struct SparseChannel2Spatial(Copyable, Movable):
    var factor: Int

    def __init__(out self, factor: Int = 2):
        self.factor = factor

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        """Cache path: requires a preceding SparseSpatial2Channel."""
        var cached = x.get_spatial_cache("channel2spatial_" + String(self.factor))
        if not cached:
            raise Error("SparseChannel2Spatial: cache not found; provide subdivision or pair with SparseSpatial2Channel")
        var v = cached.value().copy()
        var new_coords = v.mat.value().copy()
        var idx = v.ints[0].copy()
        var subidx = v.ints[1].copy()
        return self._unfold(x, new_coords^, idx, subidx, share_cache=True)

    def forward_subdivision(
        self, x: SparseTensor[F32], subdivision: SparseTensor[F32]
    ) raises -> SparseTensor[F32]:
        var dim = x.coords.cols - 1
        var f3 = self.factor ** dim
        var sub = subdivision.vl.feats.copy()
        var idx = List[Int]()
        var subidx = List[Int]()
        for r in range(x.coords.rows):
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
        return self._unfold(x, new_coords^, idx, subidx, share_cache=False)

    def _unfold(
        self,
        x: SparseTensor[F32],
        var new_coords: IntMatrix,
        idx: List[Int],
        subidx: List[Int],
        share_cache: Bool,
    ) raises -> SparseTensor[F32]:
        var dim = x.coords.cols - 1
        var f3 = self.factor ** dim
        var c_out = x.vl.feats.shape[1] // f3
        var m = len(idx)
        var shape: List[Int] = [m, c_out]
        var new_feats = Tensor[F32](shape)
        for e in range(m):
            var slot = idx[e] * f3 + subidx[e]
            for ci in range(c_out):
                new_feats.data[e * c_out + ci] = x.vl.feats.data[slot * c_out + ci]
        var out = SparseTensor[F32](new_feats^, new_coords^, x.batch_size())
        out.scale = [
            x.scale[0].div_int(self.factor),
            x.scale[1].div_int(self.factor),
            x.scale[2].div_int(self.factor),
        ]
        if share_cache:
            out.cache = x.cache
        return out^
