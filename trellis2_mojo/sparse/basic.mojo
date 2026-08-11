# Mojo port of trellis2/modules/sparse/basic.py (VarLenTensor, SparseTensor).
#
# Design (per docs/decisions/0003 + docs/conversion/sparse_tensor_in_mojo.md):
# - Feats/coords live in the minimal Tensor/IntMatrix from tensor.mojo.
# - Python's List[slice] layout is represented as an offsets array of length
#   B+1 (== cum_seqlen), which also replaces the seqlen/cum_seqlen caches.
# - Python's _spatial_cache dict is shared by reference between tensors that
#   descend from each other (replace/downsample). Mojo has value semantics, so
#   the cache lives behind an ArcPointer. Keys are flattened "scale::name".
# - No inheritance in Mojo: SparseTensor composes a VarLenTensor instead of
#   subclassing it.
# - v1 is CPU + correctness only. dtype is a struct parameter; fp16 parity and
#   performance (SIMD/GPU) are later work packages.

from std.memory import ArcPointer

from trellis2_mojo.sparse.tensor import (
    Tensor,
    IntMatrix,
    Frac,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    RED_SUM,
    RED_MEAN,
    RED_PROD,
)


struct CacheValue(Copyable, Movable):
    """Heterogeneous value for the spatial cache. Exactly one 'shape' of
    payload is used per key: index lists (conv neighbor maps, idx maps),
    a coordinate matrix + index list (down/upsample caches), or a shape."""

    var ints: List[List[Int]]
    var mat: Optional[IntMatrix]
    var shape: List[Int]
    var floats: List[Tensor[DType.float32]]

    def __init__(out self) raises:
        self.ints = List[List[Int]]()
        self.mat = None
        self.shape = List[Int]()
        self.floats = List[Tensor[DType.float32]]()

    @staticmethod
    def from_shape(shape: List[Int]) raises -> Self:
        var v = Self()
        v.shape = shape.copy()
        return v^

    @staticmethod
    def from_ints(ints: List[List[Int]]) raises -> Self:
        var v = Self()
        v.ints = ints.copy()
        return v^

    @staticmethod
    def from_mat_idx(mat: IntMatrix, idx: List[Int]) raises -> Self:
        var v = Self()
        v.mat = mat.copy()
        v.ints.append(idx.copy())
        return v^

    @staticmethod
    def from_tensor(t: Tensor[DType.float32]) raises -> Self:
        var v = Self()
        v.floats.append(t.copy())
        return v^


comptime SpatialCache = Dict[String, CacheValue]


def _offsets_ok(offsets: List[Int], total_rows: Int) raises -> Bool:
    if len(offsets) < 1 or offsets[0] != 0:
        return False
    for i in range(1, len(offsets)):
        if offsets[i] < offsets[i - 1]:
            return False
    return offsets[len(offsets) - 1] == total_rows


# =============================================================================
# VarLenTensor
# =============================================================================

struct VarLenTensor[dtype: DType](Copyable, Movable):
    """Sequential tensor with variable length per batch element.

    feats:   [T, *tail] where T = sum of per-batch lengths.
    offsets: length B+1; batch b owns rows [offsets[b], offsets[b+1]).
    """

    var feats: Tensor[Self.dtype]
    var offsets: List[Int]

    def __init__(out self, var feats: Tensor[Self.dtype], offsets: List[Int]) raises:
        if not _offsets_ok(offsets, feats.rows()):
            raise Error("VarLenTensor: invalid offsets")
        self.feats = feats^
        self.offsets = offsets.copy()

    @staticmethod
    def offsets_from_seqlen(seqlen: List[Int]) raises -> List[Int]:
        var offsets: List[Int] = [0]
        var start = 0
        for l in seqlen:
            start += l
            offsets.append(start)
        return offsets^

    @staticmethod
    def from_tensor_list(tensors: List[Tensor[Self.dtype]]) raises -> Self:
        var feats = Tensor[Self.dtype].cat_rows(tensors)
        var seqlen = List[Int]()
        for i in range(len(tensors)):
            seqlen.append(tensors[i].rows())
        return Self(feats^, Self.offsets_from_seqlen(seqlen))

    def to_tensor_list(self) raises -> List[Tensor[Self.dtype]]:
        var out = List[Tensor[Self.dtype]]()
        for b in range(self.batch_size()):
            out.append(self.feats.slice_rows(self.offsets[b], self.offsets[b + 1]))
        return out^

    def batch_size(self) raises -> Int:
        return len(self.offsets) - 1

    def __len__(self) raises -> Int:
        return self.batch_size()

    def shape(self) raises -> List[Int]:
        var s: List[Int] = [self.batch_size()]
        for d in self.feats.tail_shape():
            s.append(d)
        return s^

    def seqlen(self, b: Int) raises -> Int:
        return self.offsets[b + 1] - self.offsets[b]

    def seqlen_list(self) raises -> List[Int]:
        var out = List[Int]()
        for b in range(self.batch_size()):
            out.append(self.seqlen(b))
        return out^

    def batch_broadcast_map(self) raises -> List[Int]:
        """Row index -> batch index (Python's batch_boardcast_map)."""
        var out = List[Int]()
        for b in range(self.batch_size()):
            for _ in range(self.seqlen(b)):
                out.append(b)
        return out^

    def replace(self, var feats: Tensor[Self.dtype]) raises -> Self:
        if feats.rows() != self.feats.rows():
            raise Error("VarLenTensor.replace: row count mismatch")
        return Self(feats^, self.offsets)

    def reshape(self, tail: List[Int]) raises -> Self:
        return self.replace(self.feats.reshape_rows(tail))

    def cast[target: DType](self) raises -> VarLenTensor[target]:
        return VarLenTensor[target](self.feats.cast[target](), self.offsets)

    def to_dense(self, max_length: Int = -1) raises -> Tuple[Tensor[Self.dtype], Tensor[DType.uint8]]:
        """-> (dense [B, L, *tail], mask [B, L]) with zero padding."""
        var b = self.batch_size()
        var l = max_length
        if l < 0:
            l = 0
            for i in range(b):
                if self.seqlen(i) > l:
                    l = self.seqlen(i)
        var dense_shape: List[Int] = [b, l]
        for d in self.feats.tail_shape():
            dense_shape.append(d)
        var dense = Tensor[Self.dtype](dense_shape)
        var mask = Tensor[DType.uint8]([b, l])
        var rs = self.feats.row_size()
        for i in range(b):
            for j in range(self.seqlen(i)):
                var src = self.offsets[i] + j
                for k in range(rs):
                    dense.data[(i * l + j) * rs + k] = self.feats.data[src * rs + k]
                mask.data[i * l + j] = 1
        return (dense^, mask^)

    # -- elementwise ---------------------------------------------------------
    # Three operand kinds, mirroring VarLenTensor.__elemwise__:
    #   elemwise:        other is a VarLenTensor/flat tensor with same rows
    #   elemwise_scalar: other is a scalar
    #   elemwise_batch:  other is [B, *tail] (or [B]) broadcast over each
    #                    batch's rows via batch_broadcast_map

    def elemwise(self, other: Self, op: Int) raises -> Self:
        return self.replace(self.feats._binop_flat(other.feats, op))

    def elemwise_scalar(self, other: Scalar[Self.dtype], op: Int, reverse: Bool = False) raises -> Self:
        return self.replace(self.feats._binop_scalar(other, op, reverse))

    def elemwise_batch(self, other: Tensor[Self.dtype], op: Int) raises -> Self:
        if other.rows() != self.batch_size():
            raise Error("VarLenTensor.elemwise_batch: batch size mismatch")
        return self.replace(self.feats._binop_rows(other, self.batch_broadcast_map(), op))

    def __add__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_ADD)

    def __add__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_ADD)

    def __sub__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_SUB)

    def __sub__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_SUB)

    def __mul__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_MUL)

    def __mul__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_MUL)

    def __truediv__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_DIV)

    def __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_DIV)

    def __neg__(self) raises -> Self:
        return self.elemwise_scalar(-1, OP_MUL)

    # -- indexing ------------------------------------------------------------

    def __getitem__(self, idx: List[Int]) raises -> Self:
        """Select/reorder batches (Python __getitem__ with int list)."""
        var parts = List[Tensor[Self.dtype]]()
        for i in idx:
            parts.append(self.feats.slice_rows(self.offsets[i], self.offsets[i + 1]))
        return Self.from_tensor_list(parts)

    def unbind(self, dim: Int) raises -> List[Self]:
        """dim 0: split into per-batch VarLenTensors; else unbind feats dim."""
        var out = List[Self]()
        if dim == 0:
            for b in range(self.batch_size()):
                var one: List[Int] = [b]
                out.append(self[one])
        else:
            for f in self.feats.unbind(dim):
                out.append(self.replace(f.copy()))
        return out^

    # -- reductions ----------------------------------------------------------

    def reduce_all(self, op: Int) raises -> Scalar[Self.dtype]:
        return self.feats.reduce_all(op)

    def reduce_batch(self, op: Int) raises -> Tensor[Self.dtype]:
        """Reduce feature dims per row, then segment-reduce rows per batch.
        Mirrors VarLenTensor.reduce(op, dim=feature dims) -> [B]."""
        var per_row = self.feats.reduce_tail(op)
        return per_row.segment_reduce(self.offsets, op)


def varlen_cat[dt: DType](inputs: List[VarLenTensor[dt]], dim: Int = 0) raises -> VarLenTensor[dt]:
    if len(inputs) == 0:
        raise Error("varlen_cat: empty input")
    if dim == 0:
        var feats_list = List[Tensor[dt]]()
        var offsets: List[Int] = [0]
        var start = 0
        for i in range(len(inputs)):
            feats_list.append(inputs[i].feats.copy())
            for b in range(inputs[i].batch_size()):
                start += inputs[i].seqlen(b)
                offsets.append(start)
        return VarLenTensor[dt](Tensor[dt].cat_rows(feats_list), offsets)
    var feats = inputs[0].feats.copy()
    for i in range(1, len(inputs)):
        feats = feats.cat_dim(inputs[i].feats, dim)
    return inputs[0].replace(feats^)


def varlen_unbind[dt: DType](input: VarLenTensor[dt], dim: Int) raises -> List[VarLenTensor[dt]]:
    return input.unbind(dim)


# =============================================================================
# SparseTensor
# =============================================================================

struct SparseTensor[dtype: DType](Copyable, Movable):
    """N-D sparse tensor: feats [N, *tail] + coords [N, 4] (batch, x, y, z).

    Rows for the same batch must be contiguous and batches ordered (same
    contract as the Python original). scale tracks down/upsampling as exact
    rationals; the spatial cache is keyed by scale and shared by reference
    across derived tensors (ArcPointer)."""

    var vl: VarLenTensor[Self.dtype]
    var coords: IntMatrix
    var scale: List[Frac]
    var cache: ArcPointer[SpatialCache]

    def __init__(out self, var feats: Tensor[Self.dtype], var coords: IntMatrix, batch_size: Int = -1) raises:
        if feats.rows() != coords.rows:
            raise Error("SparseTensor: feats/coords row mismatch")
        var b = batch_size
        if b < 0:
            b = coords.col_max(0) + 1
        self.vl = VarLenTensor[Self.dtype](feats^, Self._cal_offsets(coords, b))
        self.coords = coords^
        self.scale = [Frac(1, 1), Frac(1, 1), Frac(1, 1)]
        self.cache = ArcPointer(SpatialCache())

    @staticmethod
    def _cal_offsets(coords: IntMatrix, batch_size: Int) raises -> List[Int]:
        """bincount over the batch column + cumsum (Python __cal_layout).
        Also enforces the batch-contiguity contract."""
        var counts = List[Int](length=batch_size, fill=0)
        var prev = -1
        for r in range(coords.rows):
            var b = coords.at(r, 0)
            if b < prev:
                raise Error("SparseTensor: batches must be ordered/contiguous")
            if b >= batch_size:
                raise Error("SparseTensor: batch index out of range")
            prev = b
            counts[b] += 1
        var offsets: List[Int] = [0]
        var start = 0
        for b in range(batch_size):
            start += counts[b]
            offsets.append(start)
        return offsets^

    @staticmethod
    def from_tensor_list(
        feats_list: List[Tensor[Self.dtype]], coords_list: List[IntMatrix]
    ) raises -> Self:
        """Batch column of each coords entry is overwritten with the list index."""
        if len(feats_list) != len(coords_list):
            raise Error("SparseTensor.from_tensor_list: length mismatch")
        var coords = List[IntMatrix]()
        for i in range(len(coords_list)):
            var c = coords_list[i].copy()
            for r in range(c.rows):
                c.set(r, 0, i)
            coords.append(c^)
        return Self(
            Tensor[Self.dtype].cat_rows(feats_list),
            IntMatrix.cat_rows(coords),
            len(feats_list),
        )

    def to_tensor_list(self) raises -> Tuple[List[Tensor[Self.dtype]], List[IntMatrix]]:
        var feats_list = self.vl.to_tensor_list()
        var coords_list = List[IntMatrix]()
        for b in range(self.batch_size()):
            coords_list.append(self.coords.slice_rows(self.vl.offsets[b], self.vl.offsets[b + 1]))
        return (feats_list^, coords_list^)

    # -- forwarding to the varlen view ---------------------------------------

    def batch_size(self) raises -> Int:
        return self.vl.batch_size()

    def __len__(self) raises -> Int:
        return self.batch_size()

    def shape(self) raises -> List[Int]:
        return self.vl.shape()

    def seqlen(self, b: Int) raises -> Int:
        return self.vl.seqlen(b)

    def seqlen_list(self) raises -> List[Int]:
        return self.vl.seqlen_list()

    def batch_broadcast_map(self) raises -> List[Int]:
        return self.vl.batch_broadcast_map()

    def spatial_shape(self) raises -> List[Int]:
        var cached = self.get_spatial_cache("shape")
        if cached:
            return cached.value().shape.copy()
        var s = List[Int]()
        for c in range(1, self.coords.cols):
            s.append(self.coords.col_max(c) + 1)
        self.register_spatial_cache("shape", CacheValue.from_shape(s))
        return s^

    # -- derivation ----------------------------------------------------------

    def replace(self, var feats: Tensor[Self.dtype]) raises -> Self:
        """New tensor with same coords/scale and a shared spatial cache."""
        var out = Self(feats^, self.coords.copy(), self.batch_size())
        out.scale = self.scale.copy()
        out.cache = self.cache
        return out^

    def replace_with_coords(self, var feats: Tensor[Self.dtype], var coords: IntMatrix) raises -> Self:
        var out = Self(feats^, coords^, self.batch_size())
        out.scale = self.scale.copy()
        out.cache = self.cache
        return out^

    def with_scale(self, scale: List[Frac]) raises -> Self:
        var out = self.copy()
        out.scale = scale.copy()
        return out^

    def reshape(self, tail: List[Int]) raises -> Self:
        return self.replace(self.vl.feats.reshape_rows(tail))

    def cast[target: DType](self) raises -> SparseTensor[target]:
        var out = SparseTensor[target](self.vl.feats.cast[target](), self.coords.copy(), self.batch_size())
        out.scale = self.scale.copy()
        return out^

    # -- elementwise ---------------------------------------------------------

    def elemwise(self, other: Self, op: Int) raises -> Self:
        return self.replace(self.vl.feats._binop_flat(other.vl.feats, op))

    def elemwise_scalar(self, other: Scalar[Self.dtype], op: Int, reverse: Bool = False) raises -> Self:
        return self.replace(self.vl.feats._binop_scalar(other, op, reverse))

    def elemwise_batch(self, other: Tensor[Self.dtype], op: Int) raises -> Self:
        if other.rows() != self.batch_size():
            raise Error("SparseTensor.elemwise_batch: batch size mismatch")
        return self.replace(self.vl.feats._binop_rows(other, self.batch_broadcast_map(), op))

    def __add__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_ADD)

    def __add__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_ADD)

    def __sub__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_SUB)

    def __sub__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_SUB)

    def __mul__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_MUL)

    def __mul__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_MUL)

    def __truediv__(self, other: Self) raises -> Self:
        return self.elemwise(other, OP_DIV)

    def __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.elemwise_scalar(other, OP_DIV)

    def __neg__(self) raises -> Self:
        return self.elemwise_scalar(-1, OP_MUL)

    # -- indexing ------------------------------------------------------------

    def __getitem__(self, idx: List[Int]) raises -> Self:
        """Select/reorder batches; batch column is renumbered 0..len(idx)-1.
        Fresh spatial cache (coords change relative to the parent)."""
        var feats_parts = List[Tensor[Self.dtype]]()
        var coords_parts = List[IntMatrix]()
        for new_idx in range(len(idx)):
            var old_idx = idx[new_idx]
            var s = self.vl.offsets[old_idx]
            var e = self.vl.offsets[old_idx + 1]
            feats_parts.append(self.vl.feats.slice_rows(s, e))
            var c = self.coords.slice_rows(s, e)
            for r in range(c.rows):
                c.set(r, 0, new_idx)
            coords_parts.append(c^)
        var out = Self(
            Tensor[Self.dtype].cat_rows(feats_parts),
            IntMatrix.cat_rows(coords_parts),
            len(idx),
        )
        out.scale = self.scale.copy()
        return out^

    def unbind(self, dim: Int) raises -> List[Self]:
        return sparse_unbind(self, dim)

    # -- dense conversion ----------------------------------------------------

    def to_dense(self) raises -> Tensor[Self.dtype]:
        """-> [B, C, *spatial] for rank-2 feats [N, C]."""
        if self.vl.feats.ndim() != 2:
            raise Error("SparseTensor.to_dense: only rank-2 feats supported")
        var b = self.batch_size()
        var ch = self.vl.feats.shape[1]
        var sp = self.spatial_shape()
        var dense_shape: List[Int] = [b, ch]
        var sp_size = 1
        for d in sp:
            dense_shape.append(d)
            sp_size *= d
        var dense = Tensor[Self.dtype](dense_shape)
        for r in range(self.coords.rows):
            var bi = self.coords.at(r, 0)
            var flat = 0
            for a in range(len(sp)):
                flat = flat * sp[a] + self.coords.at(r, a + 1)
            for c in range(ch):
                dense.data[(bi * ch + c) * sp_size + flat] = self.vl.feats.at(r, c)
        return dense^

    @staticmethod
    def full(aabb: List[Int], n: Int, c: Int, value: Scalar[Self.dtype]) raises -> Self:
        """Dense axis-aligned box of voxels, feats filled with `value`.
        aabb = [x0, y0, z0, x1, y1, z1] inclusive."""
        if len(aabb) != 6:
            raise Error("SparseTensor.full: aabb must have 6 entries")
        var nx = aabb[3] - aabb[0] + 1
        var ny = aabb[4] - aabb[1] + 1
        var nz = aabb[5] - aabb[2] + 1
        var per_batch = nx * ny * nz
        var coords = IntMatrix(n * per_batch, 4)
        var r = 0
        for b in range(n):
            for x in range(aabb[0], aabb[3] + 1):
                for y in range(aabb[1], aabb[4] + 1):
                    for z in range(aabb[2], aabb[5] + 1):
                        coords.set(r, 0, b)
                        coords.set(r, 1, x)
                        coords.set(r, 2, y)
                        coords.set(r, 3, z)
                        r += 1
        var feats = Tensor[Self.dtype]([n * per_batch, c], value)
        return Self(feats^, coords^, n)

    # -- spatial cache -------------------------------------------------------

    def _scale_key(self) raises -> String:
        return (
            self.scale[0].to_string() + ","
            + self.scale[1].to_string() + ","
            + self.scale[2].to_string()
        )

    def register_spatial_cache(self, key: String, var value: CacheValue) raises:
        var cache = self.cache
        cache[][self._scale_key() + "::" + key] = value^

    def get_spatial_cache(self, key: String) raises -> Optional[CacheValue]:
        return self.cache[].get(self._scale_key() + "::" + key)

    def clear_spatial_cache(self) raises:
        var cache = self.cache
        cache[] = SpatialCache()

    # -- reductions ----------------------------------------------------------

    def reduce_all(self, op: Int) raises -> Scalar[Self.dtype]:
        return self.vl.reduce_all(op)

    def reduce_batch(self, op: Int) raises -> Tensor[Self.dtype]:
        return self.vl.reduce_batch(op)


def sparse_cat[dt: DType](inputs: List[SparseTensor[dt]], dim: Int = 0) raises -> SparseTensor[dt]:
    if len(inputs) == 0:
        raise Error("sparse_cat: empty input")
    if dim == 0:
        var feats_list = List[Tensor[dt]]()
        var coords_list = List[IntMatrix]()
        var batch_offset = 0
        var total_batches = 0
        for i in range(len(inputs)):
            feats_list.append(inputs[i].vl.feats.copy())
            var c = inputs[i].coords.copy()
            for r in range(c.rows):
                c.set(r, 0, c.at(r, 0) + batch_offset)
            coords_list.append(c^)
            batch_offset += inputs[i].batch_size()
            total_batches += inputs[i].batch_size()
        return SparseTensor[dt](
            Tensor[dt].cat_rows(feats_list),
            IntMatrix.cat_rows(coords_list),
            total_batches,
        )
    var feats = inputs[0].vl.feats.copy()
    for i in range(1, len(inputs)):
        feats = feats.cat_dim(inputs[i].vl.feats, dim)
    return inputs[0].replace(feats^)


def sparse_unbind[dt: DType](input: SparseTensor[dt], dim: Int) raises -> List[SparseTensor[dt]]:
    var out = List[SparseTensor[dt]]()
    if dim == 0:
        for b in range(input.batch_size()):
            var one: List[Int] = [b]
            out.append(input[one])
    else:
        for f in input.vl.feats.unbind(dim):
            out.append(input.replace(f.copy()))
    return out^
