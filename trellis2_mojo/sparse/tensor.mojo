# Minimal CPU tensor primitives for the Mojo port of trellis2.modules.sparse.
# Only what VarLenTensor/SparseTensor (basic.py) need — this is not a general
# ndarray library. Row-major, contiguous, value semantics.

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime RED_SUM = 0
comptime RED_MEAN = 1
comptime RED_PROD = 2


def _gcd(a: Int, b: Int) raises -> Int:
    var x = a if a >= 0 else -a
    var y = b if b >= 0 else -b
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


struct Frac(Copyable, Movable):
    """Exact rational number — mirrors fractions.Fraction as used for
    SparseTensor._scale. Always normalized, den > 0."""

    var num: Int
    var den: Int

    def __init__(out self, num: Int = 1, den: Int = 1) raises:
        if den == 0:
            raise Error("Frac: zero denominator")
        var n = num
        var d = den
        if d < 0:
            n = -n
            d = -d
        var g = _gcd(n, d)
        if g > 1:
            n //= g
            d //= g
        self.num = n
        self.den = d

    def __eq__(self, other: Self) raises -> Bool:
        return self.num == other.num and self.den == other.den

    def mul_int(self, k: Int) raises -> Self:
        return Frac(self.num * k, self.den)

    def div_int(self, k: Int) raises -> Self:
        return Frac(self.num, self.den * k)

    def to_string(self) raises -> String:
        return String(self.num) + "/" + String(self.den)


struct IntMatrix(Copyable, Movable):
    """Dense [rows, cols] Int32 matrix. Used for sparse coords (N x 4:
    batch, x, y, z)."""

    var data: List[Int32]
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int, fill: Int32 = 0) raises:
        self.rows = rows
        self.cols = cols
        self.data = List[Int32](length=rows * cols, fill=fill)

    def at(self, r: Int, c: Int) raises -> Int:
        return Int(self.data[r * self.cols + c])

    def set(mut self, r: Int, c: Int, v: Int) raises:
        self.data[r * self.cols + c] = Int32(v)

    def col_max(self, c: Int) raises -> Int:
        if self.rows == 0:
            raise Error("IntMatrix.col_max: empty matrix")
        var m = self.at(0, c)
        for r in range(1, self.rows):
            var v = self.at(r, c)
            if v > m:
                m = v
        return m

    def select_rows(self, idx: List[Int]) raises -> Self:
        var out = Self(len(idx), self.cols)
        for i in range(len(idx)):
            for c in range(self.cols):
                out.set(i, c, self.at(idx[i], c))
        return out^

    def slice_rows(self, start: Int, stop: Int) raises -> Self:
        var out = Self(stop - start, self.cols)
        for r in range(start, stop):
            for c in range(self.cols):
                out.set(r - start, c, self.at(r, c))
        return out^

    @staticmethod
    def cat_rows(mats: List[Self]) raises -> Self:
        if len(mats) == 0:
            raise Error("IntMatrix.cat_rows: empty input")
        var cols = mats[0].cols
        var total = 0
        for i in range(len(mats)):
            if mats[i].cols != cols:
                raise Error("IntMatrix.cat_rows: column mismatch")
            total += mats[i].rows
        var out = Self(total, cols)
        var r0 = 0
        for i in range(len(mats)):
            for r in range(mats[i].rows):
                for c in range(cols):
                    out.set(r0 + r, c, mats[i].at(r, c))
            r0 += mats[i].rows
        return out^


struct Tensor[dtype: DType](Copyable, Movable):
    """Dense row-major tensor: flat data + shape. dim 0 is the row/token dim."""

    var data: List[Scalar[Self.dtype]]
    var shape: List[Int]

    def __init__(out self, shape: List[Int], fill: Scalar[Self.dtype] = 0) raises:
        self.shape = shape.copy()
        var n = 1
        for s in shape:
            if s < 0:
                raise Error("Tensor: negative dim")
            n *= s
        self.data = List[Scalar[Self.dtype]](length=n, fill=fill)

    @staticmethod
    def from_values(shape: List[Int], values: List[Scalar[Self.dtype]]) raises -> Self:
        var t = Self(shape)
        if len(values) != len(t.data):
            raise Error("Tensor.from_values: size mismatch")
        for i in range(len(values)):
            t.data[i] = values[i]
        return t^

    def numel(self) raises -> Int:
        return len(self.data)

    def ndim(self) raises -> Int:
        return len(self.shape)

    def rows(self) raises -> Int:
        return self.shape[0]

    def row_size(self) raises -> Int:
        var n = 1
        for i in range(1, len(self.shape)):
            n *= self.shape[i]
        return n

    def tail_shape(self) raises -> List[Int]:
        var t = List[Int]()
        for i in range(1, len(self.shape)):
            t.append(self.shape[i])
        return t^

    def at(self, row: Int, offset: Int) raises -> Scalar[Self.dtype]:
        return self.data[row * self.row_size() + offset]

    def set(mut self, row: Int, offset: Int, v: Scalar[Self.dtype]) raises:
        self.data[row * self.row_size() + offset] = v

    def reshape_rows(self, tail: List[Int]) raises -> Self:
        """torch's feats.reshape(feats.shape[0], *tail): keep dim 0, reshape rest."""
        var new_shape = List[Int]()
        new_shape.append(self.rows())
        for s in tail:
            new_shape.append(s)
        var out = Self(new_shape)
        if out.numel() != self.numel():
            raise Error("Tensor.reshape_rows: size mismatch")
        _copy_span(out.data, 0, self.data, 0, len(self.data))
        return out^

    def cast[target: DType](self) raises -> Tensor[target]:
        var out = Tensor[target](self.shape)
        for i in range(len(self.data)):
            out.data[i] = self.data[i].cast[target]()
        return out^

    # -- elementwise --------------------------------------------------------

    def _binop_flat(self, other: Self, op: Int) raises -> Self:
        if self.numel() != other.numel():
            raise Error("Tensor binop: size mismatch")
        var out = Self(self.shape)
        _op_span(out.data, 0, self.data, 0, other.data, 0, len(self.data), op)
        return out^

    def _binop_scalar(self, other: Scalar[Self.dtype], op: Int, reverse: Bool = False) raises -> Self:
        var out = Self(self.shape)
        _opscalar_span(out.data, 0, self.data, 0, other, len(self.data), op, reverse)
        return out^

    def _binop_rows(self, other: Self, row_map: List[Int], op: Int) raises -> Self:
        """self[r, :] OP other[row_map[r], :] — batch-broadcast over rows.
        other's row size must equal self's or be 1 (scalar per batch)."""
        var out = Self(self.shape)
        var rs = self.row_size()
        var ors = other.row_size()
        if ors != rs and ors != 1:
            raise Error("Tensor._binop_rows: incompatible row size")
        for r in range(self.rows()):
            var src = row_map[r]
            if ors == rs:
                _op_span(out.data, r * rs, self.data, r * rs, other.data, src * rs, rs, op)
            else:
                _opscalar_span(out.data, r * rs, self.data, r * rs, other.data[src], rs, op, False)
        return out^

    def __neg__(self) raises -> Self:
        return self._binop_scalar(-1, OP_MUL)

    # -- rows ---------------------------------------------------------------

    def select_rows(self, idx: List[Int]) raises -> Self:
        var new_shape = self.shape.copy()
        new_shape[0] = len(idx)
        var out = Self(new_shape)
        var rs = self.row_size()
        for i in range(len(idx)):
            _copy_span(out.data, i * rs, self.data, idx[i] * rs, rs)
        return out^

    def slice_rows(self, start: Int, stop: Int) raises -> Self:
        var new_shape = self.shape.copy()
        new_shape[0] = stop - start
        var out = Self(new_shape)
        var rs = self.row_size()
        _copy_span(out.data, 0, self.data, start * rs, (stop - start) * rs)
        return out^

    @staticmethod
    def cat_rows(ts: List[Self]) raises -> Self:
        if len(ts) == 0:
            raise Error("Tensor.cat_rows: empty input")
        var new_shape = ts[0].shape.copy()
        var rows = 0
        var rs = ts[0].row_size()
        for i in range(len(ts)):
            if ts[i].row_size() != rs:
                raise Error("Tensor.cat_rows: row size mismatch")
            rows += ts[i].rows()
        new_shape[0] = rows
        var out = Self(new_shape)
        var k = 0
        for i in range(len(ts)):
            _copy_span(out.data, k, ts[i].data, 0, len(ts[i].data))
            k += len(ts[i].data)
        return out^

    def cat_dim(self, other: Self, dim: Int) raises -> Self:
        """Concatenate along a non-row dim. Both tensors must agree on all
        other dims."""
        if dim == 0:
            var pair = List[Self]()
            pair.append(self.copy())
            pair.append(other.copy())
            return Self.cat_rows(pair)
        var new_shape = self.shape.copy()
        new_shape[dim] += other.shape[dim]
        var out = Self(new_shape)
        var outer = 1
        for i in range(dim):
            outer *= self.shape[i]
        var inner = 1
        for i in range(dim + 1, len(self.shape)):
            inner *= self.shape[i]
        var a_blk = self.shape[dim] * inner
        var b_blk = other.shape[dim] * inner
        for o in range(outer):
            _copy_span(out.data, o * (a_blk + b_blk), self.data, o * a_blk, a_blk)
            _copy_span(out.data, o * (a_blk + b_blk) + a_blk, other.data, o * b_blk, b_blk)
        return out^

    def slice_dim(self, dim: Int, start: Int, stop: Int) raises -> Self:
        """Slice [start, stop) along dim (torch x[..., start:stop, ...])."""
        var new_shape = self.shape.copy()
        new_shape[dim] = stop - start
        var out = Self(new_shape)
        var outer = 1
        for i in range(dim):
            outer *= self.shape[i]
        var inner = 1
        for i in range(dim + 1, len(self.shape)):
            inner *= self.shape[i]
        var blk = self.shape[dim]
        var nblk = stop - start
        for o in range(outer):
            _copy_span(out.data, o * nblk * inner, self.data, (o * blk + start) * inner, nblk * inner)
        return out^

    @staticmethod
    def stack_dim1(ts: List[Self]) raises -> Self:
        """Stack n tensors [R, *tail] into [R, n, *tail] (torch.stack dim=1)."""
        if len(ts) == 0:
            raise Error("Tensor.stack_dim1: empty input")
        var rows = ts[0].rows()
        var rs = ts[0].row_size()
        var n = len(ts)
        var new_shape: List[Int] = [rows, n]
        for d in ts[0].tail_shape():
            new_shape.append(d)
        var out = Self(new_shape)
        for i in range(n):
            if ts[i].rows() != rows or ts[i].row_size() != rs:
                raise Error("Tensor.stack_dim1: shape mismatch")
            for r in range(rows):
                _copy_span(out.data, (r * n + i) * rs, ts[i].data, r * rs, rs)
        return out^

    def flatten_leading(self, k: Int) raises -> Self:
        """Merge the first k dims: [d0..dk-1, *rest] -> [d0*..*dk-1, *rest]."""
        var lead = 1
        for i in range(k):
            lead *= self.shape[i]
        var new_shape: List[Int] = [lead]
        for i in range(k, len(self.shape)):
            new_shape.append(self.shape[i])
        var out = Self(new_shape)
        _copy_span(out.data, 0, self.data, 0, len(self.data))
        return out^

    def unbind(self, dim: Int) raises -> List[Self]:
        """Split along dim (>= 1), removing it — torch.unbind."""
        if dim < 1 or dim >= len(self.shape):
            raise Error("Tensor.unbind: dim out of range (must be >= 1)")
        var n = self.shape[dim]
        var new_shape = List[Int]()
        for i in range(len(self.shape)):
            if i != dim:
                new_shape.append(self.shape[i])
        var outer = 1
        for i in range(dim):
            outer *= self.shape[i]
        var inner = 1
        for i in range(dim + 1, len(self.shape)):
            inner *= self.shape[i]
        var out = List[Self]()
        for k in range(n):
            var t = Self(new_shape)
            for o in range(outer):
                _copy_span(t.data, o * inner, self.data, (o * n + k) * inner, inner)
            out.append(t^)
        return out^

    # -- reductions ---------------------------------------------------------

    def reduce_all(self, op: Int) raises -> Scalar[Self.dtype]:
        var acc: Scalar[Self.dtype] = 1 if op == RED_PROD else 0
        for i in range(len(self.data)):
            if op == RED_PROD:
                acc *= self.data[i]
            else:
                acc += self.data[i]
        if op == RED_MEAN:
            acc /= Scalar[Self.dtype](len(self.data))
        return acc

    def reduce_tail(self, op: Int) raises -> Self:
        """Reduce over all dims except dim 0 -> shape [rows]."""
        var out = Self([self.rows()])
        var rs = self.row_size()
        for r in range(self.rows()):
            var acc: Scalar[Self.dtype] = 1 if op == RED_PROD else 0
            for j in range(rs):
                if op == RED_PROD:
                    acc *= self.data[r * rs + j]
                else:
                    acc += self.data[r * rs + j]
            if op == RED_MEAN:
                acc /= Scalar[Self.dtype](rs)
            out.data[r] = acc
        return out^

    def segment_reduce(self, offsets: List[Int], op: Int) raises -> Self:
        """Reduce rows within [offsets[b], offsets[b+1]) -> [B, *tail]."""
        var b = len(offsets) - 1
        var new_shape = self.shape.copy()
        new_shape[0] = b
        var out = Self(new_shape, Scalar[Self.dtype](1) if op == RED_PROD else Scalar[Self.dtype](0))
        var rs = self.row_size()
        for seg in range(b):
            var n = offsets[seg + 1] - offsets[seg]
            for r in range(offsets[seg], offsets[seg + 1]):
                for j in range(rs):
                    if op == RED_PROD:
                        out.data[seg * rs + j] *= self.data[r * rs + j]
                    else:
                        out.data[seg * rs + j] += self.data[r * rs + j]
            if op == RED_MEAN and n > 0:
                for j in range(rs):
                    out.data[seg * rs + j] /= Scalar[Self.dtype](n)
        return out^


# SIMD span helpers (WP10 pass 5): the copy/elementwise primitives below are
# pure value-preserving memory ops, so vectorizing them is bit-identical by
# construction. The op branch is hoisted out of the loops.
comptime _EW = 8


def _copy_span[dt: DType](mut dst: List[Scalar[dt]], d0: Int, src: List[Scalar[dt]], s0: Int, n: Int) raises:
    """dst[d0:d0+n] = src[s0:s0+n]."""
    var dp = dst.unsafe_ptr()
    var sp = src.unsafe_ptr()
    var i = 0
    while i + _EW <= n:
        dp.store(d0 + i, sp.load[width=_EW](s0 + i))
        i += _EW
    while i < n:
        dp[d0 + i] = sp[s0 + i]
        i += 1


def _op_span[dt: DType](
    mut dst: List[Scalar[dt]], d0: Int,
    a: List[Scalar[dt]], a0: Int,
    b: List[Scalar[dt]], b0: Int,
    n: Int, op: Int,
) raises:
    """dst[d0+i] = a[a0+i] OP b[b0+i]."""
    var dp = dst.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var i = 0
    if op == OP_ADD:
        while i + _EW <= n:
            dp.store(d0 + i, ap.load[width=_EW](a0 + i) + bp.load[width=_EW](b0 + i))
            i += _EW
        while i < n:
            dp[d0 + i] = ap[a0 + i] + bp[b0 + i]
            i += 1
    elif op == OP_SUB:
        while i + _EW <= n:
            dp.store(d0 + i, ap.load[width=_EW](a0 + i) - bp.load[width=_EW](b0 + i))
            i += _EW
        while i < n:
            dp[d0 + i] = ap[a0 + i] - bp[b0 + i]
            i += 1
    elif op == OP_MUL:
        while i + _EW <= n:
            dp.store(d0 + i, ap.load[width=_EW](a0 + i) * bp.load[width=_EW](b0 + i))
            i += _EW
        while i < n:
            dp[d0 + i] = ap[a0 + i] * bp[b0 + i]
            i += 1
    elif op == OP_DIV:
        while i + _EW <= n:
            dp.store(d0 + i, ap.load[width=_EW](a0 + i) / bp.load[width=_EW](b0 + i))
            i += _EW
        while i < n:
            dp[d0 + i] = ap[a0 + i] / bp[b0 + i]
            i += 1
    else:
        raise Error("unknown op code")


def _opscalar_span[dt: DType](
    mut dst: List[Scalar[dt]], d0: Int,
    a: List[Scalar[dt]], a0: Int,
    b: Scalar[dt], n: Int, op: Int, reverse: Bool,
) raises:
    """dst[d0+i] = a[a0+i] OP b (or b OP a[a0+i] when reverse)."""
    var dp = dst.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bv = SIMD[dt, _EW](b)
    var i = 0
    if op == OP_ADD:
        while i + _EW <= n:
            dp.store(d0 + i, ap.load[width=_EW](a0 + i) + bv)
            i += _EW
        while i < n:
            dp[d0 + i] = ap[a0 + i] + b
            i += 1
    elif op == OP_MUL:
        while i + _EW <= n:
            dp.store(d0 + i, ap.load[width=_EW](a0 + i) * bv)
            i += _EW
        while i < n:
            dp[d0 + i] = ap[a0 + i] * b
            i += 1
    elif op == OP_SUB:
        if reverse:
            while i + _EW <= n:
                dp.store(d0 + i, bv - ap.load[width=_EW](a0 + i))
                i += _EW
            while i < n:
                dp[d0 + i] = b - ap[a0 + i]
                i += 1
        else:
            while i + _EW <= n:
                dp.store(d0 + i, ap.load[width=_EW](a0 + i) - bv)
                i += _EW
            while i < n:
                dp[d0 + i] = ap[a0 + i] - b
                i += 1
    elif op == OP_DIV:
        if reverse:
            while i + _EW <= n:
                dp.store(d0 + i, bv / ap.load[width=_EW](a0 + i))
                i += _EW
            while i < n:
                dp[d0 + i] = b / ap[a0 + i]
                i += 1
        else:
            while i + _EW <= n:
                dp.store(d0 + i, ap.load[width=_EW](a0 + i) / bv)
                i += _EW
            while i < n:
                dp[d0 + i] = ap[a0 + i] / b
                i += 1
    else:
        raise Error("unknown op code")


def _apply_op[dt: DType](a: Scalar[dt], b: Scalar[dt], op: Int) raises -> Scalar[dt]:
    if op == OP_ADD:
        return a + b
    if op == OP_SUB:
        return a - b
    if op == OP_MUL:
        return a * b
    if op == OP_DIV:
        return a / b
    raise Error("unknown op code")


def stable_argsort(keys: List[Int]) raises -> List[Int]:
    """Indices that sort keys ascending, ties in original order (merge sort)."""
    var n = len(keys)
    var idx = List[Int]()
    for i in range(n):
        idx.append(i)
    var buf = List[Int](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            var hi = lo + 2 * width
            if mid > n:
                mid = n
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[idx[i]] <= keys[idx[j]]:
                    buf[k] = idx[i]
                    i += 1
                else:
                    buf[k] = idx[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
            for x in range(lo, hi):
                idx[x] = buf[x]
            lo += 2 * width
        width *= 2
    return idx^
