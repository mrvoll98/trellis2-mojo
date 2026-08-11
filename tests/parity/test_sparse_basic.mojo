# WP3 unit tests: VarLenTensor/SparseTensor semantics against hand-computed
# expectations.
#
# Run: pixi run mojo run -I . tests/parity/test_sparse_basic.mojo

from std.testing import assert_equal, assert_true, assert_almost_equal

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix, Frac, OP_ADD, OP_MUL, RED_SUM, RED_MEAN
from trellis2_mojo.sparse.basic import (
    VarLenTensor,
    SparseTensor,
    CacheValue,
    varlen_cat,
    varlen_unbind,
    sparse_cat,
    sparse_unbind,
)

comptime F32 = DType.float32


def make_varlen() raises -> VarLenTensor[F32]:
    # batch 0: 2 rows, batch 1: 3 rows; C = 2
    # feats = [[1,2],[3,4],[5,6],[7,8],[9,10]]
    var feats = Tensor[F32].from_values([5, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return VarLenTensor[F32](feats^, [0, 2, 5])


def make_sparse() raises -> SparseTensor[F32]:
    # 2 batches, coords within a 2x2x2 grid
    var feats = Tensor[F32].from_values([5, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    var coords = IntMatrix(5, 4)
    # (b, x, y, z)
    var vals: List[List[Int]] = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ]
    for r in range(5):
        for c in range(4):
            coords.set(r, c, vals[r][c])
    return SparseTensor[F32](feats^, coords^)


def test_varlen_shape_and_layout() raises:
    var x = make_varlen()
    assert_equal(x.batch_size(), 2)
    var s = x.shape()
    assert_equal(s[0], 2)
    assert_equal(s[1], 2)
    assert_equal(x.seqlen(0), 2)
    assert_equal(x.seqlen(1), 3)
    var bm = x.batch_broadcast_map()
    assert_equal(bm[0], 0)
    assert_equal(bm[1], 0)
    assert_equal(bm[2], 1)
    assert_equal(bm[4], 1)


def test_varlen_roundtrip() raises:
    var x = make_varlen()
    var parts = x.to_tensor_list()
    assert_equal(len(parts), 2)
    assert_equal(parts[1].rows(), 3)
    assert_equal(parts[1].data[0], 5.0)
    var y = VarLenTensor[F32].from_tensor_list(parts)
    assert_equal(y.feats.data[4], x.feats.data[4])
    assert_equal(y.offsets[2], 5)


def test_varlen_elemwise() raises:
    var x = make_varlen()
    var y = (x + 1.0) * 2.0
    assert_equal(y.feats.data[0], 4.0)   # (1+1)*2
    assert_equal(y.feats.data[9], 22.0)  # (10+1)*2
    var z = x + x
    assert_equal(z.feats.data[3], 8.0)
    var n = -x
    assert_equal(n.feats.data[0], -1.0)
    # batch broadcast: per-batch scalar [B, 1] -> rows of that batch
    var b = Tensor[F32].from_values([2, 1], [10, 100])
    var w = x.elemwise_batch(b, OP_ADD)
    assert_equal(w.feats.data[0], 11.0)   # batch 0 row
    assert_equal(w.feats.data[4], 105.0)  # batch 1 row (feats[2,0]=5)
    # batch broadcast full tail [B, C]
    var b2 = Tensor[F32].from_values([2, 2], [10, 20, 100, 200])
    var w2 = x.elemwise_batch(b2, OP_ADD)
    assert_equal(w2.feats.data[1], 22.0)  # 2 + 20
    assert_equal(w2.feats.data[9], 210.0)  # 10 + 200


def test_varlen_getitem_and_unbind() raises:
    var x = make_varlen()
    var y = x[[1, 0]]  # swap batches
    assert_equal(y.seqlen(0), 3)
    assert_equal(y.feats.data[0], 5.0)
    assert_equal(y.feats.data[6], 1.0)
    var parts = varlen_unbind(x, 0)
    assert_equal(len(parts), 2)
    assert_equal(parts[1].batch_size(), 1)
    assert_equal(parts[1].feats.data[0], 5.0)
    # unbind along feature dim
    var cols = x.unbind(1)
    assert_equal(len(cols), 2)
    assert_equal(cols[1].feats.data[0], 2.0)
    assert_equal(cols[1].seqlen(1), 3)


def test_varlen_cat() raises:
    var x = make_varlen()
    var y = make_varlen()
    var z = varlen_cat([x.copy(), y.copy()], 0)
    assert_equal(z.batch_size(), 4)
    assert_equal(z.seqlen(2), 2)
    assert_equal(z.feats.data[10], 1.0)
    var f = varlen_cat([x^, y^], 1)  # feature concat
    assert_equal(f.batch_size(), 2)
    var s = f.shape()
    assert_equal(s[1], 4)
    assert_equal(f.feats.at(0, 2), 1.0)


def test_varlen_to_dense() raises:
    var x = make_varlen()
    var dm = x.to_dense()
    var dense = dm[0].copy()
    var mask = dm[1].copy()
    # dense: [2, 3, 2]; batch 0 padded at l=2
    assert_equal(dense.shape[1], 3)
    assert_equal(dense.data[0], 1.0)      # [0,0,0]
    assert_equal(dense.data[4], 0.0)      # [0,2,0] padding
    assert_equal(dense.data[6], 5.0)      # [1,0,0]
    assert_equal(mask.data[2], 0)         # [0,2] invalid
    assert_equal(mask.data[5], 1)         # [1,2] valid


def test_varlen_reduce() raises:
    var x = make_varlen()
    assert_almost_equal(x.reduce_all(RED_SUM), 55.0)
    var m = x.reduce_batch(RED_MEAN)
    # batch 0: mean(1,2,3,4) = 2.5 -> segment mean over 2 rows: (1.5+3.5)/2
    assert_almost_equal(m.data[0], 2.5)
    # batch 1: mean of rows (5.5, 7.5, 9.5) = 7.5
    assert_almost_equal(m.data[1], 7.5)


def test_sparse_layout_from_coords() raises:
    var x = make_sparse()
    assert_equal(x.batch_size(), 2)
    assert_equal(x.seqlen(0), 2)
    assert_equal(x.seqlen(1), 3)
    var sp = x.spatial_shape()
    assert_equal(len(sp), 3)
    assert_equal(sp[0], 2)
    assert_equal(sp[1], 2)
    assert_equal(sp[2], 2)


def test_sparse_contiguity_enforced() raises:
    var feats = Tensor[F32]([2, 1], 0)
    var coords = IntMatrix(2, 4)
    coords.set(0, 0, 1)  # batch 1 before batch 0 -> must raise
    coords.set(1, 0, 0)
    var raised = False
    try:
        var _bad = SparseTensor[F32](feats^, coords^)
    except:
        raised = True
    assert_true(raised)


def test_sparse_getitem_renumbers_batches() raises:
    var x = make_sparse()
    var y = x[[1]]
    assert_equal(y.batch_size(), 1)
    assert_equal(y.coords.at(0, 0), 0)  # renumbered from 1 -> 0
    assert_equal(y.vl.feats.data[0], 5.0)
    var z = x[[1, 0]]
    assert_equal(z.coords.at(0, 0), 0)
    assert_equal(z.coords.at(3, 0), 1)
    assert_equal(z.vl.feats.data[6], 1.0)


def test_sparse_cat_rebases_batch_column() raises:
    var x = make_sparse()
    var y = make_sparse()
    var z = sparse_cat([x.copy(), y.copy()], 0)
    assert_equal(z.batch_size(), 4)
    assert_equal(z.coords.at(5, 0), 2)  # first row of y rebased 0 -> 2
    assert_equal(z.seqlen(3), 3)
    var f = sparse_cat([x^, y^], 1)
    var s = f.shape()
    assert_equal(s[1], 4)
    assert_equal(f.batch_size(), 2)


def test_sparse_unbind_dim0() raises:
    var x = make_sparse()
    var parts = sparse_unbind(x, 0)
    assert_equal(len(parts), 2)
    assert_equal(parts[1].coords.at(0, 0), 0)
    assert_equal(parts[1].vl.feats.data[0], 5.0)


def test_sparse_to_dense() raises:
    var x = make_sparse()
    var d = x.to_dense()
    # shape [2, 2, 2, 2, 2]: dense[b, c, x, y, z]
    assert_equal(len(d.shape), 5)
    # row 0: b0 (0,0,0) feats (1,2) -> d[0,0,0,0,0]=1, d[0,1,0,0,0]=2
    assert_equal(d.data[0], 1.0)
    assert_equal(d.data[8], 2.0)
    # row 4: b1 (1,0,0) feats (9,10) -> d[1,0,1,0,0]=9
    # flat: ((1*2+0)*8) + (1*4+0*2+0) = 16+4 = 20
    assert_equal(d.data[20], 9.0)
    # empty voxel stays zero: d[0,0,1,1,1] -> 0*16+0*8+7 = 7
    assert_equal(d.data[7], 0.0)


def test_sparse_full() raises:
    var x = SparseTensor[F32].full([0, 0, 0, 1, 1, 1], 2, 3, 0.5)
    assert_equal(x.batch_size(), 2)
    assert_equal(x.coords.rows, 16)
    assert_equal(x.seqlen(0), 8)
    assert_equal(x.vl.feats.data[0], 0.5)
    var sp = x.spatial_shape()
    assert_equal(sp[2], 2)


def test_sparse_replace_shares_cache() raises:
    var x = make_sparse()
    var sp: List[Int] = [4, 4, 4]
    x.register_spatial_cache("test_entry", CacheValue.from_shape(sp))
    var y = x.replace(x.vl.feats._binop_scalar(1.0, OP_ADD))
    var hit = y.get_spatial_cache("test_entry")
    assert_true(Bool(hit))
    assert_equal(hit.value().shape[0], 4)
    # and writes on the derived tensor are visible on the parent
    y.register_spatial_cache("from_child", CacheValue.from_shape(sp))
    assert_true(Bool(x.get_spatial_cache("from_child")))
    # getitem must NOT share cache
    var z = x[[0]]
    assert_true(not z.get_spatial_cache("test_entry"))


def test_sparse_cache_is_scale_keyed() raises:
    var x = make_sparse()
    var sp: List[Int] = [4, 4, 4]
    x.register_spatial_cache("k", CacheValue.from_shape(sp))
    # same tensor at another scale: entry invisible
    var y = x.with_scale([Frac(2, 1), Frac(2, 1), Frac(2, 1)])
    assert_true(not y.get_spatial_cache("k"))
    assert_true(Bool(x.get_spatial_cache("k")))
    # scale arithmetic stays exact: (1/2)*2 == 1
    var f = Frac(1, 2).mul_int(2)
    assert_true(f == Frac(1, 1))


def test_sparse_elemwise_batch() raises:
    var x = make_sparse()
    var b = Tensor[F32].from_values([2, 2], [10, 20, 100, 200])
    var y = x.elemwise_batch(b, OP_ADD)
    assert_equal(y.vl.feats.data[0], 11.0)
    assert_equal(y.vl.feats.data[9], 210.0)
    var m = x.elemwise_batch(b, OP_MUL)
    assert_equal(m.vl.feats.data[1], 40.0)


def main() raises:
    test_varlen_shape_and_layout()
    test_varlen_roundtrip()
    test_varlen_elemwise()
    test_varlen_getitem_and_unbind()
    test_varlen_cat()
    test_varlen_to_dense()
    test_varlen_reduce()
    test_sparse_layout_from_coords()
    test_sparse_contiguity_enforced()
    test_sparse_getitem_renumbers_batches()
    test_sparse_cat_rebases_batch_column()
    test_sparse_unbind_dim0()
    test_sparse_to_dense()
    test_sparse_full()
    test_sparse_replace_shares_cache()
    test_sparse_cache_is_scale_keyed()
    test_sparse_elemwise_batch()
    print("all sparse basic tests passed (17/17)")
