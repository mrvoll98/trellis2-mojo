# Mojo port of modules/sparse/attention/windowed_attn.py.
#
# calc_window_partition ravels each voxel's (batch, window) into one index,
# stable-sorts rows by it, and attention runs block-diagonally over the
# window segments. The partition is cached in the spatial cache like the
# original.
#
# Note: the original only implements xformers/flash_attn backends here (no
# sdpa/naive branch — it cannot run on CPU at all); this port IS the naive
# backend. The windowed cross-attention variant is unused by the models and
# not ported.

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix, stable_argsort
from trellis2_mojo.sparse.basic import SparseTensor, CacheValue
from trellis2_mojo.sparse.attention.full_attn import varlen_sdpa

comptime F32 = DType.float32




def calc_window_partition(
    x: SparseTensor[F32], window_size: List[Int], shift_window: List[Int]
) raises -> Tuple[List[Int], List[Int], List[Int]]:
    """-> (fwd_indices, bwd_indices, seq_lens). Row r of the partitioned
    order is original row fwd_indices[r]; bwd_indices inverts it."""
    var dim = x.coords.cols - 1
    var sp = x.spatial_shape()
    var num_windows = List[Int]()
    for i in range(dim):
        var mc = sp[i] + shift_window[i]
        num_windows.append((mc + 1 + window_size[i] - 1) // window_size[i])
    # index = ravel(batch, wx, wy, wz), batch-major, last axis fastest
    var keys = List[Int]()
    for r in range(x.coords.rows):
        var key = x.coords.at(r, 0)
        for i in range(dim):
            var w = (x.coords.at(r, i + 1) + shift_window[i]) // window_size[i]
            key = key * num_windows[i] + w
        keys.append(key)
    var fwd = stable_argsort(keys)
    var n = len(fwd)
    var bwd = List[Int](length=n, fill=0)
    for r in range(n):
        bwd[fwd[r]] = r
    var seq_lens = List[Int]()
    var run = 0
    for r in range(n):
        if r > 0 and keys[fwd[r]] != keys[fwd[r - 1]]:
            seq_lens.append(run)
            run = 0
        run += 1
    if n > 0:
        seq_lens.append(run)
    return (fwd^, bwd^, seq_lens^)


def _partition_cached(
    x: SparseTensor[F32], window_size: Int, shift_window: List[Int]
) raises -> Tuple[List[Int], List[Int], List[Int]]:
    var name = (
        "windowed_attention_" + String(window_size) + "_"
        + String(shift_window[0]) + "," + String(shift_window[1]) + "," + String(shift_window[2])
    )
    var cached = x.get_spatial_cache(name)
    if cached:
        var v = cached.value().copy()
        return (v.ints[0].copy(), v.ints[1].copy(), v.ints[2].copy())
    var ws: List[Int] = [window_size, window_size, window_size]
    var part = calc_window_partition(x, ws, shift_window)
    var cv = CacheValue.from_ints([part[0].copy(), part[1].copy(), part[2].copy()])
    x.register_spatial_cache(name, cv^)
    return part^


def sparse_windowed_sdpa_self(
    qkv: SparseTensor[F32], window_size: Int, shift_window: List[Int]
) raises -> SparseTensor[F32]:
    """Windowed self-attention on packed qkv feats [T, 3, H, C]."""
    var part = _partition_cached(qkv, window_size, shift_window)
    var fwd = part[0].copy()
    var bwd = part[1].copy()
    var seq_lens = part[2].copy()

    var gathered = qkv.vl.feats.select_rows(fwd)  # [M, 3, H, C] in window order
    var parts = gathered.unbind(1)
    var offsets: List[Int] = [0]
    var start = 0
    for l in seq_lens:
        start += l
        offsets.append(start)
    var attn = varlen_sdpa(parts[0], parts[1], parts[2], offsets, offsets)
    return qkv.replace(attn.select_rows(bwd))
