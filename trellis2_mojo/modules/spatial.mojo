# Mojo port of modules/spatial.py — only pixel_shuffle_3d: the file's
# patchify/unpatchify are dead code upstream (no callers in the repo).
# Used by the sparse-structure VAE decoder's UpsampleBlock3d.

from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def pixel_shuffle_3d(x: Tensor[F32], scale_factor: Int) raises -> Tensor[F32]:
    """[N, C, H, W, D] -> [N, C/s^3, H*s, W*s, D*s]; channel index decomposes
    as c_in = c_out * s^3 + sh * s^2 + sw * s + sd (reshape+permute upstream)."""
    if len(x.shape) != 5:
        raise Error("pixel_shuffle_3d: expected [N, C, H, W, D]")
    var n = x.shape[0]
    var c = x.shape[1]
    var h = x.shape[2]
    var w = x.shape[3]
    var d = x.shape[4]
    var s = scale_factor
    var s3 = s * s * s
    if c % s3 != 0:
        raise Error("pixel_shuffle_3d: channels not divisible by scale_factor^3")
    var co = c // s3
    var out_shape: List[Int] = [n, co, h * s, w * s, d * s]
    var out = Tensor[F32](out_shape)
    for b in range(n):
        for ci in range(co):
            for sh in range(s):
                for sw in range(s):
                    for sd in range(s):
                        var cin = ((ci * s + sh) * s + sw) * s + sd
                        for ih in range(h):
                            for iw in range(w):
                                for di in range(d):
                                    var src = ((((b * c + cin) * h + ih) * w) + iw) * d + di
                                    var dst = (
                                        (((b * co + ci) * (h * s) + ih * s + sh) * (w * s)
                                         + iw * s + sw) * (d * s) + di * s + sd
                                    )
                                    out.data[dst] = x.data[src]
    return out^
