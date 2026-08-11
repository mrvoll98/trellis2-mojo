# Mojo port of models/sc_vaes/fdg_vae.py — the FlexiDualGrid head.
#
# FlexiDualGridVaeDecoder is SparseUnetVaeDecoder (same weights/keys, in
# channels 7) plus this feature-transform head; the actual mesh extraction
# (flexible_dual_grid_to_mesh, o-voxel FFI / Python stub) stays in Python
# per ADR 0001/0004 and is wired up in WP9. Eval path only — the training
# branch (gt_intersected, logits) is out of scope.

from std.math import exp, log

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.basic import SparseTensor

comptime F32 = DType.float32


def fdg_head(
    h: SparseTensor[F32], voxel_margin: Float64 = 0.5
) raises -> Tuple[SparseTensor[F32], SparseTensor[F32], SparseTensor[F32]]:
    """Decoded feats [T, 7] -> (vertices [T, 3], intersected [T, 3] as
    1.0/0.0, quad_lerp [T, 1]):
      vertices    = (1 + 2*margin) * sigmoid(feats[:, 0:3]) - margin
      intersected = feats[:, 3:6] > 0
      quad_lerp   = softplus(feats[:, 6:7])
    """
    var f = h.vl.feats.copy()
    if f.shape[1] != 7:
        raise Error("fdg_head: expected 7 channels")
    var t = f.shape[0]
    var m = Float32(voxel_margin)
    var v_shape: List[Int] = [t, 3]
    var i_shape: List[Int] = [t, 3]
    var q_shape: List[Int] = [t, 1]
    var vertices = Tensor[F32](v_shape)
    var intersected = Tensor[F32](i_shape)
    var quad = Tensor[F32](q_shape)
    for r in range(t):
        for c in range(3):
            var x = f.data[r * 7 + c]
            vertices.data[r * 3 + c] = (1.0 + 2.0 * m) / (1.0 + exp(-x)) - m
        for c in range(3):
            if f.data[r * 7 + 3 + c] > 0:
                intersected.data[r * 3 + c] = 1.0
        # Numerically stable softplus: x if x > 20 else log(1 + exp(x))
        var q = f.data[r * 7 + 6]
        if q > 20:
            quad.data[r] = q
        else:
            quad.data[r] = log(1.0 + exp(q))
    return (h.replace(vertices^), h.replace(intersected^), h.replace(quad^))
