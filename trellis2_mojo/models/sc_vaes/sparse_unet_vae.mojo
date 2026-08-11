# Mojo port of models/sc_vaes/sparse_unet_vae.py — decoder side only.
#
# The real checkpoints (configs/scvae/*.json) build every stage from
# SparseConvNeXtBlock3d with SparseResBlockC2S3d as the up block, so those
# are the two block types ported. Not ported: the encoder (+ Downsample/S2C
# blocks — training/encode side), SparseResBlock3d and
# SparseResBlockUpsample3d (never referenced by any config), use_checkpoint
# and fp16 conversion (f32 v1), and the training branches of forward.
#
# The shape decoder runs pred_subdiv=True: each up block predicts an [T, 8]
# subdivision from its input (to_subdiv), binarizes it (> 0) and unfolds
# only the positive children; the predicted subs are returned and later
# guide the texture decoder (pred_subdiv=False, forward_guided), exactly
# like the pipeline's decode_shape_slat -> decode_tex_slat handoff.
# Calling a pred_subdiv=False up block WITHOUT guide subs (the source allows
# subdiv=None) is not supported — the pipelines always pass guide_subs.

from trellis2_mojo.io.state_dict import StateDict

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.conv import SparseConv3d
from trellis2_mojo.sparse.spatial.spatial2channel import SparseChannel2Spatial
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, activation, ACT_SILU
from trellis2_mojo.loaders import lin_from, ln_from, sparse_conv3d_from, dummy_lin

comptime F32 = DType.float32


def _binarize(subdiv: SparseTensor[F32]) raises -> SparseTensor[F32]:
    """subdiv.feats > 0 as 1.0/0.0 before the C2S unfold."""
    var f = subdiv.vl.feats.copy()
    var out = Tensor[F32](f.shape)
    for i in range(f.numel()):
        if f.data[i] > 0:
            out.data[i] = 1.0
    return subdiv.replace(out^)


struct SparseConvNeXtBlock3d(Copyable, Movable):
    """conv3 -> LayerNorm(affine, 1e-6) -> Linear/SiLU/Linear -> + x."""

    var conv: SparseConv3d
    var norm: LayerNorm32
    var lin1: SparseLinear
    var lin2: SparseLinear

    def __init__(
        out self,
        var conv: SparseConv3d,
        var norm: LayerNorm32,
        var lin1: SparseLinear,
        var lin2: SparseLinear,
    ):
        self.conv = conv^
        self.norm = norm^
        self.lin1 = lin1^
        self.lin2 = lin2^

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var h = self.conv.forward(x)
        h = h.replace(self.norm.forward(h.vl.feats))
        h = h.replace(self.lin2.forward(activation(self.lin1.forward(h.vl.feats), ACT_SILU)))
        return h + x


struct SparseResBlockC2S3d(Copyable, Movable):
    """Upsample block: norm1/SiLU -> conv1 (C -> C_out*8) -> Channel2Spatial
    guided by the (predicted or given) subdivision -> norm2/SiLU -> conv2,
    plus a repeat_interleave skip on the C2S of x itself."""

    var channels: Int
    var out_channels: Int
    var pred_subdiv: Bool
    var norm1: LayerNorm32  # affine, eps 1e-6
    var norm2: LayerNorm32  # no affine, eps 1e-6
    var conv1: SparseConv3d
    var conv2: SparseConv3d
    var to_subdiv: SparseLinear  # C -> 8, only when pred_subdiv
    var updown: SparseChannel2Spatial

    def __init__(
        out self,
        channels: Int,
        out_channels: Int,
        pred_subdiv: Bool,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var conv1: SparseConv3d,
        var conv2: SparseConv3d,
        var to_subdiv: SparseLinear,
    ):
        self.channels = channels
        self.out_channels = out_channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.conv1 = conv1^
        self.conv2 = conv2^
        self.to_subdiv = to_subdiv^
        self.updown = SparseChannel2Spatial(2)

    def _skip(self, x2: SparseTensor[F32]) raises -> SparseTensor[F32]:
        """x after C2S has channels/8 features; repeat_interleave each
        channel out_channels/(channels/8) times to reach out_channels."""
        var cin = x2.vl.feats.shape[1]
        var k = self.out_channels // cin
        var m = x2.vl.feats.shape[0]
        var shape: List[Int] = [m, self.out_channels]
        var out = Tensor[F32](shape)
        for r in range(m):
            for c in range(cin):
                for j in range(k):
                    out.data[r * self.out_channels + c * k + j] = x2.vl.feats.data[r * cin + c]
        return x2.replace(out^)

    def _core(self, x: SparseTensor[F32], subdiv: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var h = x.replace(activation(self.norm1.forward(x.vl.feats), ACT_SILU))
        h = self.conv1.forward(h)
        var mask = _binarize(subdiv)
        h = self.updown.forward_subdivision(h, mask)
        var x2 = self.updown.forward_subdivision(x, mask)
        h = h.replace(activation(self.norm2.forward(h.vl.feats), ACT_SILU))
        h = self.conv2.forward(h)
        return h + self._skip(x2)

    def forward(self, x: SparseTensor[F32]) raises -> Tuple[SparseTensor[F32], SparseTensor[F32]]:
        """pred_subdiv path: predicts the subdivision, returns (h, subdiv)."""
        if not self.pred_subdiv:
            raise Error("SparseResBlockC2S3d: use forward_guided when pred_subdiv=False")
        var subdiv = self.to_subdiv.forward(x)
        var h = self._core(x, subdiv)
        return (h^, subdiv^)

    def forward_guided(self, x: SparseTensor[F32], subdiv: SparseTensor[F32]) raises -> SparseTensor[F32]:
        """guide_subs path (pred_subdiv=False): subdivision comes from the
        shape decoder's predictions."""
        return self._core(x, subdiv)


struct SparseUnetVaeDecoder(Copyable, Movable):
    var pred_subdiv: Bool
    var num_blocks: List[Int]  # ConvNeXt blocks per stage
    var from_latent: SparseLinear
    var convnext: List[SparseConvNeXtBlock3d]   # flattened per stage
    var ups: List[SparseResBlockC2S3d]          # one per stage except the last
    var final_norm: LayerNorm32                  # F.layer_norm: eps 1e-5, no affine
    var output_layer: SparseLinear

    def __init__(
        out self,
        pred_subdiv: Bool,
        var num_blocks: List[Int],
        var from_latent: SparseLinear,
        var convnext: List[SparseConvNeXtBlock3d],
        var ups: List[SparseResBlockC2S3d],
        model_channels_last: Int,
        var output_layer: SparseLinear,
    ) raises:
        self.pred_subdiv = pred_subdiv
        self.num_blocks = num_blocks^
        self.from_latent = from_latent^
        self.convnext = convnext^
        self.ups = ups^
        self.final_norm = LayerNorm32(model_channels_last, 1e-5, False)
        self.output_layer = output_layer^

    def _finish(self, h: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var y = h.replace(self.final_norm.forward(h.vl.feats))
        return self.output_layer.forward(y)

    def forward(self, x: SparseTensor[F32]) raises -> Tuple[SparseTensor[F32], List[SparseTensor[F32]]]:
        """pred_subdiv decoder: -> (decoded, predicted subs per upsample) —
        equivalent to forward(x, return_subs=True) in the source model."""
        if not self.pred_subdiv:
            raise Error("SparseUnetVaeDecoder: use forward_guided when pred_subdiv=False")
        var h = self.from_latent.forward(x)
        var subs = List[SparseTensor[F32]]()
        var ci = 0
        var n_stages = len(self.num_blocks)
        for stage in range(n_stages):
            for _ in range(self.num_blocks[stage]):
                h = self.convnext[ci].forward(h)
                ci += 1
            if stage < n_stages - 1:
                var pair = self.ups[stage].forward(h)
                h = pair[0].copy()
                subs.append(pair[1].copy())
        return (self._finish(h), subs^)

    def forward_guided(self, x: SparseTensor[F32], guide_subs: List[SparseTensor[F32]]) raises -> SparseTensor[F32]:
        """pred_subdiv=False decoder driven by the shape decoder's subs —
        equivalent to forward(x, guide_subs=...) in the source model."""
        if self.pred_subdiv:
            raise Error("SparseUnetVaeDecoder: use forward when pred_subdiv=True")
        var h = self.from_latent.forward(x)
        var ci = 0
        var n_stages = len(self.num_blocks)
        for stage in range(n_stages):
            for _ in range(self.num_blocks[stage]):
                h = self.convnext[ci].forward(h)
                ci += 1
            if stage < n_stages - 1:
                h = self.ups[stage].forward_guided(h, guide_subs[stage])
        return self._finish(h)

    def upsample_coords(self, x: SparseTensor[F32], upsample_times: Int) raises -> IntMatrix:
        """Run stages until `upsample_times` resolutions
        are done and return the coords there (pred_subdiv only)."""
        if not self.pred_subdiv:
            raise Error("SparseUnetVaeDecoder: upsample_coords needs pred_subdiv=True")
        var h = self.from_latent.forward(x)
        var ci = 0
        var n_stages = len(self.num_blocks)
        for stage in range(n_stages):
            if stage == upsample_times:
                return h.coords.copy()
            for _ in range(self.num_blocks[stage]):
                h = self.convnext[ci].forward(h)
                ci += 1
            if stage < n_stages - 1:
                var pair = self.ups[stage].forward(h)
                h = pair[0].copy()
        return h.coords.copy()


def convnext_from(sd: StateDict, prefix: String, channels: Int) raises -> SparseConvNeXtBlock3d:
    return SparseConvNeXtBlock3d(
        sparse_conv3d_from(sd, prefix + ".conv"),
        ln_from(sd, prefix + ".norm", channels, affine=True),
        lin_from(sd, prefix + ".mlp.0"),
        lin_from(sd, prefix + ".mlp.2"),
    )


def c2s_block_from(
    sd: StateDict, prefix: String, channels: Int, out_channels: Int, pred_subdiv: Bool
) raises -> SparseResBlockC2S3d:
    var to_subdiv: SparseLinear
    if pred_subdiv:
        to_subdiv = lin_from(sd, prefix + ".to_subdiv")
    else:
        to_subdiv = dummy_lin()
    return SparseResBlockC2S3d(
        channels, out_channels, pred_subdiv,
        ln_from(sd, prefix + ".norm1", channels, affine=True),
        ln_from(sd, prefix + ".norm2", out_channels),
        sparse_conv3d_from(sd, prefix + ".conv1"),
        sparse_conv3d_from(sd, prefix + ".conv2"),
        to_subdiv^,
    )


def sparse_unet_vae_decoder_from(
    sd: StateDict,
    model_channels: List[Int],
    num_blocks: List[Int],
    pred_subdiv: Bool,
) raises -> SparseUnetVaeDecoder:
    """Build the decoder from a native StateDict. Assumes the config shape
    every checkpoint uses: ConvNeXt stages with C2S up blocks."""
    var convnext = List[SparseConvNeXtBlock3d]()
    var ups = List[SparseResBlockC2S3d]()
    var n_stages = len(num_blocks)
    for stage in range(n_stages):
        var p = "blocks." + String(stage) + "."
        for j in range(num_blocks[stage]):
            var pj = p + String(j)
            convnext.append(convnext_from(sd, pj, model_channels[stage]))
        if stage < n_stages - 1:
            var pn = p + String(num_blocks[stage])
            ups.append(c2s_block_from(
                sd, pn,
                model_channels[stage], model_channels[stage + 1], pred_subdiv,
            ))
    return SparseUnetVaeDecoder(
        pred_subdiv,
        num_blocks.copy(),
        lin_from(sd, "from_latent"),
        convnext^,
        ups^,
        model_channels[n_stages - 1],
        lin_from(sd, "output_layer"),
    )
