# Mojo port of models/sparse_structure_vae.py — decoder side only
# (SparseStructureDecoder, ResBlock3d, UpsampleBlock3d, norm_layer).
# The encoder (and DownsampleBlock3d) is training-only: the pipeline samples
# the latent from the flow model and only decodes. UpsampleBlock3d's
# "nearest" mode is never constructed (decoder always uses "conv") and is
# not ported. zero_module/fp16-conversion are init/dtype details with no
# effect when weights come from a state_dict in f32.
#
# Upstream quirk (verified against the original): the decoder never forwards
# norm_type to its ResBlock3d's — they are constructed with the default
# ("layer"), so norm_type only affects out_layer. The loader mirrors that.

from trellis2_mojo.sparse.tensor import Tensor, OP_ADD
from trellis2_mojo.modules.nn import GroupNorm32, ChannelLayerNorm32, activation, ACT_SILU
from trellis2_mojo.modules.conv import Conv3d
from trellis2_mojo.modules.spatial import pixel_shuffle_3d
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.loaders import conv3d_from

comptime F32 = DType.float32

comptime NORM_GROUP = 0
comptime NORM_LAYER = 1


struct NormLayer3d(Copyable, Movable):
    """norm_layer(): GroupNorm32(32, C) for "group", ChannelLayerNorm32(C)
    for "layer" — both on dense [N, C, *spatial]."""

    var kind: Int
    var group: GroupNorm32
    var chlayer: ChannelLayerNorm32

    def __init__(out self, var group: GroupNorm32) raises:
        self.kind = NORM_GROUP
        self.group = group^
        self.chlayer = ChannelLayerNorm32(1)

    def __init__(out self, var chlayer: ChannelLayerNorm32) raises:
        self.kind = NORM_LAYER
        self.chlayer = chlayer^
        self.group = GroupNorm32(1, 1)

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        if self.kind == NORM_GROUP:
            return self.group.forward(x)
        return self.chlayer.forward(x)


struct ResBlock3d(Copyable, Movable):
    var norm1: NormLayer3d
    var norm2: NormLayer3d
    var conv1: Conv3d  # 3x3x3, padding 1
    var conv2: Conv3d  # 3x3x3, padding 1
    var has_skip: Bool
    var skip: Conv3d   # 1x1x1 when channels != out_channels

    def __init__(
        out self,
        var norm1: NormLayer3d,
        var norm2: NormLayer3d,
        var conv1: Conv3d,
        var conv2: Conv3d,
        has_skip: Bool,
        var skip: Conv3d,
    ):
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.conv1 = conv1^
        self.conv2 = conv2^
        self.has_skip = has_skip
        self.skip = skip^

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        var h = self.conv1.forward(activation(self.norm1.forward(x), ACT_SILU))
        h = self.conv2.forward(activation(self.norm2.forward(h), ACT_SILU))
        if self.has_skip:
            return h._binop_flat(self.skip.forward(x), OP_ADD)
        return h._binop_flat(x, OP_ADD)


struct UpsampleBlock3d(Copyable, Movable):
    """conv mode: Conv3d(C_in, C_out*8, 3, padding=1) + pixel_shuffle_3d(2)."""

    var conv: Conv3d

    def __init__(out self, var conv: Conv3d):
        self.conv = conv^

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        return pixel_shuffle_3d(self.conv.forward(x), 2)


struct SparseStructureDecoder(Copyable, Movable):
    """forward: input conv -> middle ResBlocks -> per stage (num_res_blocks
    ResBlocks, then Upsample except after the last stage) -> norm/SiLU/conv."""

    var num_res_blocks: Int  # per stage
    var input_layer: Conv3d
    var middle: List[ResBlock3d]
    var res_blocks: List[ResBlock3d]      # num_stages * num_res_blocks
    var upsamples: List[UpsampleBlock3d]  # num_stages - 1
    var out_norm: NormLayer3d
    var out_conv: Conv3d

    def __init__(
        out self,
        num_res_blocks: Int,
        var input_layer: Conv3d,
        var middle: List[ResBlock3d],
        var res_blocks: List[ResBlock3d],
        var upsamples: List[UpsampleBlock3d],
        var out_norm: NormLayer3d,
        var out_conv: Conv3d,
    ):
        self.num_res_blocks = num_res_blocks
        self.input_layer = input_layer^
        self.middle = middle^
        self.res_blocks = res_blocks^
        self.upsamples = upsamples^
        self.out_norm = out_norm^
        self.out_conv = out_conv^

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        var h = self.input_layer.forward(x)
        for i in range(len(self.middle)):
            h = self.middle[i].forward(h)
        var num_stages = len(self.upsamples) + 1
        for stage in range(num_stages):
            for j in range(self.num_res_blocks):
                h = self.res_blocks[stage * self.num_res_blocks + j].forward(h)
            if stage < num_stages - 1:
                h = self.upsamples[stage].forward(h)
        return self.out_conv.forward(activation(self.out_norm.forward(h), ACT_SILU))


def _key(prefix: String, name: String) raises -> String:
    """Join a state_dict prefix and a member name ("" prefix -> bare name,
    so blocks can be loaded from their own standalone state_dict)."""
    if prefix.byte_length() == 0:
        return name
    return prefix + "." + name


def norm3d_from(sd: StateDict, prefix: String, channels: Int, norm_group: Bool) raises -> NormLayer3d:
    if norm_group:
        var gn = GroupNorm32(32, channels)
        gn.weight = sd.tensor(prefix + ".weight")
        gn.bias = sd.tensor(prefix + ".bias")
        return NormLayer3d(gn^)
    var cln = ChannelLayerNorm32(channels)
    cln.inner.weight = sd.tensor(prefix + ".weight")
    cln.inner.bias = sd.tensor(prefix + ".bias")
    return NormLayer3d(cln^)


def resblock3d_from(
    sd: StateDict, prefix: String, channels: Int, out_channels: Int, norm_group: Bool
) raises -> ResBlock3d:
    var has_skip = channels != out_channels
    var skip: Conv3d
    if has_skip:
        skip = conv3d_from(sd, _key(prefix, "skip_connection"))
    else:
        skip = Conv3d(Tensor[F32]([1, 1, 1, 1, 1]), Tensor[F32]([1]))
    return ResBlock3d(
        norm3d_from(sd, _key(prefix, "norm1"), channels, norm_group),
        norm3d_from(sd, _key(prefix, "norm2"), out_channels, norm_group),
        conv3d_from(sd, _key(prefix, "conv1"), padding=1),
        conv3d_from(sd, _key(prefix, "conv2"), padding=1),
        has_skip,
        skip^,
    )


def sparse_structure_decoder_from(
    sd: StateDict,
    num_res_blocks: Int,
    channels: List[Int],
    num_res_blocks_middle: Int,
    norm_group: Bool,
) raises -> SparseStructureDecoder:
    """Build the decoder from a native StateDict (loaders.mojo pattern);
    out/latent channel counts are implied by the conv weights."""
    # res blocks always use "layer" norm upstream (norm_type is not forwarded)
    var middle = List[ResBlock3d]()
    for i in range(num_res_blocks_middle):
        middle.append(resblock3d_from(sd, "middle_block." + String(i), channels[0], channels[0], False))
    var res_blocks = List[ResBlock3d]()
    var upsamples = List[UpsampleBlock3d]()
    var idx = 0
    for stage in range(len(channels)):
        for _ in range(num_res_blocks):
            res_blocks.append(
                resblock3d_from(sd, "blocks." + String(idx), channels[stage], channels[stage], False)
            )
            idx += 1
        if stage < len(channels) - 1:
            upsamples.append(UpsampleBlock3d(conv3d_from(sd, "blocks." + String(idx) + ".conv", padding=1)))
            idx += 1
    return SparseStructureDecoder(
        num_res_blocks,
        conv3d_from(sd, "input_layer", padding=1),
        middle^,
        res_blocks^,
        upsamples^,
        norm3d_from(sd, "out_layer.0", channels[len(channels) - 1], norm_group),
        conv3d_from(sd, "out_layer.2", padding=1),
    )
