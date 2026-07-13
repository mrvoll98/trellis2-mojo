# Builders that assemble Mojo modules from a StateDict — either a torch
# state_dict (parity tests; @implicit conversion from PythonObject) or the
# pure-Mojo safetensors dict (runner path, WP12).
#
# Key naming follows the originals, e.g. "attn.to_qkv.weight",
# "mlp.mlp.0.weight", "adaLN_modulation.1.bias".

from trellis2_mojo.gpu.attention import GpuAttnChain
from trellis2_mojo.gpu.conv import GpuSparseConv
from trellis2_mojo.gpu.linear import GpuLinear
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32
from trellis2_mojo.modules.conv import Conv3d
from trellis2_mojo.sparse.conv import SparseConv3d
from trellis2_mojo.modules.attention import MultiHeadAttention
from trellis2_mojo.sparse.attention.modules import SparseMultiHeadAttention, ATTN_MODE_FULL
from trellis2_mojo.sparse.transformer.blocks import SparseFeedForwardNet
from trellis2_mojo.sparse.transformer.modulated import Modulation
from trellis2_mojo.modules.transformer.blocks import FeedForwardNet
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def lin_from(sd: StateDict, prefix: String) raises -> SparseLinear:
    var sl = SparseLinear(
        sd.tensor(prefix + ".weight"), sd.tensor(prefix + ".bias")
    )
    sl.gpu = GpuLinear.try_build(sd.gpu, sl.weight, sl.bias, sl.has_bias)
    return sl^


def dummy_lin() raises -> SparseLinear:
    return SparseLinear(Tensor[F32]([1, 1]), Tensor[F32]([1]))


def conv3d_from(sd: StateDict, prefix: String, stride: Int = 1, padding: Int = 0) raises -> Conv3d:
    return Conv3d(
        sd.tensor(prefix + ".weight"), sd.tensor(prefix + ".bias"),
        stride, padding,
    )


def sparse_conv3d_from(sd: StateDict, prefix: String, dilation: Int = 1) raises -> SparseConv3d:
    """The conv_none state_dict weight is already [Co, Kd, Kh, Kw, Ci]."""
    var conv = SparseConv3d(
        sd.tensor(prefix + ".weight"), sd.tensor(prefix + ".bias"),
        dilation=dilation,
    )
    conv.gpu = GpuSparseConv.try_build(sd.gpu, conv.weight, conv.bias, conv.has_bias)
    return conv^


def ln_from(
    sd: StateDict, prefix: String, channels: Int, eps: Float64 = 1e-6, affine: Bool = False
) raises -> LayerNorm32:
    var ln = LayerNorm32(channels, eps, affine)
    if affine:
        ln.weight = sd.tensor(prefix + ".weight")
        ln.bias = sd.tensor(prefix + ".bias")
    return ln^


def sparse_mha_from(
    sd: StateDict,
    prefix: String,
    channels: Int,
    num_heads: Int,
    is_cross: Bool = False,
    attn_mode: Int = ATTN_MODE_FULL,
    window_size: Int = 0,
    use_rope: Bool = False,
    qk_rms_norm: Bool = False,
) raises -> SparseMultiHeadAttention:
    var mha: SparseMultiHeadAttention
    if is_cross:
        mha = SparseMultiHeadAttention(
            channels, num_heads, dummy_lin(),
            lin_from(sd, prefix + ".to_q"), lin_from(sd, prefix + ".to_kv"),
            lin_from(sd, prefix + ".to_out"),
            qk_rms_norm=qk_rms_norm,
        )
    else:
        mha = SparseMultiHeadAttention(
            channels, num_heads, lin_from(sd, prefix + ".to_qkv"),
            dummy_lin(), dummy_lin(), lin_from(sd, prefix + ".to_out"),
            attn_mode=attn_mode, window_size=window_size,
            use_rope=use_rope, qk_rms_norm=qk_rms_norm,
        )
    if qk_rms_norm:
        mha.q_rms_norm.gamma = sd.tensor(prefix + ".q_rms_norm.gamma")
        mha.k_rms_norm.gamma = sd.tensor(prefix + ".k_rms_norm.gamma")
    mha.gpu = sd.gpu.copy()
    # WP11 step 7: chained self-attention consts (None for cross — the
    # dummy to_qkv has no device weights); step 8: the cross variant
    # (None for self — the dummy to_q has no device weights)
    mha.chain = GpuAttnChain.try_build(
        sd.gpu, mha.to_qkv.gpu, mha.to_out.gpu,
        mha.q_rms_norm.gamma, mha.k_rms_norm.gamma, qk_rms_norm,
        num_heads, channels // num_heads,
    )
    mha.cross_chain = GpuAttnChain.try_build_cross(
        sd.gpu, mha.to_q.gpu, mha.to_out.gpu,
        mha.q_rms_norm.gamma, qk_rms_norm,
        num_heads, channels // num_heads,
    )
    return mha^


def dense_mha_from(
    sd: StateDict,
    prefix: String,
    channels: Int,
    num_heads: Int,
    is_cross: Bool = False,
    qk_rms_norm: Bool = False,
) raises -> MultiHeadAttention:
    var mha: MultiHeadAttention
    if is_cross:
        mha = MultiHeadAttention(
            channels, num_heads, dummy_lin(),
            lin_from(sd, prefix + ".to_q"), lin_from(sd, prefix + ".to_kv"),
            lin_from(sd, prefix + ".to_out"),
            qk_rms_norm=qk_rms_norm,
        )
    else:
        mha = MultiHeadAttention(
            channels, num_heads, lin_from(sd, prefix + ".to_qkv"),
            dummy_lin(), dummy_lin(), lin_from(sd, prefix + ".to_out"),
            qk_rms_norm=qk_rms_norm,
        )
    if qk_rms_norm:
        mha.q_rms_norm.gamma = sd.tensor(prefix + ".q_rms_norm.gamma")
        mha.k_rms_norm.gamma = sd.tensor(prefix + ".k_rms_norm.gamma")
    mha.gpu = sd.gpu.copy()
    mha.chain = GpuAttnChain.try_build(
        sd.gpu, mha.to_qkv.gpu, mha.to_out.gpu,
        mha.q_rms_norm.gamma, mha.k_rms_norm.gamma, qk_rms_norm,
        num_heads, channels // num_heads,
    )
    mha.cross_chain = GpuAttnChain.try_build_cross(
        sd.gpu, mha.to_q.gpu, mha.to_out.gpu,
        mha.q_rms_norm.gamma, qk_rms_norm,
        num_heads, channels // num_heads,
    )
    return mha^


def sparse_ffn_from(sd: StateDict, prefix: String) raises -> SparseFeedForwardNet:
    return SparseFeedForwardNet(lin_from(sd, prefix + ".mlp.0"), lin_from(sd, prefix + ".mlp.2"))


def dense_ffn_from(sd: StateDict, prefix: String) raises -> FeedForwardNet:
    return FeedForwardNet(lin_from(sd, prefix + ".mlp.0"), lin_from(sd, prefix + ".mlp.2"))


def modulation_from(sd: StateDict, share_mod: Bool) raises -> Modulation:
    if share_mod:
        return Modulation(sd.tensor("modulation"))
    return Modulation(lin_from(sd, "adaLN_modulation.1"))


def modulation_from(sd: StateDict, prefix: String, share_mod: Bool) raises -> Modulation:
    """Prefixed variant for blocks inside a model ("blocks.N.modulation" /
    "blocks.N.adaLN_modulation.1")."""
    if share_mod:
        return Modulation(sd.tensor(prefix + ".modulation"))
    return Modulation(lin_from(sd, prefix + ".adaLN_modulation.1"))
