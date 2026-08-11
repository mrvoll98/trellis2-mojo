# Mojo port of models/sparse_structure_flow.py: TimestepEmbedder +
# SparseStructureFlowModel (dense DiT over the flattened resolution^3 grid).
#
# Ported: input/out Linear, APE or RoPE positional embedding precomputed on
# the voxel grid at build time, ModulatedTransformerCrossBlock stack,
# parameter-free final layer_norm (default eps 1e-5).
# Not ported: initialize_weights (weights always come from a state_dict),
# convert_to/manual_cast (float32-only in v1), use_checkpoint (training).
#
# Note: the original computes its rope_phases buffer with a default-freq
# RotaryPositionEmbedder — the rope_freq argument is only forwarded to the
# blocks, where it is never used (phases are always passed in). The port
# mirrors that: grid phases use the default (1.0, 10000.0) frequencies.

from std.math import cos, sin, exp, log
from trellis2_mojo.gpu.block import (
    gpu_block_phases,
    gpu_block_state_readback,
    gpu_block_state_upload,
)
from trellis2_mojo.io.state_dict import StateDict

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, activation, ACT_SILU
from trellis2_mojo.modules.rope import RotaryPositionEmbedder
from trellis2_mojo.modules.transformer.blocks import AbsolutePositionEmbedder
from trellis2_mojo.modules.transformer.modulated import ModulatedTransformerCrossBlock
from trellis2_mojo.loaders import (
    lin_from,
    ln_from,
    dense_mha_from,
    dense_ffn_from,
    modulation_from,
    dummy_lin,
)

comptime F32 = DType.float32


struct TimestepEmbedder(Copyable, Movable):
    """Sinusoidal timestep embedding -> MLP (Linear, SiLU, Linear)."""

    var frequency_embedding_size: Int
    var max_period: Float64
    var lin1: SparseLinear
    var lin2: SparseLinear

    def __init__(
        out self,
        var lin1: SparseLinear,
        var lin2: SparseLinear,
        frequency_embedding_size: Int = 256,
        max_period: Float64 = 10000.0,
    ):
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.lin1 = lin1^
        self.lin2 = lin2^

    def timestep_embedding(self, t: Tensor[F32]) raises -> Tensor[F32]:
        """t [N] -> [N, F]: cat([cos(t * freqs), sin(t * freqs)], dim=-1);
        the odd-F zero pad column falls out of the zero-filled allocation."""
        var n = t.numel()
        var dim = self.frequency_embedding_size
        var half = dim // 2
        var out_shape: List[Int] = [n, dim]
        var out = Tensor[F32](out_shape)
        # freqs = exp(-log(max_period) * arange(half) / half), computed in f32
        var neg_log = -Float32(log(Float64(self.max_period)))
        for r in range(n):
            for j in range(half):
                var freq = exp(neg_log * Float32(j) / Float32(half))
                var angle = t.data[r] * freq
                out.data[r * dim + j] = cos(angle)
                out.data[r * dim + half + j] = sin(angle)
        return out^

    def forward(self, t: Tensor[F32]) raises -> Tensor[F32]:
        var h = self.lin1.forward(self.timestep_embedding(t))
        return self.lin2.forward(activation(h, ACT_SILU))


struct SparseStructureFlowModel(Copyable, Movable):
    var resolution: Int
    var in_channels: Int
    var model_channels: Int
    var cond_channels: Int
    var out_channels: Int
    var num_heads: Int
    var use_rope: Bool   # pe_mode == "rope"
    var share_mod: Bool
    var t_embedder: TimestepEmbedder
    var share_lin: SparseLinear   # model-level adaLN_modulation.1 (share_mod)
    var pos_emb: Tensor[F32]      # [R^3, C] when APE, else [1]
    var rope_phases: Tensor[F32]  # [R^3, head_dim/2, 2] when RoPE, else [1]
    var input_layer: SparseLinear
    var blocks: List[ModulatedTransformerCrossBlock]
    var out_layer: SparseLinear
    var final_norm: LayerNorm32   # F.layer_norm(h, [C]): eps 1e-5, no affine

    def __init__(
        out self,
        resolution: Int,
        in_channels: Int,
        model_channels: Int,
        cond_channels: Int,
        out_channels: Int,
        num_heads: Int,
        use_rope: Bool,
        share_mod: Bool,
        var t_embedder: TimestepEmbedder,
        var share_lin: SparseLinear,
        var input_layer: SparseLinear,
        var blocks: List[ModulatedTransformerCrossBlock],
        var out_layer: SparseLinear,
    ) raises:
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.share_mod = share_mod
        self.t_embedder = t_embedder^
        self.share_lin = share_lin^
        self.input_layer = input_layer^
        self.blocks = blocks^
        self.out_layer = out_layer^
        self.final_norm = LayerNorm32(model_channels, 1e-5, False)

        # meshgrid(indexing='ij') voxel coordinates [R^3, 3]: l = i*R^2 + j*R + k
        var r3 = resolution * resolution * resolution
        var grid_shape: List[Int] = [r3, 3]
        var grid = Tensor[F32](grid_shape)
        var l = 0
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    grid.data[l * 3] = Float32(i)
                    grid.data[l * 3 + 1] = Float32(j)
                    grid.data[l * 3 + 2] = Float32(k)
                    l += 1
        if use_rope:
            var rope = RotaryPositionEmbedder(model_channels // num_heads, 3)
            self.rope_phases = rope.forward(grid)
            self.pos_emb = Tensor[F32]([1])
        else:
            var ape = AbsolutePositionEmbedder(model_channels, 3)
            self.pos_emb = ape.forward(grid)
            self.rope_phases = Tensor[F32]([1])

    def forward(self, x: Tensor[F32], t: Tensor[F32], cond: Tensor[F32]) raises -> Tensor[F32]:
        """x [N, C_in, R, R, R], t [N], cond [N, Lc, C_cond] -> [N, C_out, R, R, R]."""
        var r = self.resolution
        if (
            len(x.shape) != 5 or x.shape[1] != self.in_channels
            or x.shape[2] != r or x.shape[3] != r or x.shape[4] != r
        ):
            raise Error("SparseStructureFlowModel: input shape mismatch")
        var n = x.shape[0]
        var r3 = r * r * r

        # x.view(N, C_in, R^3).permute(0, 2, 1) -> [N, L, C_in]
        var h_shape: List[Int] = [n, r3, self.in_channels]
        var h = Tensor[F32](h_shape)
        for b in range(n):
            for c in range(self.in_channels):
                for l in range(r3):
                    h.data[(b * r3 + l) * self.in_channels + c] = x.data[(b * self.in_channels + c) * r3 + l]

        h = self.input_layer.forward(h)
        if not self.use_rope:
            # h = h + pos_emb[None]
            var mc = self.model_channels
            for b in range(n):
                for l in range(r3):
                    for c in range(mc):
                        h.data[(b * r3 + l) * mc + c] += self.pos_emb.data[l * mc + c]

        var t_emb = self.t_embedder.forward(t)
        if self.share_mod:
            t_emb = self.share_lin.forward(activation(t_emb, ACT_SILU))

        # WP11 step 12: model-level residency — when EVERY block takes the
        # whole-block GPU path, x stays device-resident across all blocks
        # (one upload + one readback per forward; the per-block consts/kv
        # uploads act as the inter-block syncs). Bit-identical to the
        # per-block path: the dropped readback/upload was an exact copy.
        var all_gpu = len(self.blocks) > 0
        for i in range(len(self.blocks)):
            if not self.blocks[i]._gpu_block_ok(n, r3, cond.shape[1]):
                all_gpu = False
                break
        if all_gpu:
            var g = self.blocks[0].self_attn.chain.value().g.copy()
            var ph_buf = self.blocks[0].self_attn.chain.value().consts
            if self.use_rope:
                ph_buf = gpu_block_phases(
                    g, self.rope_phases, r3, self.blocks[0].self_attn.head_dim
                )
            gpu_block_state_upload(g, h, r3, self.model_channels)
            for i in range(len(self.blocks)):
                self.blocks[i]._gpu_enqueue_resident(t_emb, cond, self.use_rope, ph_buf, r3)
            g.barrier()
            h = gpu_block_state_readback(g, h.shape, r3, self.model_channels)
        else:
            for i in range(len(self.blocks)):
                if self.use_rope:
                    h = self.blocks[i].forward(h, t_emb, cond, self.rope_phases)
                else:
                    h = self.blocks[i].forward(h, t_emb, cond)

        h = self.final_norm.forward(h)
        h = self.out_layer.forward(h)

        # [N, L, C_out].permute(0, 2, 1).view(N, C_out, R, R, R)
        var out_shape: List[Int] = [n, self.out_channels, r, r, r]
        var out = Tensor[F32](out_shape)
        for b in range(n):
            for l in range(r3):
                for c in range(self.out_channels):
                    out.data[(b * self.out_channels + c) * r3 + l] = h.data[(b * r3 + l) * self.out_channels + c]
        return out^


def sparse_structure_flow_from(
    sd: StateDict,
    resolution: Int,
    in_channels: Int,
    model_channels: Int,
    cond_channels: Int,
    out_channels: Int,
    num_blocks: Int,
    num_heads: Int,
    use_rope: Bool,
    share_mod: Bool,
    qk_rms_norm: Bool,
    qk_rms_norm_cross: Bool,
) raises -> SparseStructureFlowModel:
    """Build the model from a native StateDict (loaders.mojo pattern)."""
    var t_embedder = TimestepEmbedder(
        lin_from(sd, "t_embedder.mlp.0"), lin_from(sd, "t_embedder.mlp.2")
    )
    var share_lin: SparseLinear
    if share_mod:
        share_lin = lin_from(sd, "adaLN_modulation.1")
    else:
        share_lin = dummy_lin()
    var blocks = List[ModulatedTransformerCrossBlock]()
    for i in range(num_blocks):
        var p = "blocks." + String(i)
        blocks.append(
            ModulatedTransformerCrossBlock(
                model_channels,
                ln_from(sd, p + ".norm1", model_channels),
                ln_from(sd, p + ".norm2", model_channels, affine=True),
                ln_from(sd, p + ".norm3", model_channels),
                dense_mha_from(sd, p + ".self_attn", model_channels, num_heads, qk_rms_norm=qk_rms_norm),
                dense_mha_from(sd, p + ".cross_attn", model_channels, num_heads, is_cross=True, qk_rms_norm=qk_rms_norm_cross),
                dense_ffn_from(sd, p + ".mlp"),
                modulation_from(sd, p, share_mod),
            )
        )
    return SparseStructureFlowModel(
        resolution, in_channels, model_channels, cond_channels, out_channels,
        num_heads, use_rope, share_mod,
        t_embedder^, share_lin^,
        lin_from(sd, "input_layer"), blocks^, lin_from(sd, "out_layer"),
    )
