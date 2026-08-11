# Mojo port of models/structured_latent_flow.py: SLatFlowModel (sparse DiT).
#
# Ported: SparseLinear input/out, APE on coords (pe_mode='ape') or block-level
# RoPE (pe_mode='rope'; the sparse attention computes phases from coords
# itself, cached), ModulatedSparseTransformerCrossBlock stack, share_mod
# adaLN, parameter-free final layer_norm on feats, concat_cond
# (sparse_cat(dim=-1): feature concat on shared coords — the texture-SLat
# sampling path). Not ported: cond as List[Tensor] (VarLen cross-context;
# the pipelines pass dense [N, Lc, C] tensors), ElasticSLatFlowModel's
# elastic mixin (training-only, see ADR), initialize_weights /
# convert_to / use_checkpoint. `resolution` is metadata the forward never
# touches and is not stored.

from trellis2_mojo.gpu.block import (
    gpu_block_phases,
    gpu_block_state_readback,
    gpu_block_state_upload,
)
from trellis2_mojo.io.state_dict import StateDict

from trellis2_mojo.sparse.tensor import Tensor, OP_ADD
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, activation, ACT_SILU
from trellis2_mojo.modules.transformer.blocks import AbsolutePositionEmbedder
from trellis2_mojo.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock
from trellis2_mojo.models.sparse_structure_flow import TimestepEmbedder
from trellis2_mojo.loaders import (
    lin_from,
    ln_from,
    sparse_mha_from,
    sparse_ffn_from,
    modulation_from,
    dummy_lin,
)

comptime F32 = DType.float32


def _cat_features(a: SparseTensor[F32], b: SparseTensor[F32]) raises -> SparseTensor[F32]:
    """sparse_cat([a, b], dim=-1): feats concat on the channel dim; the
    original reuses a's coords/layout unchecked, we at least require equal
    row counts."""
    if a.coords.rows != b.coords.rows:
        raise Error("_cat_features: row count mismatch")
    return a.replace(a.vl.feats.cat_dim(b.vl.feats, 1))


struct SLatFlowModel(Copyable, Movable):
    var in_channels: Int
    var model_channels: Int
    var cond_channels: Int
    var out_channels: Int
    var num_heads: Int
    var use_rope: Bool   # pe_mode == "rope"
    var share_mod: Bool
    var t_embedder: TimestepEmbedder
    var share_lin: SparseLinear   # model-level adaLN_modulation.1 (share_mod)
    var pos_embedder: AbsolutePositionEmbedder  # used when APE
    var input_layer: SparseLinear
    var blocks: List[ModulatedSparseTransformerCrossBlock]
    var out_layer: SparseLinear
    var final_norm: LayerNorm32   # F.layer_norm(feats, [C]): eps 1e-5, no affine

    def __init__(
        out self,
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
        var blocks: List[ModulatedSparseTransformerCrossBlock],
        var out_layer: SparseLinear,
    ) raises:
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.share_mod = share_mod
        self.t_embedder = t_embedder^
        self.share_lin = share_lin^
        self.pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
        self.input_layer = input_layer^
        self.blocks = blocks^
        self.out_layer = out_layer^
        self.final_norm = LayerNorm32(model_channels, 1e-5, False)

    def forward(self, x: SparseTensor[F32], t: Tensor[F32], cond: Tensor[F32]) raises -> SparseTensor[F32]:
        """x sparse [T, C_in] over batch coords, t [N], cond dense [N, Lc, C_cond]."""
        if x.vl.feats.shape[1] != self.in_channels:
            raise Error("SLatFlowModel: input channel mismatch")
        var h = self.input_layer.forward(x)

        var t_emb = self.t_embedder.forward(t)
        if self.share_mod:
            t_emb = self.share_lin.forward(activation(t_emb, ACT_SILU))

        if not self.use_rope:
            # pe = pos_embedder(coords[:, 1:]); h = h + pe
            var n = h.coords.rows
            var d = h.coords.cols - 1
            var pos_shape: List[Int] = [n, d]
            var pos = Tensor[F32](pos_shape)
            for r in range(n):
                for c in range(d):
                    pos.data[r * d + c] = Float32(h.coords.at(r, c + 1))
            var pe = self.pos_embedder.forward(pos)
            h = h.replace(h.vl.feats._binop_flat(pe, OP_ADD))

        # WP11 step 12: model-level residency (see sparse_structure_flow) —
        # feats stay device-resident across all blocks when every block
        # takes the whole-block GPU path
        var all_gpu = len(self.blocks) > 0
        for i in range(len(self.blocks)):
            if not self.blocks[i]._gpu_block_ok(h, cond):
                all_gpu = False
                break
        if all_gpu:
            var t_rows = h.vl.feats.shape[0]
            var g = self.blocks[0].self_attn.chain.value().g.copy()
            var ph_buf = self.blocks[0].self_attn.chain.value().consts
            var use_rope = self.blocks[0].self_attn.use_rope
            if use_rope:
                var phases = self.blocks[0].self_attn.rope._phases(h)
                ph_buf = gpu_block_phases(
                    g, phases, t_rows, self.blocks[0].self_attn.head_dim
                )
            gpu_block_state_upload(g, h.vl.feats, t_rows, self.model_channels)
            for i in range(len(self.blocks)):
                self.blocks[i]._gpu_enqueue_resident(t_emb, cond, use_rope, ph_buf, t_rows)
            g.barrier()
            h = h.replace(gpu_block_state_readback(
                g, h.vl.feats.shape, t_rows, self.model_channels
            ))
        else:
            for i in range(len(self.blocks)):
                h = self.blocks[i].forward(h, t_emb, cond)

        h = h.replace(self.final_norm.forward(h.vl.feats))
        return self.out_layer.forward(h)

    def forward(
        self, x: SparseTensor[F32], t: Tensor[F32], cond: Tensor[F32], concat_cond: SparseTensor[F32]
    ) raises -> SparseTensor[F32]:
        """Texture-SLat path: x and concat_cond share coords; feats concat."""
        return self.forward(_cat_features(x, concat_cond), t, cond)


def slat_flow_from(
    sd: StateDict,
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
) raises -> SLatFlowModel:
    """Build the model from a native StateDict (loaders.mojo pattern).
    Works for ElasticSLatFlowModel checkpoints too — same keys."""
    var t_embedder = TimestepEmbedder(
        lin_from(sd, "t_embedder.mlp.0"), lin_from(sd, "t_embedder.mlp.2")
    )
    var share_lin: SparseLinear
    if share_mod:
        share_lin = lin_from(sd, "adaLN_modulation.1")
    else:
        share_lin = dummy_lin()
    var blocks = List[ModulatedSparseTransformerCrossBlock]()
    for i in range(num_blocks):
        var p = "blocks." + String(i)
        blocks.append(
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                ln_from(sd, p + ".norm1", model_channels),
                ln_from(sd, p + ".norm2", model_channels, affine=True),
                ln_from(sd, p + ".norm3", model_channels),
                sparse_mha_from(sd, p + ".self_attn", model_channels, num_heads,
                                use_rope=use_rope, qk_rms_norm=qk_rms_norm),
                sparse_mha_from(sd, p + ".cross_attn", model_channels, num_heads,
                                is_cross=True, qk_rms_norm=qk_rms_norm_cross),
                sparse_ffn_from(sd, p + ".mlp"),
                modulation_from(sd, p, share_mod),
            )
        )
    return SLatFlowModel(
        in_channels, model_channels, cond_channels, out_channels, num_heads,
        use_rope, share_mod,
        t_embedder^, share_lin^,
        lin_from(sd, "input_layer"), blocks^, lin_from(sd, "out_layer"),
    )
