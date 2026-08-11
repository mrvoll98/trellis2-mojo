# Mojo port of modules/sparse/transformer/modulated.py:
# ModulatedSparseTransformerBlock / ModulatedSparseTransformerCrossBlock
# (adaLN conditioning). The six shift/scale/gate chunks are applied with the
# SparseTensor batch-broadcast ops (matching the original's x * (1+scale) +
# shift on VarLen semantics).
#
# WP11 step 10: the single-segment cross-block runs WHOLE-BLOCK
# device-resident when all three chains qualify (same orchestrator as the
# dense block — identical structure on flat feats [T, C]).

from max.gpu.host import DeviceBuffer

from trellis2_mojo.gpu.block import gpu_cross_block_forward, gpu_cross_block_enqueue
from trellis2_mojo.gpu.linear import gpu_mlp_wants
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor, OP_ADD, OP_MUL
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, activation, ACT_SILU
from trellis2_mojo.sparse.attention.modules import SparseMultiHeadAttention, ATTN_MODE_FULL
from trellis2_mojo.sparse.transformer.blocks import SparseFeedForwardNet

comptime F32 = DType.float32


struct Modulation(Copyable, Movable):
    """adaLN_modulation (SiLU -> Linear(C, 6C)) or the share_mod parameter
    path ((modulation + mod).chunk(6)). chunks() -> 6 x [N, C] in the order
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp."""

    var share_mod: Bool
    var lin: SparseLinear        # used when not share_mod
    var modulation: Tensor[F32]  # [6C], used when share_mod

    def __init__(out self, var lin: SparseLinear) raises:
        self.share_mod = False
        self.lin = lin^
        self.modulation = Tensor[F32]([1])

    def __init__(out self, var modulation: Tensor[F32]) raises:
        self.share_mod = True
        self.modulation = modulation^
        self.lin = SparseLinear(Tensor[F32]([1, 1]), Tensor[F32]([1]))

    def chunks(self, mod: Tensor[F32], channels: Int) raises -> List[Tensor[F32]]:
        var m: Tensor[F32]
        if self.share_mod:
            # mod is [N, 6C]; add the [6C] parameter row-broadcast
            m = Tensor[F32](mod.shape)
            var rs = mod.row_size()
            for r in range(mod.rows()):
                for j in range(rs):
                    m.data[r * rs + j] = mod.data[r * rs + j] + self.modulation.data[j]
        else:
            m = self.lin.forward(activation(mod, ACT_SILU))
        var out = List[Tensor[F32]]()
        for i in range(6):
            out.append(m.slice_dim(1, i * channels, (i + 1) * channels))
        return out^


def _mod_shift_scale(
    h: SparseTensor[F32], shift: Tensor[F32], scale: Tensor[F32]
) raises -> SparseTensor[F32]:
    """h * (1 + scale) + shift with [N, C] batch broadcast."""
    var scaled = h.elemwise_batch(scale._binop_scalar(1.0, OP_ADD), OP_MUL)
    return scaled.elemwise_batch(shift, OP_ADD)


struct ModulatedSparseTransformerBlock(Copyable, Movable):
    var channels: Int
    var norm1: LayerNorm32
    var norm2: LayerNorm32
    var attn: SparseMultiHeadAttention
    var mlp: SparseFeedForwardNet
    var modulation: Modulation

    def __init__(
        out self,
        channels: Int,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var attn: SparseMultiHeadAttention,
        var mlp: SparseFeedForwardNet,
        var modulation: Modulation,
    ):
        self.channels = channels
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.attn = attn^
        self.mlp = mlp^
        self.modulation = modulation^

    def forward(self, x: SparseTensor[F32], mod: Tensor[F32]) raises -> SparseTensor[F32]:
        var m = self.modulation.chunks(mod, self.channels)
        var h = x.replace(self.norm1.forward(x.vl.feats))
        h = _mod_shift_scale(h, m[0], m[1])
        h = self.attn.forward(h)
        h = h.elemwise_batch(m[2], OP_MUL)
        var y = x + h
        var h2 = y.replace(self.norm2.forward(y.vl.feats))
        h2 = _mod_shift_scale(h2, m[3], m[4])
        h2 = self.mlp.forward(h2)
        h2 = h2.elemwise_batch(m[5], OP_MUL)
        return y + h2


struct ModulatedSparseTransformerCrossBlock(Copyable, Movable):
    var channels: Int
    var norm1: LayerNorm32
    var norm2: LayerNorm32  # elementwise_affine=True in the original
    var norm3: LayerNorm32
    var self_attn: SparseMultiHeadAttention
    var cross_attn: SparseMultiHeadAttention
    var mlp: SparseFeedForwardNet
    var modulation: Modulation

    def __init__(
        out self,
        channels: Int,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var norm3: LayerNorm32,
        var self_attn: SparseMultiHeadAttention,
        var cross_attn: SparseMultiHeadAttention,
        var mlp: SparseFeedForwardNet,
        var modulation: Modulation,
    ):
        self.channels = channels
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.norm3 = norm3^
        self.self_attn = self_attn^
        self.cross_attn = cross_attn^
        self.mlp = mlp^
        self.modulation = modulation^

    def _gpu_block_ok(self, x: SparseTensor[F32], context: Tensor[F32]) -> Bool:
        """WP11 step 10: whole-block residency gate (single segment, full
        self-attention mode, all three chains qualify, norm layout)."""
        if context.shape[0] != 1 or len(x.vl.offsets) != 2:
            return False
        if self.self_attn.attn_mode != ATTN_MODE_FULL:
            return False
        var l = x.vl.offsets[1]
        if not self.self_attn.chain or not self.cross_attn.cross_chain:
            return False
        if not self.self_attn.chain.value().wants(l, self.self_attn.to_qkv.gpu.value()):
            return False
        if not self.cross_attn.cross_chain.value().wants_cross(l, context.shape[1]):
            return False
        if not self.mlp.lin1.gpu or not self.mlp.lin2.gpu:
            return False
        if not gpu_mlp_wants(self.mlp.lin1.gpu.value(), self.mlp.lin2.gpu.value(), l):
            return False
        if self.norm1.affine or self.norm3.affine or not self.norm2.affine:
            return False
        if self.norm1.eps != 1e-6 or self.norm2.eps != 1e-6 or self.norm3.eps != 1e-6:
            return False
        return True

    def _gpu_enqueue_resident(
        self, mod: Tensor[F32], context: Tensor[F32],
        use_rope: Bool, ph_buf: DeviceBuffer[DType.float32], rows: Int,
    ) raises:
        """WP11 step 12: enqueue this block against the RESIDENT xs state.
        Callers gate every block with _gpu_block_ok first; rope phases are
        uploaded once per model forward (same coords for every block)."""
        var m = self.modulation.chunks(mod, self.channels)
        var lkv = context.shape[1]
        var heads = self.cross_attn.num_heads
        var hd = self.cross_attn.head_dim
        var ckv_shape: List[Int] = [lkv, 2, heads, hd]
        var ckv = Tensor[F32].from_values(ckv_shape, self.cross_attn.to_kv.forward(context).data)
        var parts = ckv.unbind(1)
        var k = parts[0].copy()
        if self.cross_attn.qk_rms_norm:
            k = self.cross_attn.k_rms_norm.forward(k)
        gpu_cross_block_enqueue(
            rows, m, k, parts[1], use_rope, ph_buf,
            self.norm2.weight, self.norm2.bias,
            self.self_attn.chain.value(),
            self.self_attn.to_qkv.gpu.value(), self.self_attn.to_out.gpu.value(),
            self.cross_attn.cross_chain.value(),
            self.cross_attn.to_q.gpu.value(), self.cross_attn.to_out.gpu.value(),
            self.mlp.lin1.gpu.value(), self.mlp.lin2.gpu.value(),
        )

    def forward(
        self, x: SparseTensor[F32], mod: Tensor[F32], context: Tensor[F32]
    ) raises -> SparseTensor[F32]:
        if self._gpu_block_ok(x, context):
            var m = self.modulation.chunks(mod, self.channels)
            var lkv = context.shape[1]
            var heads = self.cross_attn.num_heads
            var hd = self.cross_attn.head_dim
            var ckv_shape: List[Int] = [lkv, 2, heads, hd]
            var ckv = Tensor[F32].from_values(ckv_shape, self.cross_attn.to_kv.forward(context).data)
            var parts = ckv.unbind(1)
            var k = parts[0].copy()
            if self.cross_attn.qk_rms_norm:
                k = self.cross_attn.k_rms_norm.forward(k)
            var use_rope = self.self_attn.use_rope
            var phases = Tensor[F32]([1, 1, 2])
            if use_rope:
                phases = self.self_attn.rope._phases(x)
            return x.replace(gpu_cross_block_forward(
                x.vl.feats, m, k, parts[1], use_rope, phases,
                self.norm2.weight, self.norm2.bias,
                self.self_attn.chain.value(),
                self.self_attn.to_qkv.gpu.value(), self.self_attn.to_out.gpu.value(),
                self.cross_attn.cross_chain.value(),
                self.cross_attn.to_q.gpu.value(), self.cross_attn.to_out.gpu.value(),
                self.mlp.lin1.gpu.value(), self.mlp.lin2.gpu.value(),
            ))
        var m = self.modulation.chunks(mod, self.channels)
        var h = x.replace(self.norm1.forward(x.vl.feats))
        h = _mod_shift_scale(h, m[0], m[1])
        h = self.self_attn.forward(h)
        h = h.elemwise_batch(m[2], OP_MUL)
        var y = x + h
        var h2 = self.cross_attn.forward_cross(y.replace(self.norm2.forward(y.vl.feats)), context)
        y = y + h2
        var h3 = y.replace(self.norm3.forward(y.vl.feats))
        h3 = _mod_shift_scale(h3, m[3], m[4])
        h3 = self.mlp.forward(h3)
        h3 = h3.elemwise_batch(m[5], OP_MUL)
        return y + h3
