# Mojo port of modules/transformer/modulated.py: ModulatedTransformerBlock /
# ModulatedTransformerCrossBlock on dense [N, L, C], adaLN conditioning.
# Reuses the Modulation helper from the sparse variant (identical math).
#
# WP11 step 10: when all three chains qualify (self/cross attention +
# mlp) the cross-block runs WHOLE-BLOCK device-resident
# (gpu_cross_block_forward) — one upload, one readback, glue on the GPU.

from std.gpu.host import DeviceBuffer

from trellis2_mojo.gpu.block import gpu_cross_block_forward, gpu_cross_block_enqueue
from trellis2_mojo.gpu.linear import gpu_mlp_wants
from trellis2_mojo.sparse.tensor import Tensor, OP_ADD
from trellis2_mojo.modules.nn import LayerNorm32, modulate
from trellis2_mojo.modules.attention import MultiHeadAttention
from trellis2_mojo.modules.transformer.blocks import FeedForwardNet
from trellis2_mojo.sparse.transformer.modulated import Modulation

comptime F32 = DType.float32


def _gate(x: Tensor[F32], g: Tensor[F32]) raises -> Tensor[F32]:
    """x [N, L, C] * g [N, C] broadcast over L (h * gate.unsqueeze(1))."""
    var n = x.shape[0]
    var l = x.shape[1]
    var c = x.shape[2]
    var out = Tensor[F32](x.shape)
    for b in range(n):
        for j in range(l):
            for ci in range(c):
                var idx = (b * l + j) * c + ci
                out.data[idx] = x.data[idx] * g.data[b * c + ci]
    return out^


struct ModulatedTransformerBlock(Copyable, Movable):
    var channels: Int
    var norm1: LayerNorm32
    var norm2: LayerNorm32
    var attn: MultiHeadAttention
    var mlp: FeedForwardNet
    var modulation: Modulation

    def __init__(
        out self,
        channels: Int,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var attn: MultiHeadAttention,
        var mlp: FeedForwardNet,
        var modulation: Modulation,
    ):
        self.channels = channels
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.attn = attn^
        self.mlp = mlp^
        self.modulation = modulation^

    def forward(self, x: Tensor[F32], mod: Tensor[F32]) raises -> Tensor[F32]:
        var m = self.modulation.chunks(mod, self.channels)
        var h = modulate(self.norm1.forward(x), m[0], m[1])
        h = self.attn.forward(h)
        h = _gate(h, m[2])
        var y = x._binop_flat(h, OP_ADD)
        var h2 = modulate(self.norm2.forward(y), m[3], m[4])
        h2 = self.mlp.forward(h2)
        h2 = _gate(h2, m[5])
        return y._binop_flat(h2, OP_ADD)


struct ModulatedTransformerCrossBlock(Copyable, Movable):
    var channels: Int
    var norm1: LayerNorm32
    var norm2: LayerNorm32  # elementwise_affine=True in the original
    var norm3: LayerNorm32
    var self_attn: MultiHeadAttention
    var cross_attn: MultiHeadAttention
    var mlp: FeedForwardNet
    var modulation: Modulation

    def __init__(
        out self,
        channels: Int,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var norm3: LayerNorm32,
        var self_attn: MultiHeadAttention,
        var cross_attn: MultiHeadAttention,
        var mlp: FeedForwardNet,
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

    def _gpu_block_ok(self, n: Int, l: Int, lkv: Int) -> Bool:
        """WP11 step 10: whole-block residency gate — all three chains
        must qualify and the norms must match the fused kernels' layout
        (eps 1e-6; norm1/3 plain, norm2 affine)."""
        if n != 1:
            return False
        if not self.self_attn.chain or not self.cross_attn.cross_chain:
            return False
        if not self.self_attn.chain.value().wants(l, self.self_attn.to_qkv.gpu.value()):
            return False
        if not self.cross_attn.cross_chain.value().wants_cross(l, lkv):
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

    def _gpu_forward(
        self, x: Tensor[F32], mod: Tensor[F32], context: Tensor[F32],
        use_rope: Bool, phases: Tensor[F32],
    ) raises -> Tensor[F32]:
        """Whole-block device-resident path; kv (+ k-rms) on the CPU as in
        the cross chain's dispatch."""
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
        return gpu_cross_block_forward(
            x, m, k, parts[1], use_rope, phases,
            self.norm2.weight, self.norm2.bias,
            self.self_attn.chain.value(),
            self.self_attn.to_qkv.gpu.value(), self.self_attn.to_out.gpu.value(),
            self.cross_attn.cross_chain.value(),
            self.cross_attn.to_q.gpu.value(), self.cross_attn.to_out.gpu.value(),
            self.mlp.lin1.gpu.value(), self.mlp.lin2.gpu.value(),
        )

    def _gpu_enqueue_resident(
        self, mod: Tensor[F32], context: Tensor[F32],
        use_rope: Bool, ph_buf: DeviceBuffer[DType.float32], rows: Int,
    ) raises:
        """WP11 step 12: enqueue this block against the RESIDENT xs state
        (the model uploads x once and reads back after the last block).
        Callers gate every block with _gpu_block_ok first."""
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

    def forward(self, x: Tensor[F32], mod: Tensor[F32], context: Tensor[F32]) raises -> Tensor[F32]:
        if self._gpu_block_ok(x.shape[0], x.shape[1], context.shape[1]):
            return self._gpu_forward(x, mod, context, False, Tensor[F32]([1, 1, 2]))
        var m = self.modulation.chunks(mod, self.channels)
        var h = self.self_attn.forward(modulate(self.norm1.forward(x), m[0], m[1]))
        return self._tail(x, m, h, context)

    def forward(self, x: Tensor[F32], mod: Tensor[F32], context: Tensor[F32], phases: Tensor[F32]) raises -> Tensor[F32]:
        """RoPE variant: phases [L, head_dim/2, 2] go to self-attention only."""
        if self._gpu_block_ok(x.shape[0], x.shape[1], context.shape[1]):
            return self._gpu_forward(x, mod, context, True, phases)
        var m = self.modulation.chunks(mod, self.channels)
        var h = self.self_attn.forward(modulate(self.norm1.forward(x), m[0], m[1]), phases)
        return self._tail(x, m, h, context)

    def _tail(self, x: Tensor[F32], m: List[Tensor[F32]], attn_h: Tensor[F32], context: Tensor[F32]) raises -> Tensor[F32]:
        var h = _gate(attn_h, m[2])
        var y = x._binop_flat(h, OP_ADD)
        var h2 = self.cross_attn.forward_cross(self.norm2.forward(y), context)
        y = y._binop_flat(h2, OP_ADD)
        var h3 = modulate(self.norm3.forward(y), m[3], m[4])
        h3 = self.mlp.forward(h3)
        h3 = _gate(h3, m[5])
        return y._binop_flat(h3, OP_ADD)
