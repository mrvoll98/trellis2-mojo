# Mojo port of modules/sparse/transformer/blocks.py:
# SparseFeedForwardNet, SparseTransformerBlock, SparseTransformerCrossBlock.
# Pure composition of WP4/WP5 modules; use_checkpoint is a training concern
# and not ported.

from trellis2_mojo.gpu.linear import gpu_mlp_forward, gpu_mlp_wants
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, activation, ACT_GELU_TANH
from trellis2_mojo.sparse.attention.modules import SparseMultiHeadAttention

comptime F32 = DType.float32


struct SparseFeedForwardNet(Copyable, Movable):
    var lin1: SparseLinear
    var lin2: SparseLinear

    def __init__(out self, var lin1: SparseLinear, var lin2: SparseLinear):
        self.lin1 = lin1^
        self.lin2 = lin2^

    def _mlp(self, x: Tensor[F32]) raises -> Tensor[F32]:
        # WP11 step 5: chain the two GEMMs + gelu on the GPU with the
        # [rows, hidden] intermediate device-resident
        if self.lin1.gpu:
            if self.lin2.gpu:
                var rows = x.numel() // x.shape[len(x.shape) - 1]
                if gpu_mlp_wants(self.lin1.gpu.value(), self.lin2.gpu.value(), rows):
                    return gpu_mlp_forward(x, self.lin1.gpu.value(), self.lin2.gpu.value())
        return self.lin2.forward(activation(self.lin1.forward(x), ACT_GELU_TANH))

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        return x.replace(self._mlp(x.vl.feats))

    def forward_dense(self, x: Tensor[F32]) raises -> Tensor[F32]:
        return self._mlp(x)


struct SparseTransformerBlock(Copyable, Movable):
    var norm1: LayerNorm32
    var norm2: LayerNorm32
    var attn: SparseMultiHeadAttention
    var mlp: SparseFeedForwardNet

    def __init__(
        out self,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var attn: SparseMultiHeadAttention,
        var mlp: SparseFeedForwardNet,
    ):
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.attn = attn^
        self.mlp = mlp^

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        var h = self.attn.forward(x.replace(self.norm1.forward(x.vl.feats)))
        var y = x + h
        var h2 = self.mlp.forward(y.replace(self.norm2.forward(y.vl.feats)))
        return y + h2


struct SparseTransformerCrossBlock(Copyable, Movable):
    var norm1: LayerNorm32
    var norm2: LayerNorm32
    var norm3: LayerNorm32
    var self_attn: SparseMultiHeadAttention
    var cross_attn: SparseMultiHeadAttention
    var mlp: SparseFeedForwardNet

    def __init__(
        out self,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var norm3: LayerNorm32,
        var self_attn: SparseMultiHeadAttention,
        var cross_attn: SparseMultiHeadAttention,
        var mlp: SparseFeedForwardNet,
    ):
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.norm3 = norm3^
        self.self_attn = self_attn^
        self.cross_attn = cross_attn^
        self.mlp = mlp^

    def forward(self, x: SparseTensor[F32], context: Tensor[F32]) raises -> SparseTensor[F32]:
        var h = self.self_attn.forward(x.replace(self.norm1.forward(x.vl.feats)))
        var y = x + h
        var h2 = self.cross_attn.forward_cross(y.replace(self.norm2.forward(y.vl.feats)), context)
        y = y + h2
        var h3 = self.mlp.forward(y.replace(self.norm3.forward(y.vl.feats)))
        return y + h3
