# Mojo port of modules/transformer/blocks.py: AbsolutePositionEmbedder,
# FeedForwardNet, TransformerBlock, TransformerCrossBlock (dense [N, L, C]).
# The `phases` rope pass-through is deferred to WP8 with the model port.

from std.math import sin, cos

from trellis2_mojo.gpu.linear import gpu_mlp_forward, gpu_mlp_wants
from trellis2_mojo.sparse.tensor import Tensor, OP_ADD
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, activation, ACT_GELU_TANH
from trellis2_mojo.modules.attention import MultiHeadAttention

comptime F32 = DType.float32


struct AbsolutePositionEmbedder(Copyable, Movable):
    """Sin/cos positional embedding of integer positions [N, D] -> [N, channels],
    zero-padded when D * 2 * freq_dim < channels."""

    var channels: Int
    var in_channels: Int
    var freq_dim: Int
    var freqs: List[Float32]

    def __init__(out self, channels: Int, in_channels: Int = 3) raises:
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = List[Float32]()
        for j in range(self.freq_dim):
            var e = Float32(j) / Float32(self.freq_dim)
            self.freqs.append(1.0 / Float32(10000.0) ** e)

    def forward(self, pos: Tensor[F32]) raises -> Tensor[F32]:
        var n = pos.shape[0]
        var d = pos.shape[1]
        if d != self.in_channels:
            raise Error("AbsolutePositionEmbedder: input dim mismatch")
        var out_shape: List[Int] = [n, self.channels]
        var out = Tensor[F32](out_shape)
        var blk = 2 * self.freq_dim
        for r in range(n):
            for a in range(d):
                for j in range(self.freq_dim):
                    var angle = pos.data[r * d + a] * self.freqs[j]
                    out.data[r * self.channels + a * blk + j] = sin(angle)
                    out.data[r * self.channels + a * blk + self.freq_dim + j] = cos(angle)
        return out^


struct FeedForwardNet(Copyable, Movable):
    var lin1: SparseLinear
    var lin2: SparseLinear

    def __init__(out self, var lin1: SparseLinear, var lin2: SparseLinear):
        self.lin1 = lin1^
        self.lin2 = lin2^

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        # WP11 step 5: chain the two GEMMs + gelu on the GPU with the
        # [rows, hidden] intermediate device-resident
        if self.lin1.gpu:
            if self.lin2.gpu:
                var rows = x.numel() // x.shape[len(x.shape) - 1]
                if gpu_mlp_wants(self.lin1.gpu.value(), self.lin2.gpu.value(), rows):
                    return gpu_mlp_forward(x, self.lin1.gpu.value(), self.lin2.gpu.value())
        return self.lin2.forward(activation(self.lin1.forward(x), ACT_GELU_TANH))


struct TransformerBlock(Copyable, Movable):
    var norm1: LayerNorm32
    var norm2: LayerNorm32
    var attn: MultiHeadAttention
    var mlp: FeedForwardNet

    def __init__(
        out self,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var attn: MultiHeadAttention,
        var mlp: FeedForwardNet,
    ):
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.attn = attn^
        self.mlp = mlp^

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        var h = self.attn.forward(self.norm1.forward(x))
        var y = x._binop_flat(h, OP_ADD)
        var h2 = self.mlp.forward(self.norm2.forward(y))
        return y._binop_flat(h2, OP_ADD)


struct TransformerCrossBlock(Copyable, Movable):
    var norm1: LayerNorm32
    var norm2: LayerNorm32
    var norm3: LayerNorm32
    var self_attn: MultiHeadAttention
    var cross_attn: MultiHeadAttention
    var mlp: FeedForwardNet

    def __init__(
        out self,
        var norm1: LayerNorm32,
        var norm2: LayerNorm32,
        var norm3: LayerNorm32,
        var self_attn: MultiHeadAttention,
        var cross_attn: MultiHeadAttention,
        var mlp: FeedForwardNet,
    ):
        self.norm1 = norm1^
        self.norm2 = norm2^
        self.norm3 = norm3^
        self.self_attn = self_attn^
        self.cross_attn = cross_attn^
        self.mlp = mlp^

    def forward(self, x: Tensor[F32], context: Tensor[F32]) raises -> Tensor[F32]:
        var h = self.self_attn.forward(self.norm1.forward(x))
        var y = x._binop_flat(h, OP_ADD)
        var h2 = self.cross_attn.forward_cross(self.norm2.forward(y), context)
        y = y._binop_flat(h2, OP_ADD)
        var h3 = self.mlp.forward(self.norm3.forward(y))
        return y._binop_flat(h3, OP_ADD)
