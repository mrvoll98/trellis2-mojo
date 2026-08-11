# Mojo port of modules/sparse/attention/modules.py:
# SparseMultiHeadRMSNorm + SparseMultiHeadAttention.
#
# self-attention: forward(x); cross-attention: forward_cross(x, context)
# with dense context [N, L, Cctx] (the models' cross-attn conditioning is
# always dense image features). Modes: full / windowed / double_windowed.

from max.algorithm import parallelize
from std.math import sqrt

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.attention import (
    GpuAttnChain,
    gpu_attn_cross_chain,
    gpu_attn_self_chain,
    gpu_varlen_sdpa_single,
    gpu_sdpa_wants,
)
from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.basic import SparseTensor
from trellis2_mojo.modules.nn import SparseLinear
from trellis2_mojo.sparse.attention.full_attn import (
    varlen_sdpa,
    sparse_sdpa_q_kv_dense,
    uniform_offsets,
)
from trellis2_mojo.sparse.attention.windowed_attn import sparse_windowed_sdpa_self
from trellis2_mojo.sparse.attention.rope import SparseRotaryPositionEmbedder

comptime F32 = DType.float32

comptime ATTN_MODE_FULL = 0
comptime ATTN_MODE_WINDOWED = 1
comptime ATTN_MODE_DOUBLE_WINDOWED = 2


struct MultiHeadRMSNorm(Copyable, Movable):
    """F.normalize(x, dim=-1) * gamma * sqrt(dim); x [..., H, D].
    Serves both the sparse and dense originals (same math on flat data)."""

    var gamma: Tensor[F32]  # [H, D]
    var scale: Float32

    def __init__(out self, dim: Int, heads: Int) raises:
        var shape: List[Int] = [heads, dim]
        self.gamma = Tensor[F32](shape, 1)
        self.scale = Float32(Float64(dim) ** 0.5)

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        """SIMD over the head dim (WP10 pass 6): the squared sum uses
        W-lane accumulation + reduce_add (a different order than the
        scalar loop — within parity tolerance); the scale pass applies the
        same per-element formula. Row chunks are parallelized for large
        inputs."""
        comptime W = 8
        comptime RC = 64
        var d = x.shape[len(x.shape) - 1]
        var h = x.shape[len(x.shape) - 2]
        var rows = x.numel() // (h * d)
        var out = Tensor[F32](x.shape)
        var xp = x.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()
        var gp = self.gamma.data.unsafe_ptr()
        var sc = self.scale

        @parameter
        def chunk(w: Int):
            var r1 = min((w + 1) * RC, rows)
            for r in range(w * RC, r1):
                for head in range(h):
                    var base = (r * h + head) * d
                    var gbase = head * d
                    var accv = SIMD[F32, W](0)
                    var i = 0
                    while i + W <= d:
                        var v = xp.unsafe_load[width=W](base + i)
                        accv += v * v
                        i += W
                    var norm = accv.reduce_add()
                    while i < d:
                        norm += xp[unsafe_offset=base + i] * xp[unsafe_offset=base + i]
                        i += 1
                    norm = sqrt(norm)
                    if norm < 1e-12:
                        norm = 1e-12
                    var nv = SIMD[F32, W](norm)
                    var sv = SIMD[F32, W](sc)
                    i = 0
                    while i + W <= d:
                        op.unsafe_store(base + i, xp.unsafe_load[width=W](base + i) / nv * gp.unsafe_load[width=W](gbase + i) * sv)
                        i += W
                    while i < d:
                        op[unsafe_offset=base + i] = xp[unsafe_offset=base + i] / norm * gp[unsafe_offset=gbase + i] * sc
                        i += 1

        var n_chunks = (rows + RC - 1) // RC
        if rows * h * d < 1 << 17:
            for w in range(n_chunks):
                chunk(w)
        else:
            parallelize[chunk](n_chunks)
        return out^


struct SparseMultiHeadAttention(Copyable, Movable):
    var channels: Int
    var num_heads: Int
    var head_dim: Int
    var attn_mode: Int
    var window_size: Int
    var shift_window: List[Int]
    var use_rope: Bool
    var qk_rms_norm: Bool
    # self-attn uses to_qkv; cross-attn uses to_q/to_kv (ctx_channels wide)
    var to_qkv: SparseLinear
    var to_q: SparseLinear
    var to_kv: SparseLinear
    var to_out: SparseLinear
    var q_rms_norm: MultiHeadRMSNorm
    var k_rms_norm: MultiHeadRMSNorm
    var rope: SparseRotaryPositionEmbedder
    # WP11 step 4: set by sparse_mha_from when TRELLIS2_GPU=1 — the
    # single-segment (B=1) full/cross cases run the GPU SDPA composition
    var gpu: Optional[GpuContext]
    # WP11 step 7: device-resident qkv->rms/rope->sdpa->out chain for the
    # single-segment full self-attention case
    var chain: Optional[GpuAttnChain]
    # WP11 step 8: device-resident q->rms->sdpa->out cross chain
    # (single-segment; kv computed + k-rms-normalized on the CPU)
    var cross_chain: Optional[GpuAttnChain]

    def __init__(
        out self,
        channels: Int,
        num_heads: Int,
        var to_qkv: SparseLinear,
        var to_q: SparseLinear,
        var to_kv: SparseLinear,
        var to_out: SparseLinear,
        attn_mode: Int = ATTN_MODE_FULL,
        window_size: Int = 0,
        shift_window: List[Int] = [0, 0, 0],
        use_rope: Bool = False,
        qk_rms_norm: Bool = False,
    ) raises:
        if channels % num_heads != 0:
            raise Error("SparseMultiHeadAttention: channels % num_heads != 0")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window.copy()
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.to_qkv = to_qkv^
        self.to_q = to_q^
        self.to_kv = to_kv^
        self.to_out = to_out^
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.rope = SparseRotaryPositionEmbedder(self.head_dim)
        self.gpu = None
        self.chain = None
        self.cross_chain = None

    def _use_gpu(self, q_offsets: List[Int], lkv: Int) -> Bool:
        """Single q segment (B=1) with GPU-worthy lengths."""
        if not self.gpu:
            return False
        if len(q_offsets) != 2:
            return False
        return gpu_sdpa_wants(q_offsets[1], lkv, self.head_dim, self.num_heads)

    def _use_chain(self, q_offsets: List[Int]) -> Bool:
        """Full-mode single-segment self-attention on the chained path."""
        if not self.chain:
            return False
        if len(q_offsets) != 2:
            return False
        return self.chain.value().wants(q_offsets[1], self.to_qkv.gpu.value())

    def forward(self, x: SparseTensor[F32]) raises -> SparseTensor[F32]:
        """Self-attention."""
        if self.attn_mode == ATTN_MODE_FULL and self._use_chain(x.vl.offsets):
            # WP11 step 7: rope phases come from the coords (spatial-cached
            # like the CPU path); rms/rope run in the chain's fused kernel
            if self.use_rope:
                var phases = self.rope._phases(x)
                return x.replace(gpu_attn_self_chain(
                    self.chain.value(), x.vl.feats,
                    self.to_qkv.gpu.value(), self.to_out.gpu.value(), phases,
                ))
            return x.replace(gpu_attn_self_chain(
                self.chain.value(), x.vl.feats,
                self.to_qkv.gpu.value(), self.to_out.gpu.value(),
            ))
        var qkv_feats = self.to_qkv.forward(x.vl.feats)  # [T, 3C]
        var tail: List[Int] = [3, self.num_heads, self.head_dim]
        qkv_feats = qkv_feats.reshape_rows(tail)          # [T, 3, H, D]

        if self.qk_rms_norm or self.use_rope:
            var parts = qkv_feats.unbind(1)
            var q = parts[0].copy()
            var k = parts[1].copy()
            if self.qk_rms_norm:
                q = self.q_rms_norm.forward(q)
                k = self.k_rms_norm.forward(k)
            if self.use_rope:
                var qk = self.rope.embed(x.replace(q^), x.replace(k^))
                q = qk[0].vl.feats.copy()
                k = qk[1].vl.feats.copy()
            qkv_feats = Tensor[F32].stack_dim1([q^, k^, parts[2].copy()])

        var h: Tensor[F32]
        if self.attn_mode == ATTN_MODE_FULL:
            var parts = qkv_feats.unbind(1)
            if self._use_gpu(x.vl.offsets, x.vl.feats.rows()):
                h = gpu_varlen_sdpa_single(
                    self.gpu.value(), parts[0], parts[1], parts[2]
                )
            else:
                h = varlen_sdpa(parts[0], parts[1], parts[2], x.vl.offsets, x.vl.offsets)
        elif self.attn_mode == ATTN_MODE_WINDOWED:
            var packed = x.replace(qkv_feats^)
            h = sparse_windowed_sdpa_self(packed, self.window_size, self.shift_window).vl.feats.copy()
        elif self.attn_mode == ATTN_MODE_DOUBLE_WINDOWED:
            # heads [H/2:] with no shift, heads [:H/2] with half-window shift,
            # outputs concatenated in that order (as the original)
            var hh = self.num_heads // 2
            var qkv0 = x.replace(qkv_feats.slice_dim(2, hh, self.num_heads))
            var qkv1 = x.replace(qkv_feats.slice_dim(2, 0, hh))
            var shift: List[Int] = [self.window_size // 2, self.window_size // 2, self.window_size // 2]
            var noshift: List[Int] = [0, 0, 0]
            var h0 = sparse_windowed_sdpa_self(qkv0, self.window_size, noshift)
            var h1 = sparse_windowed_sdpa_self(qkv1, self.window_size, shift)
            h = h0.vl.feats.cat_dim(h1.vl.feats, 1)
        else:
            raise Error("SparseMultiHeadAttention: unknown attn_mode")

        var flat: List[Int] = [self.channels]
        return x.replace(self.to_out.forward(h.reshape_rows(flat)))

    def forward_cross(self, x: SparseTensor[F32], context: Tensor[F32]) raises -> SparseTensor[F32]:
        """Cross-attention against dense context [N, L, Cctx]."""
        if self.cross_chain and context.shape[0] == 1 and len(x.vl.offsets) == 2:
            if self.cross_chain.value().wants_cross(x.vl.offsets[1], context.shape[1]):
                # WP11 step 8: kv (+ k-rms) on the CPU, q side chained
                var lkv = context.shape[1]
                var ckv_shape: List[Int] = [lkv, 2, self.num_heads, self.head_dim]
                var ckv = Tensor[F32].from_values(ckv_shape, self.to_kv.forward(context).data)
                var parts = ckv.unbind(1)
                var k = parts[0].copy()
                if self.qk_rms_norm:
                    k = self.k_rms_norm.forward(k)
                return x.replace(gpu_attn_cross_chain(
                    self.cross_chain.value(), x.vl.feats, k, parts[1],
                    self.to_q.gpu.value(), self.to_out.gpu.value(),
                ))
        var q_tail: List[Int] = [self.num_heads, self.head_dim]
        var q = self.to_q.forward(x.vl.feats).reshape_rows(q_tail)  # [T, H, D]

        var n = context.shape[0]
        var l = context.shape[1]
        var kv = self.to_kv.forward(context)  # [N, L, 2C]
        var kv_shape: List[Int] = [n, l, 2, self.num_heads, self.head_dim]
        kv = Tensor[F32].from_values(kv_shape, kv.data)

        var h: Tensor[F32]
        var use_gpu = n == 1 and self._use_gpu(x.vl.offsets, l)
        if self.qk_rms_norm:
            q = self.q_rms_norm.forward(q)
            var parts = kv.flatten_leading(2).unbind(1)  # [N*L, H, D] x2
            var k = self.k_rms_norm.forward(parts[0])
            if use_gpu:
                h = gpu_varlen_sdpa_single(self.gpu.value(), q, k, parts[1])
            else:
                h = varlen_sdpa(q, k, parts[1], x.vl.offsets, uniform_offsets(n, l))
        else:
            var parts = kv.flatten_leading(2).unbind(1)
            if use_gpu:
                h = gpu_varlen_sdpa_single(self.gpu.value(), q, parts[0], parts[1])
            else:
                h = varlen_sdpa(q, parts[0], parts[1], x.vl.offsets, uniform_offsets(n, l))

        var flat: List[Int] = [self.channels]
        return x.replace(self.to_out.forward(h.reshape_rows(flat)))
