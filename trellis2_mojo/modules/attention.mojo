# Mojo port of modules/attention/modules.py: dense MultiHeadAttention
# (used by the dense sparse-structure flow DiT). MultiHeadRMSNorm is shared
# with the sparse variant (same math).
#
# RoPE (pe_mode='rope' in the flow model) takes externally computed phases:
# the forward(x, phases) overload applies them after qk_rms_norm, matching
# the original's ordering.

from trellis2_mojo.gpu.context import GpuContext
from trellis2_mojo.gpu.attention import (
    GpuAttnChain,
    gpu_attn_cross_chain,
    gpu_attn_self_chain,
    gpu_dense_sdpa,
    gpu_sdpa_wants,
)
from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.modules.nn import SparseLinear
from trellis2_mojo.modules.rope import apply_rotary_embedding
from trellis2_mojo.sparse.attention.modules import MultiHeadRMSNorm
from trellis2_mojo.sparse.attention.full_attn import dense_sdpa_q_k_v, dense_sdpa_q_kv

comptime F32 = DType.float32


struct MultiHeadAttention(Copyable, Movable):
    var channels: Int
    var num_heads: Int
    var head_dim: Int
    var qk_rms_norm: Bool
    var to_qkv: SparseLinear
    var to_q: SparseLinear
    var to_kv: SparseLinear
    var to_out: SparseLinear
    var q_rms_norm: MultiHeadRMSNorm
    var k_rms_norm: MultiHeadRMSNorm
    # WP11 step 3: set by dense_mha_from when TRELLIS2_GPU=1 — big
    # self/cross attention runs the GPU GEMM-composition SDPA
    var gpu: Optional[GpuContext]
    # WP11 step 7: device-resident qkv->rms/rope->sdpa->out chain for big
    # self-attention (per-MHA consts built by dense_mha_from)
    var chain: Optional[GpuAttnChain]
    # WP11 step 8: device-resident q->rms->sdpa->out cross chain (kv is
    # computed + k-rms-normalized on the CPU and host-packed)
    var cross_chain: Optional[GpuAttnChain]

    def __init__(
        out self,
        channels: Int,
        num_heads: Int,
        var to_qkv: SparseLinear,
        var to_q: SparseLinear,
        var to_kv: SparseLinear,
        var to_out: SparseLinear,
        qk_rms_norm: Bool = False,
    ) raises:
        if channels % num_heads != 0:
            raise Error("MultiHeadAttention: channels % num_heads != 0")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm
        self.to_qkv = to_qkv^
        self.to_q = to_q^
        self.to_kv = to_kv^
        self.to_out = to_out^
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.gpu = None
        self.chain = None
        self.cross_chain = None

    def _use_gpu(self, n: Int, l: Int, lkv: Int) -> Bool:
        if not self.gpu:
            return False
        return n == 1 and gpu_sdpa_wants(l, lkv, self.head_dim, self.num_heads)

    def _use_chain(self, n: Int, l: Int) -> Bool:
        if not self.chain:
            return False
        return n == 1 and self.chain.value().wants(l, self.to_qkv.gpu.value())

    def forward(self, x: Tensor[F32]) raises -> Tensor[F32]:
        """Self-attention on dense x [N, L, C]."""
        var n = x.shape[0]
        var l = x.shape[1]
        if self._use_chain(n, l):
            return gpu_attn_self_chain(
                self.chain.value(), x, self.to_qkv.gpu.value(), self.to_out.gpu.value()
            )
        var qkv = self.to_qkv.forward(x)  # [N, L, 3C]
        var shape: List[Int] = [n, l, 3, self.num_heads, self.head_dim]
        qkv = Tensor[F32].from_values(shape, qkv.data)

        var parts = qkv.unbind(2)  # [N, L, H, D] x3
        var q = parts[0].copy()
        var k = parts[1].copy()
        if self.qk_rms_norm:
            q = self.q_rms_norm.forward(q)
            k = self.k_rms_norm.forward(k)
        var h: Tensor[F32]
        if self._use_gpu(n, l, l):
            h = gpu_dense_sdpa(self.gpu.value(), q, k, parts[2])
        else:
            h = dense_sdpa_q_k_v(q, k, parts[2])

        var out_shape: List[Int] = [n, l, self.channels]
        return self.to_out.forward(Tensor[F32].from_values(out_shape, h.data))

    def forward(self, x: Tensor[F32], phases: Tensor[F32]) raises -> Tensor[F32]:
        """Self-attention with RoPE phases [L, head_dim/2, 2]."""
        var n = x.shape[0]
        var l = x.shape[1]
        if self._use_chain(n, l):
            return gpu_attn_self_chain(
                self.chain.value(), x, self.to_qkv.gpu.value(), self.to_out.gpu.value(),
                phases,
            )
        var qkv = self.to_qkv.forward(x)  # [N, L, 3C]
        var shape: List[Int] = [n, l, 3, self.num_heads, self.head_dim]
        qkv = Tensor[F32].from_values(shape, qkv.data)

        var parts = qkv.unbind(2)  # [N, L, H, D] x3
        var q = parts[0].copy()
        var k = parts[1].copy()
        if self.qk_rms_norm:
            q = self.q_rms_norm.forward(q)
            k = self.k_rms_norm.forward(k)
        q = apply_rotary_embedding(q, phases)
        k = apply_rotary_embedding(k, phases)
        var h: Tensor[F32]
        if self._use_gpu(n, l, l):
            h = gpu_dense_sdpa(self.gpu.value(), q, k, parts[2])
        else:
            h = dense_sdpa_q_k_v(q, k, parts[2])

        var out_shape: List[Int] = [n, l, self.channels]
        return self.to_out.forward(Tensor[F32].from_values(out_shape, h.data))

    def forward_cross(self, x: Tensor[F32], context: Tensor[F32]) raises -> Tensor[F32]:
        """Cross-attention: x [N, L, C], context [N, Lkv, Cctx]."""
        var n = x.shape[0]
        var l = x.shape[1]
        var lkv = context.shape[1]
        if self.cross_chain and n == 1:
            if self.cross_chain.value().wants_cross(l, lkv):
                # WP11 step 8: kv (+ k-rms) on the CPU, q side chained
                var ckv_shape: List[Int] = [lkv, 2, self.num_heads, self.head_dim]
                var ckv = Tensor[F32].from_values(ckv_shape, self.to_kv.forward(context).data)
                var parts = ckv.unbind(1)
                var k = parts[0].copy()
                if self.qk_rms_norm:
                    k = self.k_rms_norm.forward(k)
                return gpu_attn_cross_chain(
                    self.cross_chain.value(), x, k, parts[1],
                    self.to_q.gpu.value(), self.to_out.gpu.value(),
                )
        var q_shape: List[Int] = [n, l, self.num_heads, self.head_dim]
        var q = Tensor[F32].from_values(q_shape, self.to_q.forward(x).data)
        var kv_shape: List[Int] = [n, lkv, 2, self.num_heads, self.head_dim]
        var kv = Tensor[F32].from_values(kv_shape, self.to_kv.forward(context).data)

        var h: Tensor[F32]
        var use_gpu = self._use_gpu(n, l, lkv)
        if self.qk_rms_norm:
            q = self.q_rms_norm.forward(q)
            var parts = kv.unbind(2)
            var k = self.k_rms_norm.forward(parts[0])
            if use_gpu:
                h = gpu_dense_sdpa(self.gpu.value(), q, k, parts[1])
            else:
                h = dense_sdpa_q_k_v(q, k, parts[1])
        elif use_gpu:
            var parts = kv.unbind(2)
            h = gpu_dense_sdpa(self.gpu.value(), q, parts[0], parts[1])
        else:
            h = dense_sdpa_q_kv(q, kv)

        var out_shape: List[Int] = [n, l, self.channels]
        return self.to_out.forward(Tensor[F32].from_values(out_shape, h.data))
