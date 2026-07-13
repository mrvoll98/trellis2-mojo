# Mojo port of the DINOv3 ViT image encoder (WP13) — mirrors the
# transformers implementation (models/dinov3_vit/modeling_dinov3_vit.py)
# as driven by the upstream DinoV3FeatureExtractor / cond_io.get_cond:
# embeddings -> rope -> manual layer loop -> final F.layer_norm WITHOUT
# affine. The checkpoint's `norm.{weight,bias}` is deliberately NOT used —
# the extractor never applies the model's own affine norm.
#
# Architecture notes (facebook/dinov3-vitl16, mirrored exactly):
#   - patch conv 16x16 stride 16 == im2col + linear over [Cin*p*p] in
#     (c, i, j) order — the conv weight [Co, Cin, p, p] flattens row-major
#     to the matching linear layout.
#   - tokens = [cls; 4 register tokens; patches (row-major over the grid)].
#   - 2D-RoPE theta=100: inv_freq over head_dim/4, angle row layout
#     [y*f0..y*f_{q-1}, x*f0..x*f_{q-1}] then tile(2); rotate_half pairs
#     (i, i+D/2) — NOT the per-pair interleave our flow models use. Applied
#     to patch tokens only (prefix tokens pass through), q and k both.
#     pos_embed_{shift,jitter,rescale} are train-time augments — eval skips
#     them (transformers gates on self.training).
#   - attention: separate q/k/v projections, q/v/o have bias, k does NOT
#     (config key_bias=false); softmax scale 1/sqrt(head_dim) (varlen_sdpa).
#   - blocks: LN -> attn -> LayerScale -> res; LN -> MLP(up, exact-erf gelu,
#     down — NOT gated) -> LayerScale -> res. LN eps 1e-5.
#   - mask_token is pre-training only (bool_masked_pos=None) — not loaded.
#
# Weights load through the WP12 pure-Mojo safetensors reader (f32 on disk);
# the parity test feeds torch state_dicts through the same StateDict facade.

from std.math import cos, sin

from trellis2_mojo.gpu.linear import GpuLinear
from trellis2_mojo.io.state_dict import StateDict
from trellis2_mojo.modules.nn import SparseLinear, LayerNorm32, linear, activation, ACT_GELU
from trellis2_mojo.sparse.attention.full_attn import dense_sdpa_q_k_v
from trellis2_mojo.sparse.tensor import Tensor, OP_ADD, OP_MUL

comptime F32 = DType.float32


def _scale_channels(x: Tensor[F32], lam: Tensor[F32]) raises -> Tensor[F32]:
    """LayerScale: x [..., C] * lambda1 [C], broadcast over all leading dims."""
    var c = lam.numel()
    var rows = x.numel() // c
    var flat = Tensor[F32].from_values([rows, c], x.data)
    var lam_row = Tensor[F32].from_values([1, c], lam.data)
    var row_map = List[Int](length=rows, fill=0)
    var out = flat._binop_rows(lam_row, row_map, OP_MUL)
    return Tensor[F32].from_values(x.shape, out.data)


def rope_cos_sin(
    num_patches_h: Int, num_patches_w: Int, head_dim: Int, theta: Float64
) raises -> Tuple[Tensor[F32], Tensor[F32]]:
    """cos/sin [P, head_dim/2] for the patch grid, all math in f32 like the
    transformers rope (autocast disabled there). Row p = (py, px) row-major;
    angle layout [y*f0..y*f_{q-1}, x*f0..x*f_{q-1}] with q = head_dim/4
    freqs 1/theta^(4i/head_dim). tile(2) is implied: the apply step reuses
    angle i for lane i + head_dim/2."""
    var quarter = head_dim // 4
    var half = head_dim // 2
    var inv_freq = List[Float32]()
    for i in range(quarter):
        var t = Float32(Float64(i) * 4.0 / Float64(head_dim))
        inv_freq.append(Float32(1.0) / Float32(theta) ** t)
    var two_pi = Float32(6.283185307179586)
    var p = num_patches_h * num_patches_w
    var cos_t = Tensor[F32]([p, half])
    var sin_t = Tensor[F32]([p, half])
    for py in range(num_patches_h):
        var cy = Float32(2.0) * (Float32(py) + 0.5) / Float32(num_patches_h) - 1.0
        for px in range(num_patches_w):
            var cx = Float32(2.0) * (Float32(px) + 0.5) / Float32(num_patches_w) - 1.0
            var base = (py * num_patches_w + px) * half
            for i in range(quarter):
                var ay = (two_pi * cy) * inv_freq[i]
                var ax = (two_pi * cx) * inv_freq[i]
                cos_t.data[base + i] = cos(ay)
                sin_t.data[base + i] = sin(ay)
                cos_t.data[base + quarter + i] = cos(ax)
                sin_t.data[base + quarter + i] = sin(ax)
    return (cos_t^, sin_t^)


def _apply_rope_half(mut t: Tensor[F32], cos_t: Tensor[F32], sin_t: Tensor[F32], n_prefix: Int) raises:
    """q' = q*cos + rotate_half(q)*sin on the patch tokens of t [N, L, H, D]:
    lane pair (i, i+D/2) rotates by angle i (cos/sin [P, D/2]). Prefix
    tokens (cls + registers) are untouched."""
    var n = t.shape[0]
    var l = t.shape[1]
    var h = t.shape[2]
    var d = t.shape[3]
    var half = d // 2
    var p = cos_t.shape[0]
    if n_prefix + p != l:
        raise Error("rope: token count mismatch")
    for b in range(n):
        for j in range(p):
            var abase = j * half
            for hh in range(h):
                var base = ((b * l + n_prefix + j) * h + hh) * d
                for i in range(half):
                    var c = cos_t.data[abase + i]
                    var s = sin_t.data[abase + i]
                    var x1 = t.data[base + i]
                    var x2 = t.data[base + half + i]
                    t.data[base + i] = x1 * c - x2 * s
                    t.data[base + half + i] = x2 * c + x1 * s


struct Dinov3Attention(Copyable, Movable):
    var num_heads: Int
    var head_dim: Int
    var q_proj: SparseLinear
    var k_proj: SparseLinear  # has_bias=False (config key_bias)
    var v_proj: SparseLinear
    var o_proj: SparseLinear

    def __init__(
        out self,
        num_heads: Int,
        head_dim: Int,
        var q_proj: SparseLinear,
        var k_proj: SparseLinear,
        var v_proj: SparseLinear,
        var o_proj: SparseLinear,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = q_proj^
        self.k_proj = k_proj^
        self.v_proj = v_proj^
        self.o_proj = o_proj^

    def forward(
        self, x: Tensor[F32], cos_t: Tensor[F32], sin_t: Tensor[F32], n_prefix: Int
    ) raises -> Tensor[F32]:
        var n = x.shape[0]
        var l = x.shape[1]
        var hd_shape: List[Int] = [n, l, self.num_heads, self.head_dim]
        var q = Tensor[F32].from_values(hd_shape, self.q_proj.forward(x).data)
        var k = Tensor[F32].from_values(hd_shape, self.k_proj.forward(x).data)
        var v = Tensor[F32].from_values(hd_shape, self.v_proj.forward(x).data)
        _apply_rope_half(q, cos_t, sin_t, n_prefix)
        _apply_rope_half(k, cos_t, sin_t, n_prefix)
        var h = dense_sdpa_q_k_v(q, k, v)
        var out_shape: List[Int] = [n, l, self.num_heads * self.head_dim]
        return self.o_proj.forward(Tensor[F32].from_values(out_shape, h.data))


struct Dinov3Layer(Copyable, Movable):
    var norm1: LayerNorm32
    var attention: Dinov3Attention
    var layer_scale1: Tensor[F32]  # lambda1 [C]
    var norm2: LayerNorm32
    var mlp_up: SparseLinear
    var mlp_down: SparseLinear
    var layer_scale2: Tensor[F32]

    def __init__(
        out self,
        var norm1: LayerNorm32,
        var attention: Dinov3Attention,
        var layer_scale1: Tensor[F32],
        var norm2: LayerNorm32,
        var mlp_up: SparseLinear,
        var mlp_down: SparseLinear,
        var layer_scale2: Tensor[F32],
    ):
        self.norm1 = norm1^
        self.attention = attention^
        self.layer_scale1 = layer_scale1^
        self.norm2 = norm2^
        self.mlp_up = mlp_up^
        self.mlp_down = mlp_down^
        self.layer_scale2 = layer_scale2^

    def forward(
        self, x: Tensor[F32], cos_t: Tensor[F32], sin_t: Tensor[F32], n_prefix: Int
    ) raises -> Tensor[F32]:
        var h = self.attention.forward(self.norm1.forward(x), cos_t, sin_t, n_prefix)
        h = _scale_channels(h, self.layer_scale1)._binop_flat(x, OP_ADD)
        var m = self.mlp_down.forward(activation(self.mlp_up.forward(self.norm2.forward(h)), ACT_GELU))
        return _scale_channels(m, self.layer_scale2)._binop_flat(h, OP_ADD)


struct Dinov3ViT(Copyable, Movable):
    var hidden: Int
    var num_heads: Int
    var head_dim: Int
    var patch_size: Int
    var num_registers: Int
    var rope_theta: Float64
    var cls_token: Tensor[F32]        # [1, C]
    var register_tokens: Tensor[F32]  # [R, C]
    var patch_weight: Tensor[F32]     # [C, Cin*p*p] (conv weight flattened)
    var patch_bias: Tensor[F32]       # [C]
    var layers: List[Dinov3Layer]
    var final_norm: LayerNorm32       # extractor's F.layer_norm: NO affine

    def __init__(
        out self,
        hidden: Int,
        num_heads: Int,
        patch_size: Int,
        num_registers: Int,
        rope_theta: Float64,
        var cls_token: Tensor[F32],
        var register_tokens: Tensor[F32],
        var patch_weight: Tensor[F32],
        var patch_bias: Tensor[F32],
        var layers: List[Dinov3Layer],
        eps: Float64 = 1e-5,
    ) raises:
        if hidden % num_heads != 0:
            raise Error("Dinov3ViT: hidden % num_heads != 0")
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.patch_size = patch_size
        self.num_registers = num_registers
        self.rope_theta = rope_theta
        self.cls_token = cls_token^
        self.register_tokens = register_tokens^
        self.patch_weight = patch_weight^
        self.patch_bias = patch_bias^
        self.layers = layers^
        self.final_norm = LayerNorm32(hidden, eps, affine=False)

    def forward(self, pixels: Tensor[F32]) raises -> Tensor[F32]:
        """Normalized pixels [1, Cin, H, W] -> features [1, 1+R+P, C]."""
        if pixels.ndim() != 4 or pixels.shape[0] != 1:
            raise Error("Dinov3ViT: expected pixels [1, Cin, H, W]")
        var c_in = pixels.shape[1]
        var ih = pixels.shape[2]
        var iw = pixels.shape[3]
        var p = self.patch_size
        if ih % p != 0 or iw % p != 0:
            raise Error("Dinov3ViT: image size not a multiple of patch size")
        var nh = ih // p
        var nw = iw // p
        var n_patches = nh * nw

        # patch embedding: im2col in (c, i, j) order + linear
        var cols = Tensor[F32]([n_patches, c_in * p * p])
        for py in range(nh):
            for px in range(nw):
                var row = (py * nw + px) * c_in * p * p
                for cc in range(c_in):
                    for i in range(p):
                        var src = (cc * ih + py * p + i) * iw + px * p
                        var dst = row + (cc * p + i) * p
                        for j in range(p):
                            cols.data[dst + j] = pixels.data[src + j]
        var patches = linear(cols, self.patch_weight, self.patch_bias)

        var parts = List[Tensor[F32]]()
        parts.append(self.cls_token.copy())
        parts.append(self.register_tokens.copy())
        parts.append(patches^)
        var tokens = Tensor[F32].cat_rows(parts)
        var l = tokens.rows()
        var h = Tensor[F32].from_values([1, l, self.hidden], tokens.data)

        var phases = rope_cos_sin(nh, nw, self.head_dim, self.rope_theta)
        var n_prefix = 1 + self.num_registers
        for i in range(len(self.layers)):
            h = self.layers[i].forward(h, phases[0], phases[1], n_prefix)
        return self.final_norm.forward(h)


def _lin_from(sd: StateDict, prefix: String, out_features: Int, has_bias: Bool) raises -> SparseLinear:
    var w = sd.tensor(prefix + ".weight")
    var sl: SparseLinear
    if has_bias:
        sl = SparseLinear(w^, sd.tensor(prefix + ".bias"))
    else:
        sl = SparseLinear(w^, Tensor[F32]([out_features]), has_bias=False)
    sl.gpu = GpuLinear.try_build(sd.gpu, sl.weight, sl.bias, sl.has_bias)
    return sl^


def _ln_affine_from(sd: StateDict, prefix: String, channels: Int, eps: Float64) raises -> LayerNorm32:
    var ln = LayerNorm32(channels, eps, affine=True)
    ln.weight = sd.tensor(prefix + ".weight")
    ln.bias = sd.tensor(prefix + ".bias")
    return ln^


def dinov3_layer_from(
    sd: StateDict, prefix: String, hidden: Int, num_heads: Int, eps: Float64
) raises -> Dinov3Layer:
    var attn = Dinov3Attention(
        num_heads,
        hidden // num_heads,
        _lin_from(sd, prefix + ".attention.q_proj", hidden, True),
        _lin_from(sd, prefix + ".attention.k_proj", hidden, False),
        _lin_from(sd, prefix + ".attention.v_proj", hidden, True),
        _lin_from(sd, prefix + ".attention.o_proj", hidden, True),
    )
    return Dinov3Layer(
        _ln_affine_from(sd, prefix + ".norm1", hidden, eps),
        attn^,
        sd.tensor(prefix + ".layer_scale1.lambda1"),
        _ln_affine_from(sd, prefix + ".norm2", hidden, eps),
        _lin_from(sd, prefix + ".mlp.up_proj", 0, True),
        _lin_from(sd, prefix + ".mlp.down_proj", 0, True),
        sd.tensor(prefix + ".layer_scale2.lambda1"),
    )


def dinov3_from(
    sd: StateDict,
    num_layers: Int,
    hidden: Int,
    num_heads: Int,
    patch_size: Int,
    num_registers: Int,
    rope_theta: Float64,
    eps: Float64 = 1e-5,
) raises -> Dinov3ViT:
    """Build the ViT from HF-named weights (transformers state_dict or the
    WP12 safetensors dict). mask_token and norm.{weight,bias} are unused at
    inference (see file header) and never read."""
    var cls_raw = sd.tensor("embeddings.cls_token")        # [1, 1, C]
    var reg_raw = sd.tensor("embeddings.register_tokens")  # [1, R, C]
    var pw = sd.tensor("embeddings.patch_embeddings.weight")  # [C, Cin, p, p]
    var layers = List[Dinov3Layer]()
    for i in range(num_layers):
        layers.append(dinov3_layer_from(sd, "layer." + String(i), hidden, num_heads, eps))
    return Dinov3ViT(
        hidden,
        num_heads,
        patch_size,
        num_registers,
        rope_theta,
        Tensor[F32].from_values([1, hidden], cls_raw.data),
        Tensor[F32].from_values([num_registers, hidden], reg_raw.data),
        pw.reshape_rows([pw.numel() // pw.shape[0]]),
        sd.tensor("embeddings.patch_embeddings.bias"),
        layers^,
        eps,
    )
