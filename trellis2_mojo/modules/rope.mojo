# Mojo port of modules/attention/rope.py: RotaryPositionEmbedder for the
# dense path. The flow model precomputes phases for the full voxel grid at
# init and passes them into every block, so this embedder is stateless
# per-call (no spatial cache, unlike the sparse variant).
#
# Phases are stored as an interleaved (cos, sin) tensor [L, head_dim/2, 2],
# same layout as sparse/attention/rope.mojo; slots beyond dim * freq_dim are
# unit phases (polar(1, 0) padding in the source model).

from std.math import cos, sin

from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


struct RotaryPositionEmbedder(Copyable, Movable):
    var head_dim: Int
    var dim: Int
    var freq_lo: Float64
    var freq_hi: Float64
    var freq_dim: Int
    var freqs: List[Float32]

    def __init__(out self, head_dim: Int, dim: Int = 3, freq_lo: Float64 = 1.0, freq_hi: Float64 = 10000.0) raises:
        if head_dim % 2 != 0:
            raise Error("RotaryPositionEmbedder: head_dim must be even")
        self.head_dim = head_dim
        self.dim = dim
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.freq_dim = head_dim // 2 // dim
        self.freqs = List[Float32]()
        # freqs = freq_lo / freq_hi ** (arange(freq_dim) / freq_dim), in f32
        for j in range(self.freq_dim):
            var e = Float32(j) / Float32(self.freq_dim)
            self.freqs.append(Float32(self.freq_lo) / Float32(self.freq_hi) ** e)

    def forward(self, positions: Tensor[F32]) raises -> Tensor[F32]:
        """positions [L, dim] -> phases [L, head_dim/2, 2] interleaved (cos, sin).
        Slot layout is axis-major per position (outer(indices.reshape(-1), freqs)
        reshaped back), identical to the sparse embedder."""
        if positions.shape[len(positions.shape) - 1] != self.dim:
            raise Error("RotaryPositionEmbedder: last dim of positions must be dim")
        var l = positions.shape[0]
        var half = self.head_dim // 2
        var out_shape: List[Int] = [l, half, 2]
        var phases = Tensor[F32](out_shape)
        for r in range(l):
            for slot in range(half):
                var c: Float32 = 1
                var s: Float32 = 0
                if slot < self.dim * self.freq_dim:
                    var axis = slot // self.freq_dim
                    var j = slot % self.freq_dim
                    var angle = positions.data[r * self.dim + axis] * self.freqs[j]
                    c = cos(angle)
                    s = sin(angle)
                phases.data[(r * half + slot) * 2] = c
                phases.data[(r * half + slot) * 2 + 1] = s
        return phases^


def apply_rotary_embedding(x: Tensor[F32], phases: Tensor[F32]) raises -> Tensor[F32]:
    """x [N, L, H, D] with adjacent (re, im) pairs on D, phases [L, D/2, 2];
    phases broadcast over N and H (phases.unsqueeze(-2) in the original)."""
    if len(x.shape) != 4:
        raise Error("apply_rotary_embedding: expected [N, L, H, D]")
    var n = x.shape[0]
    var l = x.shape[1]
    var h = x.shape[2]
    var d = x.shape[3]
    var half = d // 2
    if phases.shape[0] != l or phases.shape[1] != half:
        raise Error("apply_rotary_embedding: phases shape mismatch")
    var out = Tensor[F32](x.shape)
    for b in range(n):
        for jl in range(l):
            for head in range(h):
                for p in range(half):
                    var base = ((b * l + jl) * h + head) * d + 2 * p
                    var re = x.data[base]
                    var im = x.data[base + 1]
                    var c = phases.data[(jl * half + p) * 2]
                    var s = phases.data[(jl * half + p) * 2 + 1]
                    out.data[base] = re * c - im * s
                    out.data[base + 1] = re * s + im * c
    return out^
