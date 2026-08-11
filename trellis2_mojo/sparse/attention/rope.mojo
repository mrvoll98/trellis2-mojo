# Mojo port of trellis2/modules/sparse/attention/rope.py
# (SparseRotaryPositionEmbedder).
#
# The original works in complex numbers: phases = polar(1, coord * freq),
# then x viewed as complex pairs is multiplied by the phases. Here phases
# are stored as an interleaved (cos, sin) tensor [N, head_dim/2, 2] and the
# complex multiply is written out in real arithmetic. Frequencies are
# computed in float32 to preserve the source model's dtype behavior. Phases are cached
# in the spatial cache under the same name scheme as the original.

from max.algorithm import parallelize
from std.math import cos, sin

from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.sparse.basic import SparseTensor, CacheValue

comptime F32 = DType.float32


struct SparseRotaryPositionEmbedder(Copyable, Movable):
    var head_dim: Int
    var dim: Int
    var freq_lo: Float64
    var freq_hi: Float64
    var freq_dim: Int
    var freqs: List[Float32]

    def __init__(out self, head_dim: Int, dim: Int = 3, freq_lo: Float64 = 1.0, freq_hi: Float64 = 10000.0) raises:
        if head_dim % 2 != 0:
            raise Error("SparseRotaryPositionEmbedder: head_dim must be even")
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

    def _cache_name(self) raises -> String:
        return (
            "rope_phase_" + String(self.dim) + "d_freq" + String(self.freq_lo)
            + "-" + String(self.freq_hi) + "_hd" + String(self.head_dim)
        )

    def _compute_phases(self, x: SparseTensor[F32]) raises -> Tensor[F32]:
        """-> [N, head_dim/2, 2] interleaved (cos, sin); slots beyond
        dim*freq_dim are padded with unit phases (angle 0)."""
        var n = x.coords.rows
        var half = self.head_dim // 2
        var phases = Tensor[F32]([n, half, 2])
        for r in range(n):
            for slot in range(half):
                var c: Float32 = 1
                var s: Float32 = 0
                if slot < self.dim * self.freq_dim:
                    var axis = slot // self.freq_dim
                    var j = slot % self.freq_dim
                    var angle = Float32(x.coords.at(r, axis + 1)) * self.freqs[j]
                    c = cos(angle)
                    s = sin(angle)
                phases.data[(r * half + slot) * 2] = c
                phases.data[(r * half + slot) * 2 + 1] = s
        return phases^

    def _phases(self, x: SparseTensor[F32]) raises -> Tensor[F32]:
        var cached = x.get_spatial_cache(self._cache_name())
        if cached:
            return cached.value().floats[0].copy()
        var phases = self._compute_phases(x)
        x.register_spatial_cache(self._cache_name(), CacheValue.from_tensor(phases))
        return phases^

    def _rotate(self, feats: Tensor[F32], phases: Tensor[F32]) raises -> Tensor[F32]:
        """feats [N, H, D] with D = head_dim; adjacent pairs are (re, im),
        rotated by phases[n, p] broadcast over heads. SIMD (WP10 pass 6):
        pairs and phases are deinterleaved into (re, im)/(cos, sin) lanes,
        rotated with the exact per-pair formula of the scalar loop
        (bit-identical), and reinterleaved; row chunks are parallelized
        for large inputs."""
        comptime W = 8
        comptime RC = 64
        var d = feats.shape[len(feats.shape) - 1]
        if d != self.head_dim:
            raise Error("rope: feats last dim != head_dim")
        var half = d // 2
        var n_rows = feats.shape[0]
        var h = feats.numel() // (n_rows * d)
        var out = Tensor[F32](feats.shape)
        var fp = feats.data.unsafe_ptr()
        var pp = phases.data.unsafe_ptr()
        var op = out.data.unsafe_ptr()

        @parameter
        def chunk(w: Int):
            var r1 = min((w + 1) * RC, n_rows)
            for r in range(w * RC, r1):
                var pbase = r * half * 2
                for head in range(h):
                    var base = (r * h + head) * d
                    var i = 0
                    while i + W <= d:
                        var v = fp.unsafe_load[width=W](base + i).deinterleave()
                        var ph = pp.unsafe_load[width=W](pbase + i).deinterleave()
                        var ore = v[0] * ph[0] - v[1] * ph[1]
                        var oim = v[0] * ph[1] + v[1] * ph[0]
                        op.unsafe_store(base + i, ore.interleave(oim))
                        i += W
                    while i < d:
                        var re = fp[unsafe_offset=base + i]
                        var im = fp[unsafe_offset=base + i + 1]
                        var c = pp[unsafe_offset=pbase + i]
                        var s = pp[unsafe_offset=pbase + i + 1]
                        op[unsafe_offset=base + i] = re * c - im * s
                        op[unsafe_offset=base + i + 1] = re * s + im * c
                        i += 2

        var n_chunks = (n_rows + RC - 1) // RC
        if n_rows * h * d < 1 << 17:
            for w in range(n_chunks):
                chunk(w)
        else:
            parallelize[chunk](n_chunks)
        return out^

    def embed(self, q: SparseTensor[F32], k: SparseTensor[F32]) raises -> Tuple[SparseTensor[F32], SparseTensor[F32]]:
        if q.coords.cols != self.dim + 1:
            raise Error("rope: coords must have dim+1 columns")
        var phases = self._phases(q)
        var q_emb = q.replace(self._rotate(q.vl.feats, phases))
        var k_emb = k.replace(self._rotate(k.vl.feats, phases))
        return (q_emb^, k_emb^)
