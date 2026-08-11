# Deterministic, pure-Mojo random number generator for inference noise.
#
# xorshift64* is seeded through splitmix64 so nearby CLI seeds produce
# unrelated streams. Standard-normal values use Box-Muller. The stream is
# deliberately project-owned: upgrading an external framework can no longer
# change the generated geometry for a given seed.

from std.math import cos, log, sqrt

from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32
comptime TWO_PI: Float64 = 6.283185307179586
comptime INV_2_POW_53: Float64 = 1.0 / 9007199254740992.0


struct Rng(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        var z = seed + 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        self.state = z ^ (z >> 31)
        if self.state == 0:
            self.state = 0x9E3779B97F4A7C15

    def next_u64(mut self) -> UInt64:
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 0x2545F4914F6CDD1D

    def next_f64(mut self) -> Float64:
        return Float64(self.next_u64() >> 11) * INV_2_POW_53

    def normal(mut self) -> Float64:
        var u1 = self.next_f64()
        while u1 == 0.0:
            u1 = self.next_f64()
        var u2 = self.next_f64()
        return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)


def randn(mut rng: Rng, var shape: List[Int]) raises -> Tensor[F32]:
    """Return a contiguous f32 standard-normal tensor and advance rng."""
    var out = Tensor[F32](shape^)
    for i in range(len(out.data)):
        out.data[i] = Float32(rng.normal())
    return out^
