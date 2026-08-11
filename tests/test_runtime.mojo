# Pure-Mojo runtime boundary tests: CLI parsing, deterministic inference RNG,
# and the native texture-voxel NPZ writer.

from std.testing import assert_almost_equal, assert_equal, assert_true

from trellis2_mojo.cli import parse_float, parse_int
from trellis2_mojo.io.npz import write_tex_voxels_npz
from trellis2_mojo.rng import Rng, randn
from trellis2_mojo.sparse.tensor import IntMatrix, Tensor

comptime F32 = DType.float32


def _u32(bytes: List[UInt8], offset: Int) -> Int:
    return (
        Int(bytes[offset])
        | (Int(bytes[offset + 1]) << 8)
        | (Int(bytes[offset + 2]) << 16)
        | (Int(bytes[offset + 3]) << 24)
    )


def test_cli() raises:
    assert_equal(parse_int("42"), 42)
    assert_equal(parse_int("-17"), -17)
    assert_almost_equal(parse_float("1.25e-2"), 0.0125, atol=1e-15)
    var rejected = False
    try:
        _ = parse_float("1.2x")
    except:
        rejected = True
    assert_true(rejected)


def test_rng() raises:
    var golden = Rng(1337)
    assert_equal(golden.next_u64(), UInt64(76748464051394776))
    assert_equal(golden.next_u64(), UInt64(17093336004959153412))
    assert_equal(golden.next_u64(), UInt64(4605377540300676664))

    var a = Rng(42)
    var b = Rng(42)
    var ta = randn(a, [2, 3, 4])
    var tb = randn(b, [2, 3, 4])
    for i in range(len(ta.data)):
        assert_equal(ta.data[i], tb.data[i])
    assert_equal(a.state, b.state)


def test_npz() raises:
    var coords = IntMatrix(2, 4)
    var coord_values: List[Int] = [0, 1, 2, 3, 0, 4, 5, 6]
    for i in range(len(coord_values)):
        coords.data[i] = Int32(coord_values[i])
    var attrs = Tensor[F32].from_values([2, 2], [0.25, -1.5, 2.0, 3.5])
    var path = String("/tmp/trellis_mojo_runtime_test.npz")
    write_tex_voxels_npz(path, coords, attrs, 512)

    var file = open(path, "r")
    var bytes = file.read_bytes()
    file.close()
    assert_true(len(bytes) > 22)
    assert_equal(_u32(bytes, 0), 0x04034B50)
    assert_equal(_u32(bytes, len(bytes) - 22), 0x06054B50)
    # Four local entries and four central-directory entries.
    assert_equal(Int(bytes[len(bytes) - 14]), 4)
    assert_equal(Int(bytes[len(bytes) - 12]), 4)


def main() raises:
    test_cli()
    test_rng()
    test_npz()
    print("pure-Mojo runtime tests passed: cli + rng + npz")
