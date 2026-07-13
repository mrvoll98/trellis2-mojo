# Torch <-> Mojo bridge helpers (WP1 interop harness).
# Used by parity tests and by hybrid-pipeline wrappers that call Python
# models from Mojo code.
#
# The copies go through raw data_ptr() access on contiguous CPU tensors:
# the original per-element PythonObject conversion is far too slow for
# real checkpoints (1.3B params; WP9 del 3). Values are bit-identical to
# the old path — both are exact f32 copies. The copy loops are inlined
# per direction because the source pointer's origin mutability differs
# (borrowed List vs raw address).

from std.python import Python, PythonObject
from std.memory import UnsafePointer

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32
comptime W = 8


def tensor_from_torch(py: PythonObject) raises -> Tensor[F32]:
    """Copy a torch tensor (any shape, numeric dtype) into a Tensor[F32]."""
    var torch = Python.import_module("torch")
    var tt = py.detach().to(torch.float32).contiguous()
    var nd = Int(py=tt.dim())
    var shape = List[Int]()
    for i in range(nd):
        shape.append(Int(py=tt.shape[i]))
    var t = Tensor[F32](shape)
    var n = t.numel()
    var src = UnsafePointer[Scalar[F32], MutAnyOrigin](unsafe_from_address=Int(py=tt.data_ptr()))
    var dst = t.data.unsafe_ptr()
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < n:
        dst.store(i, src.load(i))
        i += 1
    _ = Int(py=tt.numel())  # keep tt alive through the copy
    return t^


def tensor_to_torch(t: Tensor[F32]) raises -> PythonObject:
    """Copy a Tensor[F32] into a float32 torch tensor with the same shape."""
    var torch = Python.import_module("torch")
    var shape = Python.list()
    for s in t.shape:
        shape.append(s)
    var out = torch.empty(torch.Size(shape), dtype=torch.float32)
    var n = t.numel()
    var src = t.data.unsafe_ptr()
    var dst = UnsafePointer[Scalar[F32], MutAnyOrigin](unsafe_from_address=Int(py=out.data_ptr()))
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < n:
        dst.store(i, src.load(i))
        i += 1
    return out


def intmatrix_from_torch(py: PythonObject) raises -> IntMatrix:
    """Copy a 2-D integer torch tensor into an IntMatrix."""
    var torch = Python.import_module("torch")
    var tt = py.detach().to(torch.int32).contiguous()
    var rows = Int(py=tt.shape[0])
    var cols = Int(py=tt.shape[1])
    var m = IntMatrix(rows, cols)
    var n = rows * cols
    var src = UnsafePointer[Scalar[DType.int32], MutAnyOrigin](unsafe_from_address=Int(py=tt.data_ptr()))
    var dst = m.data.unsafe_ptr()
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < n:
        dst.store(i, src.load(i))
        i += 1
    _ = Int(py=tt.numel())  # keep tt alive through the copy
    return m^
