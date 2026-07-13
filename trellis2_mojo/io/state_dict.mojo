# StateDict facade (WP12): the single state-dict type every loader takes.
# Wraps EITHER a torch state_dict (parity tests, via interop) or the
# pure-Mojo Dict from safetensors.mojo (runner path). The @implicit
# PythonObject constructor keeps every existing call site — the parity
# tests pass torch dicts straight in — compiling unchanged.

from std.python import PythonObject

from trellis2_mojo.gpu.linear import GpuContext
from trellis2_mojo.interop import tensor_from_torch
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


struct StateDict(Copyable, Movable):
    var is_py: Bool
    var py: PythonObject
    var d: Dict[String, Tensor[F32]]
    # WP11: set by the checkpoint loaders when TRELLIS2_GPU=1 — rides along
    # so lin_from can upload device weights without any loader signature
    # changing; None (CPU) everywhere else, incl. all parity tests
    var gpu: Optional[GpuContext]

    @implicit
    def __init__(out self, sd: PythonObject):
        self.is_py = True
        self.py = sd
        self.d = Dict[String, Tensor[F32]]()
        self.gpu = None

    def __init__(out self, var d: Dict[String, Tensor[F32]]):
        self.is_py = False
        self.py = PythonObject(None)
        self.d = d^
        self.gpu = None

    def tensor(self, key: String) raises -> Tensor[F32]:
        if self.is_py:
            return tensor_from_torch(self.py[key])
        if key in self.d:
            return self.d[key].copy()
        raise Error("StateDict: missing key " + key)
