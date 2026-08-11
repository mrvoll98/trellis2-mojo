# Pure-Mojo StateDict facade (WP12): the single state-dict type every
# loader takes. Weights come from the native safetensors reader.

from trellis2_mojo.gpu.linear import GpuContext
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


struct StateDict(Copyable, Movable):
    var d: Dict[String, Tensor[F32]]
    # WP11: set by the checkpoint loaders when TRELLIS2_GPU=1 — rides along
    # so lin_from can upload device weights without any loader signature
    # changing; None (CPU) everywhere else, incl. all parity tests
    var gpu: Optional[GpuContext]

    def __init__(out self, var d: Dict[String, Tensor[F32]]):
        self.d = d^
        self.gpu = None

    def tensor(self, key: String) raises -> Tensor[F32]:
        if key in self.d:
            return self.d[key].copy()
        raise Error("StateDict: missing key " + key)
