# VelocityModel backed by a Python/torch callable — the hybrid-pipeline
# direction (ADR 0001): the Mojo sampler loop drives the original torch
# model through interop. model(x_t, t_tensor, cond) like the Python
# samplers' _inference_model.

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch, tensor_to_torch
from trellis2_mojo.sparse.tensor import Tensor
from trellis2_mojo.samplers.flow_euler import VelocityModel

comptime F32 = DType.float32


struct PyVelocityModel(Copyable, Movable, VelocityModel):
    var model: PythonObject
    var cond: PythonObject
    var neg_cond: PythonObject

    def __init__(out self, model: PythonObject, cond: PythonObject, neg_cond: PythonObject):
        self.model = model
        self.cond = cond
        self.neg_cond = neg_cond

    def predict(self, x_t: Tensor[F32], t1000: Float64, use_neg_cond: Bool) raises -> Tensor[F32]:
        var torch = Python.import_module("torch")
        var x_py = tensor_to_torch(x_t)
        # t = torch.tensor([1000 * t] * N, dtype=float32) as in _inference_model
        var t_py = torch.full([x_t.rows()], t1000, dtype=torch.float32)
        var cond = self.neg_cond if use_neg_cond else self.cond
        var v = self.model(x_py, t_py, cond)
        return tensor_from_torch(v)
