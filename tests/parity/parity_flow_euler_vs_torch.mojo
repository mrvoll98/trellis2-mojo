# WP2 parity: Mojo FlowEuler sampler vs the original PyTorch samplers,
# full-trajectory comparison (every step's pred_x_t and pred_x_0).
#
# Two model paths are tested:
#   1. Native Mojo dummy model (pure-Mojo sampling loop).
#   2. PyVelocityModel wrapping the torch dummy model (the hybrid-pipeline
#      interop direction: Mojo loop -> Python model).
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_flow_euler_vs_torch.mojo

from std.math import sin
from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch
from trellis2_mojo.samplers.flow_euler import FlowEulerSampler, SampleResult, VelocityModel
from trellis2_mojo.samplers.py_model import PyVelocityModel
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32

comptime MODEL_A: Float32 = 0.7
comptime MODEL_B: Float32 = 1.3
comptime MODEL_C: Float32 = 0.9


struct DummyModel(Copyable, Movable, VelocityModel):
    """Native Mojo mirror of torch_ref_sampler.DummyModel, computed in
    float32 to match torch's arithmetic."""

    var cond: Tensor[F32]
    var neg_cond: Tensor[F32]

    def __init__(out self, var cond: Tensor[F32], var neg_cond: Tensor[F32]):
        self.cond = cond^
        self.neg_cond = neg_cond^

    def predict(self, x_t: Tensor[F32], t1000: Float64, use_neg_cond: Bool) raises -> Tensor[F32]:
        var t = Float32(t1000) / 1000.0
        var out = Tensor[F32](x_t.shape)
        for i in range(x_t.numel()):
            var c = self.neg_cond.data[i] if use_neg_cond else self.cond.data[i]
            out.data[i] = sin(x_t.data[i] * MODEL_A + t * MODEL_B) * MODEL_C + c
        return out^


def check_result(name: String, r: SampleResult, py: PythonObject, atol: Float64) raises:
    var steps = len(r.pred_x_t)
    if Int(py=py["x_t"].__len__()) != steps:
        raise Error("step count mismatch: " + name)
    for i in range(steps):
        check_tensor(name + ".x_t[" + String(i) + "]", r.pred_x_t[i], py["x_t"][i], atol)
        check_tensor(name + ".x_0[" + String(i) + "]", r.pred_x_0[i], py["x_0"][i], atol)
    check_tensor(name + ".samples", r.samples, py["samples"], atol)


def check_tensor(name: String, m: Tensor[F32], py: PythonObject, atol: Float64) raises:
    if Int(py=py.numel()) != m.numel():
        raise Error("size mismatch: " + name)
    var flat = py.flatten().tolist()
    for i in range(m.numel()):
        var d = Float64(py=flat[i]) - Float64(m.data[i])
        if d < 0:
            d = -d
        if d > atol:
            raise Error("value mismatch: " + name + " @ " + String(i) + " diff " + String(d))


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_sampler")

    var sigma_min = 1e-5
    var steps = 8
    var atol = 2e-4  # fp32 trajectories, diffs accumulate over steps

    for seed in range(3):
        var gen = pyref.gen(seed)
        var noise = tensor_from_torch(gen[0])
        var cond = tensor_from_torch(gen[1])
        var neg_cond = tensor_from_torch(gen[2])
        var model = DummyModel(cond.copy(), neg_cond.copy())
        var sampler = FlowEulerSampler(sigma_min)

        # plain, two t-schedules
        for variant in range(2):
            var rescale_t = 1.0 if variant == 0 else 3.0
            var expected = pyref.run_plain(gen[0], gen[1], steps, rescale_t, sigma_min)
            var r = sampler.sample(model, noise, steps, rescale_t)
            check_result("plain(rt=" + String(rescale_t) + ")", r, expected, atol)

        # CFG: strength 0 (neg only), 1 (pos only), 3.5 (mix)
        var strengths: List[Float64] = [0.0, 1.0, 3.5]
        for s in strengths:
            var expected = pyref.run_cfg(gen[0], gen[1], gen[2], steps, 1.0, sigma_min, s)
            var r = sampler.sample_cfg(model, noise, steps, 1.0, s)
            check_result("cfg(s=" + String(s) + ")", r, expected, atol)

        # guidance interval: CFG only for t in [0.3, 0.8]
        var ref_gi = pyref.run_interval(gen[0], gen[1], gen[2], steps, 1.0, sigma_min, 5.0, 0.3, 0.8)
        var r_gi = sampler.sample_cfg_interval(model, noise, steps, 1.0, 5.0, 0.3, 0.8)
        check_result("interval", r_gi, ref_gi, atol)

        # interop direction: Mojo loop driving the torch model
        var py_model = PyVelocityModel(pyref.DummyModel(), gen[1], gen[2])
        var ref_py = pyref.run_cfg(gen[0], gen[1], gen[2], steps, 1.0, sigma_min, 3.5)
        var r_py = sampler.sample_cfg(py_model, noise, steps, 1.0, 3.5)
        check_result("py-model cfg", r_py, ref_py, atol)

        # CFG rescale, dense std semantics (real ss-flow config:
        # strength 7.5, rescale 0.7, interval [0.6, 1.0], rescale_t 5.0)
        var ref_rs = pyref.run_interval_rescale(
            gen[0], gen[1], gen[2], steps, 5.0, sigma_min, 7.5, 0.6, 1.0, 0.7
        )
        var r_rs = sampler.sample_cfg_interval(model, noise, steps, 5.0, 7.5, 0.6, 1.0, 0.7)
        check_result("interval+rescale dense", r_rs, ref_rs, atol)

        # CFG rescale, VarLen std semantics (real shape-slat config:
        # strength 7.5, rescale 0.5, interval [0.6, 1.0], rescale_t 3.0);
        # uneven segment lengths so per-segment std differs from global
        var lengths = Python.list()
        lengths.append(7)
        lengths.append(11)
        var vgen = pyref.gen_varlen(seed, lengths)
        var vfeats = tensor_from_torch(vgen[0])
        var vcond = tensor_from_torch(vgen[1])
        var vneg = tensor_from_torch(vgen[2])
        var vmodel = DummyModel(vcond.copy(), vneg.copy())
        var offsets: List[Int] = [0, 7, 18]
        var ref_vrs = pyref.run_interval_rescale_varlen(
            vgen[0], vgen[1], vgen[2], vgen[3], steps, 3.0, sigma_min, 7.5, 0.6, 1.0, 0.5
        )
        var r_vrs = sampler.sample_cfg_interval(
            vmodel, vfeats, steps, 3.0, 7.5, 0.6, 1.0, 0.5, offsets
        )
        check_result("interval+rescale varlen", r_vrs, ref_vrs, atol)

    print("flow_euler parity vs torch: 3 seeds x 9 configs passed")
