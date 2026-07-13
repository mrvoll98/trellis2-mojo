# WP12 IO parity: the pure-Mojo loading path (trellis2_mojo/io/ — JSON
# parser, HF-cache resolution, safetensors reader with bf16/f16->f32
# casts) against ckpt_io.py (torch + safetensors) on ALL 8 pipeline
# checkpoints. Values must be BIT-identical (compared as int32 views), key
# sets equal, paths equal, and the pipeline.json floats (sampler params +
# slat normalizations) must parse to exactly the same f64 as Python's
# json module.
#
# Reads ~14 GB of checkpoints per run (each side loads its own copy), so
# this is its own task and NOT part of test-all:
#   pixi run test-io
#
# Run from repo root: pixi run mojo run -I . tests/parity/parity_io_vs_ckpt_io.mojo

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_to_torch
from trellis2_mojo.io.hf_cache import ckpt_base, model_path, pipeline_config_json
from trellis2_mojo.io.json import parse_json_str
from trellis2_mojo.io.safetensors import load_safetensors_f32

comptime F32 = DType.float32

comptime MODEL_KEYS = 8


def model_key(i: Int) raises -> String:
    if i == 0:
        return "sparse_structure_decoder"
    if i == 1:
        return "sparse_structure_flow_model"
    if i == 2:
        return "shape_slat_decoder"
    if i == 3:
        return "tex_slat_decoder"
    if i == 4:
        return "shape_slat_flow_model_512"
    if i == 5:
        return "tex_slat_flow_model_512"
    if i == 6:
        return "shape_slat_flow_model_1024"
    return "tex_slat_flow_model_1024"


def check_state_dict(io: PythonObject, torch: PythonObject, key: String) raises:
    var name_mojo = model_path(key)
    var name_py = String(py=io.model_path(key))
    if name_mojo != name_py:
        raise Error("model_path mismatch for " + key + ": " + name_mojo + " vs " + name_py)
    var base_mojo = ckpt_base(name_mojo)
    var base_py = String(py=io.ckpt_base(name_py))
    if base_mojo != base_py:
        raise Error("ckpt_base mismatch for " + key)

    var sd = load_safetensors_f32(base_mojo + ".safetensors")
    var refd = io.load_state_dict_f32(name_py)
    if len(sd) != Int(py=refd.__len__()):
        raise Error("key count mismatch for " + key)
    var n_tensors = 0
    var n_values = 0
    for item in refd.items():
        var k = String(py=item[0])
        var rt = item[1]
        if k not in sd:
            raise Error("missing key " + k + " in " + key)
        var mt = tensor_to_torch(sd[k])
        if Int(py=rt.numel()) != Int(py=mt.numel()):
            raise Error("numel mismatch: " + key + "/" + k)
        # shape check + bitwise value check via int32 views
        for d in range(len(sd[k].shape)):
            if sd[k].shape[d] != Int(py=rt.shape[d]):
                raise Error("shape mismatch: " + key + "/" + k)
        if not Bool(py=torch.equal(
            mt.reshape(rt.shape).view(torch.int32), rt.view(torch.int32)
        )):
            raise Error("bit mismatch: " + key + "/" + k)
        n_tensors += 1
        n_values += Int(py=rt.numel())
    print("  " + key + ": " + String(n_tensors) + " tensors / "
          + String(n_values) + " values bit-identical")


def check_pipeline_json(io: PythonObject) raises:
    """Every float in the sampler params + normalizations must parse to the
    exact f64 Python's json produces (exercises the number path)."""
    var json_mod = Python.import_module("json")
    var doc = pipeline_config_json()
    var args = doc.obj_get(doc.root, "args")
    var pyargs = io.pipeline_config()["args"]

    var stages: List[String] = ["sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"]
    var fields: List[String] = ["steps", "guidance_strength", "guidance_rescale", "rescale_t"]
    for stage in stages:
        var p = doc.obj_get(doc.obj_get(args, stage), "params")
        var pp = pyargs[stage]["params"]
        for f in fields:
            if doc.get_float(doc.obj_get(p, f)) != Float64(py=pp[f]):
                raise Error("pipeline.json float mismatch: " + stage + "." + f)
        var iv = doc.obj_get(p, "guidance_interval")
        for i in range(2):
            if doc.get_float(doc.arr_at(iv, i)) != Float64(py=pp["guidance_interval"][i]):
                raise Error("pipeline.json interval mismatch: " + stage)
        var sm = doc.obj_get(doc.obj_get(doc.obj_get(args, stage), "args"), "sigma_min")
        if doc.get_float(sm) != Float64(py=pyargs[stage]["args"]["sigma_min"]):
            raise Error("pipeline.json sigma_min mismatch: " + stage)

    var norms: List[String] = ["shape_slat_normalization", "tex_slat_normalization"]
    var checked = 0
    for nk in norms:
        var node = doc.obj_get(args, nk)
        for fk in ["mean", "std"]:
            var arr = doc.obj_get(node, fk)
            var pyarr = pyargs[nk][fk]
            if doc.arr_len(arr) != Int(py=pyarr.__len__()):
                raise Error("normalization length mismatch: " + nk)
            for i in range(doc.arr_len(arr)):
                if doc.get_float(doc.arr_at(arr, i)) != Float64(py=pyarr[i]):
                    raise Error("normalization float mismatch: " + nk + "." + fk)
                checked += 1
    print("  pipeline.json: sampler params + " + String(checked) + " normalization floats exact")


def check_json_units() raises:
    """Escape/number/structure edge cases against expected values."""
    var doc = parse_json_str(
        '{"s": "a\\"b\\\\c\\/d\\n\\t\\u00e6\\ud83d\\ude00", "neg": -0.0625, '
        '"big": 123456789012345, "exp": 2.5e3, "nexp": 25e-4, '
        '"arr": [[1], {"k": false}], "empty_o": {}, "empty_a": []}'
    )
    var s = doc.get_str(doc.obj_get(doc.root, "s"))
    var py = Python.import_module("json").loads(
        '{"s": "a\\"b\\\\c\\/d\\n\\t\\u00e6\\ud83d\\ude00"}'
    )
    if s != String(py=py["s"]):
        raise Error("json unit: escape mismatch")
    if doc.get_float(doc.obj_get(doc.root, "neg")) != -0.0625:
        raise Error("json unit: neg")
    if doc.get_int(doc.obj_get(doc.root, "big")) != 123456789012345:
        raise Error("json unit: big int")
    if doc.get_float(doc.obj_get(doc.root, "exp")) != 2500.0:
        raise Error("json unit: exp")
    if doc.get_float(doc.obj_get(doc.root, "nexp")) != 0.0025:
        raise Error("json unit: nexp")
    var arr = doc.obj_get(doc.root, "arr")
    if doc.arr_len(arr) != 2 or doc.get_int(doc.arr_at(doc.arr_at(arr, 0), 0)) != 1:
        raise Error("json unit: nesting")
    if doc.get_bool(doc.obj_get(doc.arr_at(arr, 1), "k")):
        raise Error("json unit: bool")
    if doc.arr_len(doc.obj_get(doc.root, "empty_a")) != 0:
        raise Error("json unit: empty array")
    print("  json units: escapes (incl. surrogate pair), numbers, nesting OK")


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var io = Python.import_module("trellis2_mojo.ckpt_io")
    var torch = Python.import_module("torch")

    check_json_units()
    check_pipeline_json(io)
    for i in range(MODEL_KEYS):
        check_state_dict(io, torch, model_key(i))
    print("io parity vs ckpt_io: json units + pipeline.json + 8 checkpoints bit-identical")
