# WP14 parity: pure-Mojo image IO + preprocess vs PIL/numpy ground truth
# (torch_ref_wp14.py). Everything here is BIT-exact:
#   - PAM/PPM readers vs the raw arrays the files were written from
#   - Lanczos resize vs PIL (the Mojo port mirrors Pillow's fixed-point
#     resampling) across down/up/non-square/single-axis/no-op, RGB + RGBA
#   - preprocess (alpha bbox, square crop w/ padding, premultiply) vs
#     cond_io.preprocess at 640 and 1500 (the >1024 downscale path)
#   - cond_pixels vs cond_io.pixels (f32, max|diff| 0.0)
#   - RGBA requirement: PPM (RGB) and all-opaque PAM are rejected
# Cache-independent and fast -> lives in test-all.
#
# Run from repo root: pixi run test-wp14

from std.python import Python, PythonObject

from trellis2_mojo.interop import tensor_from_torch
from trellis2_mojo.io.image import ImageU8, read_image
from trellis2_mojo.imaging.resize import resize_lanczos
from trellis2_mojo.imaging.preprocess import preprocess_rgba, cond_pixels
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def u8_from_torch(t: PythonObject) raises -> ImageU8:
    """F32 torch tensor [h, w, c] with 0..255 values -> ImageU8."""
    var f = tensor_from_torch(t)
    var img = ImageU8(f.shape[1], f.shape[0], f.shape[2])
    for i in range(f.numel()):
        img.data[i] = UInt8(Int(f.data[i]))
    return img^


def check_u8(name: String, img: ImageU8, expected: PythonObject) raises:
    var exp = tensor_from_torch(expected)
    if img.height != exp.shape[0] or img.width != exp.shape[1] or img.channels != exp.shape[2]:
        raise Error("shape mismatch: " + name)
    var bad = 0
    for i in range(exp.numel()):
        if Int(img.data[i]) != Int(exp.data[i]):
            bad += 1
    print("  " + name + ": " + String(exp.numel()) + " bytes, " + String(bad) + " mismatches")
    if bad != 0:
        raise Error("value mismatch: " + name)


def resize_case(pyref: PythonObject, seed: Int, w: Int, h: Int, c: Int, ow: Int, oh: Int) raises:
    var rc = pyref.resize_case(seed, w, h, c, ow, oh)
    var out = resize_lanczos(u8_from_torch(rc[0]), ow, oh)
    var name = "resize(" + String(w) + "x" + String(h) + "x" + String(c)
    name += " -> " + String(ow) + "x" + String(oh) + ")"
    check_u8(name, out, rc[1])


def expect_reject(name: String, path: String) raises:
    var rejected = False
    try:
        _ = preprocess_rgba(read_image(path))
    except:
        rejected = True
    if not rejected:
        raise Error("preprocess accepted invalid input: " + name)
    print("  reject(" + name + "): ok")


def main() raises:
    var sys_mod = Python.import_module("sys")
    var os_mod = Python.import_module("os")
    sys_mod.path.insert(0, os_mod.getcwd())
    var pyref = Python.import_module("tests.parity.torch_ref_wp14")

    # raster readers: PAM RGBA, PAM RGB + PPM RGB
    var r4 = pyref.raster_case(0, 21, 13, 4)
    check_u8("pam(rgba)", read_image(String(py=r4[0])), r4[2])
    var r3 = pyref.raster_case(1, 16, 9, 3)
    check_u8("pam(rgb)", read_image(String(py=r3[0])), r3[2])
    check_u8("ppm(rgb)", read_image(String(py=r3[1])), r3[2])

    # PIL-exact resize: down, up, non-square, single-axis, no-op
    resize_case(pyref, 1, 64, 48, 3, 32, 24)
    resize_case(pyref, 2, 40, 40, 4, 96, 80)
    resize_case(pyref, 3, 33, 57, 4, 512, 512)
    resize_case(pyref, 4, 50, 50, 3, 50, 37)
    resize_case(pyref, 5, 24, 24, 4, 24, 24)

    # preprocess: no-downscale and >1024-downscale paths, bit-exact
    var p640 = pyref.preprocess_case(640)
    check_u8("preprocess(640)", preprocess_rgba(read_image(String(py=p640[0]))), p640[1])
    var p1500 = pyref.preprocess_case(1500)
    var pre1500 = preprocess_rgba(read_image(String(py=p1500[0])))
    check_u8("preprocess(1500)", pre1500, p1500[1])

    # normalized pixels, f32 exact
    var torch = Python.import_module("torch")
    var pre640 = preprocess_rgba(read_image(String(py=p640[0])))
    for res in [128, 512]:
        var px = cond_pixels(pre640, res)
        var exp = tensor_from_torch(pyref.pixels_case(640, res))
        if px.numel() != exp.numel():
            raise Error("pixels(" + String(res) + ") size mismatch")
        var maxdiff: Float32 = 0
        for i in range(px.numel()):
            var d = px.data[i] - exp.data[i]
            if d < 0:
                d = -d
            if d > maxdiff:
                maxdiff = d
        print("  pixels(" + String(res) + "): max|diff| " + String(maxdiff))
        if maxdiff != 0:
            raise Error("pixels(" + String(res) + ") not bit-exact")

    # RGBA requirement (ADR 0007)
    expect_reject("ppm-rgb", String(py=r3[1]))
    expect_reject("opaque-rgba", String(py=pyref.opaque_pam()))
    _ = Int(py=torch.numel(pyref.pixels_case(640, 128)))

    print("wp14 imaging parity vs PIL: rasters + 5 resize + preprocess 640/1500 + pixels 128/512 + rejects passed")
