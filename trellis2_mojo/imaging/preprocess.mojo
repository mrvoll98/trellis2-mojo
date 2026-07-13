# Pure-Mojo port of the conditioning image prep (WP14) — the alpha branch
# of Trellis2ImageTo3DPipeline.preprocess_image plus the extractor's pixel
# normalization, mirroring cond_io.py / the upstream numerics EXACTLY
# (every cast and rounding mode matters for bit-parity):
#   - RGBA with a real alpha channel required (ADR 0007 — no rembg path)
#   - >1024 images downscale by scale = 1024/max_size, new size
#     int(w*scale) (C truncation), PIL-exact Lanczos
#   - alpha bbox: alpha > 0.8*255 (f64 204.000..03, i.e. alpha >= 205)
#   - square crop box around the bbox center: center +/- size//2 as
#     FLOATS, then PIL crop's round-half-even -> int; crop pads with 0
#     outside the image
#   - premultiply in f32 (/255, rgb*alpha, *255) and truncate-to-u8
#     (numpy astype(uint8))
#   - cond_pixels: Lanczos to resolution^2, /255, ImageNet normalize in
#     f32, channel-planar [1, 3, R, R]

from std.math import floor

from trellis2_mojo.io.image import ImageU8
from trellis2_mojo.imaging.resize import resize_lanczos
from trellis2_mojo.sparse.tensor import Tensor

comptime F32 = DType.float32


def _round_half_even(x: Float64) raises -> Int:
    """Python round() (PIL crop applies it to the box coordinates)."""
    var f = floor(x)
    var d = x - f
    if d < 0.5:
        return Int(f)
    if d > 0.5:
        return Int(f) + 1
    var i = Int(f)
    if i % 2 == 0:
        return i
    return i + 1


def _crop_pad(img: ImageU8, x0: Int, y0: Int, x1: Int, y1: Int) raises -> ImageU8:
    """PIL crop: region [x0,x1) x [y0,y1), zero-filled outside the image."""
    var out = ImageU8(x1 - x0, y1 - y0, img.channels)
    for y in range(y0, y1):
        if y < 0 or y >= img.height:
            continue
        for x in range(max(x0, 0), min(x1, img.width)):
            for cc in range(img.channels):
                out.data[((y - y0) * out.width + (x - x0)) * img.channels + cc] = img.data[
                    (y * img.width + x) * img.channels + cc
                ]
    return out^


def preprocess_rgba(img: ImageU8) raises -> ImageU8:
    """RGBA in -> premultiplied RGB u8 out (square, object-centered)."""
    if img.channels != 4:
        raise Error(
            "input must be RGBA with a real alpha channel (background removed); "
            "the rembg path is not supported per ADR 0007"
        )
    var all_opaque = True
    for i in range(img.width * img.height):
        if img.data[i * 4 + 3] != 255:
            all_opaque = False
            break
    if all_opaque:
        raise Error(
            "input must be RGBA with a real alpha channel (background removed); "
            "the rembg path is not supported per ADR 0007"
        )

    var work = img.copy()
    var max_size = max(work.width, work.height)
    if max_size > 1024:
        var scale = 1024.0 / Float64(max_size)
        work = resize_lanczos(
            work, Int(Float64(work.width) * scale), Int(Float64(work.height) * scale)
        )

    # alpha bbox (>= 205, see header); upstream would crash on an empty
    # bbox (np.min of argwhere) — raise cleanly instead
    var x0 = work.width
    var y0 = work.height
    var x1 = -1
    var y1 = -1
    for y in range(work.height):
        for x in range(work.width):
            if work.data[(y * work.width + x) * 4 + 3] > 204:
                if x < x0:
                    x0 = x
                if x > x1:
                    x1 = x
                if y < y0:
                    y0 = y
                if y > y1:
                    y1 = y
    if x1 < 0:
        raise Error("preprocess: no pixels with alpha > 0.8 — nothing to crop to")

    var cx = Float64(x0 + x1) / 2.0
    var cy = Float64(y0 + y1) / 2.0
    var size = max(x1 - x0, y1 - y0)
    var half = size // 2
    var crop = _crop_pad(
        work,
        _round_half_even(cx - Float64(half)),
        _round_half_even(cy - Float64(half)),
        _round_half_even(cx + Float64(half)),
        _round_half_even(cy + Float64(half)),
    )

    # premultiply in f32 and truncate back to u8, as numpy does
    var out = ImageU8(crop.width, crop.height, 3)
    for i in range(crop.width * crop.height):
        var a = Float32(crop.data[i * 4 + 3]) / Float32(255)
        for cc in range(3):
            var v = (Float32(crop.data[i * 4 + cc]) / Float32(255)) * a
            out.data[i * 3 + cc] = UInt8(Int(v * Float32(255)))
    return out^


def cond_pixels(rgb: ImageU8, resolution: Int) raises -> Tensor[F32]:
    """Premultiplied RGB -> normalized pixels [1, 3, R, R] f32 (mirrors
    cond_io.pixels: Lanczos resize, /255, ImageNet normalize)."""
    if rgb.channels != 3:
        raise Error("cond_pixels: expected RGB")
    var r = resize_lanczos(rgb, resolution, resolution)
    var mean = List[Float32]()
    mean.append(0.485)
    mean.append(0.456)
    mean.append(0.406)
    var std = List[Float32]()
    std.append(0.229)
    std.append(0.224)
    std.append(0.225)
    var t = Tensor[F32]([1, 3, resolution, resolution])
    for cc in range(3):
        var plane = cc * resolution * resolution
        for i in range(resolution * resolution):
            var v = Float32(r.data[i * 3 + cc]) / Float32(255)
            t.data[plane + i] = (v - mean[cc]) / std[cc]
    return t^
