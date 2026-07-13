# Pure-Mojo safetensors reader (WP12): 8 bytes LE header length + JSON
# header {name: {dtype, shape, data_offsets}} + raw little-endian data.
# Weights are cast to f32 at load time (v1 of the port is f32-only):
# BF16 -> f32 is a u16 << 16 bit shift, F16 goes through the hardware
# DType.float16 cast, F32 copies straight through. SIMD width 8 with
# alignment=1 loads (the data section is not guaranteed aligned), same
# copy pattern as interop.mojo — bit-identical to ckpt_io.py's
# torch-based cast (verified by pixi run test-io).

from std.memory import UnsafePointer, Span

from trellis2_mojo.io.json import JsonDoc, parse_json
from trellis2_mojo.sparse.tensor import Tensor, stable_argsort

comptime F32 = DType.float32
comptime W = 8


def _convert_bf16(src: UnsafePointer[UInt8, MutAnyOrigin], dst: UnsafePointer[Scalar[F32], MutAnyOrigin], n: Int):
    var sp = src.bitcast[Scalar[DType.uint16]]()
    var i = 0
    while i + W <= n:
        var bits = sp.load[width=W, alignment=1](i).cast[DType.uint32]() << 16
        dst.store(i, SIMD[F32, W](from_bits=bits))
        i += W
    while i < n:
        var b1 = sp.load[width=1, alignment=1](i).cast[DType.uint32]() << 16
        dst.store(i, SIMD[F32, 1](from_bits=b1))
        i += 1


def _convert_f16(src: UnsafePointer[UInt8, MutAnyOrigin], dst: UnsafePointer[Scalar[F32], MutAnyOrigin], n: Int):
    var sp = src.bitcast[Scalar[DType.float16]]()
    var i = 0
    while i + W <= n:
        dst.store(i, sp.load[width=W, alignment=1](i).cast[F32]())
        i += W
    while i < n:
        dst.store(i, sp.load[width=1, alignment=1](i).cast[F32]())
        i += 1


def _convert_f32(src: UnsafePointer[UInt8, MutAnyOrigin], dst: UnsafePointer[Scalar[F32], MutAnyOrigin], n: Int):
    var sp = src.bitcast[Scalar[F32]]()
    var i = 0
    while i + W <= n:
        dst.store(i, sp.load[width=W, alignment=1](i))
        i += W
    while i < n:
        dst.store(i, sp.load[width=1, alignment=1](i))
        i += 1


def load_safetensors_f32(path: String) raises -> Dict[String, Tensor[F32]]:
    """Read every tensor in the file as a contiguous f32 Tensor.

    Tensors are read one at a time in data-offset order (strictly
    sequential IO): macOS caps a single read() at 2 GiB, so the >2 GB DiT
    files cannot be slurped whole — and per-tensor reads keep the peak at
    dict + one raw chunk instead of dict + whole file."""
    var fh = open(path, "r")
    var head = fh.read_bytes(8)
    if len(head) != 8:
        raise Error("safetensors: truncated header length in " + path)
    var hlen = 0
    for i in range(8):
        hlen |= Int(head[i]) << (8 * i)
    var hbytes = fh.read_bytes(hlen)
    if len(hbytes) != hlen:
        raise Error("safetensors: truncated header in " + path)
    var doc = parse_json(hbytes)

    # collect entries, then order by start offset for sequential reads
    var names = List[String]()
    var starts = List[Int]()
    for name in doc.obj_keys(doc.root):
        if name == "__metadata__":
            continue
        var node = doc.obj_get(doc.root, name)
        names.append(name.copy())
        starts.append(doc.get_int(doc.arr_at(doc.obj_get(node, "data_offsets"), 0)))
    var order = stable_argsort(starts)

    var out = Dict[String, Tensor[F32]]()
    var cur = 0
    for oi in range(len(order)):
        var name = names[order[oi]].copy()
        var node = doc.obj_get(doc.root, name)
        var dt = doc.get_str(doc.obj_get(node, "dtype"))
        var shape_node = doc.obj_get(node, "shape")
        var shape = List[Int]()
        for i in range(doc.arr_len(shape_node)):
            shape.append(doc.get_int(doc.arr_at(shape_node, i)))
        var offs = doc.obj_get(node, "data_offsets")
        var start = doc.get_int(doc.arr_at(offs, 0))
        var end = doc.get_int(doc.arr_at(offs, 1))
        if start != cur or end < start:
            raise Error("safetensors: non-contiguous data_offsets for " + name)

        var t = Tensor[F32](shape)
        var n = t.numel()
        var esize: Int
        if dt == "BF16" or dt == "F16":
            esize = 2
        elif dt == "F32":
            esize = 4
        else:
            raise Error("safetensors: unsupported dtype " + dt + " for " + name)
        if end - start != n * esize:
            raise Error("safetensors: size mismatch for " + name)

        var raw = fh.read_bytes(end - start)
        if len(raw) != end - start:
            raise Error("safetensors: truncated data for " + name)
        cur = end
        var src = raw.unsafe_ptr()
        var dst = t.data.unsafe_ptr()
        if dt == "BF16":
            _convert_bf16(src, dst, n)
        elif dt == "F16":
            _convert_f16(src, dst, n)
        else:
            _convert_f32(src, dst, n)
        _ = len(raw)  # keep the chunk alive through the raw-pointer copy
        out[name^] = t^
    fh.close()
    return out^
