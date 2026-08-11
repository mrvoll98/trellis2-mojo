# Pure-Mojo NumPy .npz writer for the texture-voxel sidecar.
#
# The archive uses standard ZIP "store" entries (no compression), each
# containing a NumPy v1.0 .npy payload. This preserves the existing public
# output contract without importing Python, NumPy, or a tensor framework.

from std.io import FileHandle

from trellis2_mojo.sparse.tensor import IntMatrix, Tensor

comptime F32 = DType.float32
comptime ZIP_VERSION = 20
comptime ZIP_DATE_1980_01_01 = 33
comptime STREAM_BYTES = 65536


struct _ZipEntry(Copyable, Movable):
    var name: String
    var crc: UInt32
    var size: Int
    var offset: Int

    def __init__(out self, name: String, crc: UInt32, size: Int, offset: Int):
        self.name = name
        self.crc = crc
        self.size = size
        self.offset = offset


def _put_u16(mut out: List[UInt8], value: Int):
    out.append(UInt8(value & 0xFF))
    out.append(UInt8((value >> 8) & 0xFF))


def _put_u32(mut out: List[UInt8], value: Int):
    out.append(UInt8(value & 0xFF))
    out.append(UInt8((value >> 8) & 0xFF))
    out.append(UInt8((value >> 16) & 0xFF))
    out.append(UInt8((value >> 24) & 0xFF))


def _append_string(mut out: List[UInt8], value: String):
    for byte in value.as_bytes():
        out.append(byte)


def _shape_string(shape: List[Int]) -> String:
    if len(shape) == 0:
        return "()"
    var out = String("(")
    for i in range(len(shape)):
        if i > 0:
            out += ", "
        out += String(shape[i])
    if len(shape) == 1:
        out += ","
    return out + ")"


def _npy_header(descr: String, shape: List[Int]) raises -> List[UInt8]:
    var dictionary = (
        "{'descr': '" + descr + "', 'fortran_order': False, 'shape': "
        + _shape_string(shape) + ", }"
    )
    # NumPy v1.0: magic(6) + version(2) + header-len(2) + header is a
    # multiple of 16 bytes; the header ends in one newline.
    var pad = (16 - ((10 + dictionary.byte_length() + 1) % 16)) % 16
    var header_len = dictionary.byte_length() + pad + 1
    if header_len > 65535:
        raise Error("npz: NumPy v1 header is too large")
    var out = List[UInt8]()
    out.append(0x93)
    _append_string(out, "NUMPY")
    out.append(1)
    out.append(0)
    _put_u16(out, header_len)
    _append_string(out, dictionary)
    for _ in range(pad):
        out.append(0x20)
    out.append(0x0A)
    return out^


def _crc_table() -> List[UInt32]:
    var table = List[UInt32](length=256, fill=0)
    for i in range(256):
        var value = UInt32(i)
        for _ in range(8):
            if value & 1:
                value = (value >> 1) ^ UInt32(0xEDB88320)
            else:
                value >>= 1
        table[i] = value
    return table^


def _crc_byte(mut crc: UInt32, byte: UInt8, table: List[UInt32]):
    crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)


def _crc_bytes(mut crc: UInt32, bytes: List[UInt8], table: List[UInt32]):
    for byte in bytes:
        _crc_byte(crc, byte, table)


def _crc_u32(mut crc: UInt32, value: UInt32, table: List[UInt32]):
    for shift in range(0, 32, 8):
        _crc_byte(crc, UInt8((value >> UInt32(shift)) & 0xFF), table)


def _crc_u64(mut crc: UInt32, value: UInt64, table: List[UInt32]):
    for shift in range(0, 64, 8):
        _crc_byte(crc, UInt8((value >> UInt64(shift)) & 0xFF), table)


def _crc_coords(header: List[UInt8], coords: IntMatrix, table: List[UInt32]) raises -> UInt32:
    var crc = UInt32(0xFFFFFFFF)
    _crc_bytes(crc, header, table)
    for row in range(coords.rows):
        for col in range(1, 4):
            _crc_u32(crc, UInt32(coords.at(row, col)), table)
    return crc ^ UInt32(0xFFFFFFFF)


def _crc_attrs(header: List[UInt8], attrs: Tensor[F32], table: List[UInt32]) -> UInt32:
    var crc = UInt32(0xFFFFFFFF)
    _crc_bytes(crc, header, table)
    for value in attrs.data:
        _crc_u32(crc, value.to_bits[DType.uint32](), table)
    return crc ^ UInt32(0xFFFFFFFF)


def _crc_origin(header: List[UInt8], table: List[UInt32]) -> UInt32:
    var crc = UInt32(0xFFFFFFFF)
    _crc_bytes(crc, header, table)
    var bits = Float32(-0.5).to_bits[DType.uint32]()
    for _ in range(3):
        _crc_u32(crc, bits, table)
    return crc ^ UInt32(0xFFFFFFFF)


def _crc_voxel_size(header: List[UInt8], voxel_size: Float64, table: List[UInt32]) -> UInt32:
    var crc = UInt32(0xFFFFFFFF)
    _crc_bytes(crc, header, table)
    _crc_u64(crc, voxel_size.to_bits[DType.uint64](), table)
    return crc ^ UInt32(0xFFFFFFFF)


def _local_header(name: String, crc: UInt32, size: Int) raises -> List[UInt8]:
    if size < 0 or size > 0xFFFFFFFF:
        raise Error("npz: entry is too large for ZIP32: " + name)
    var out = List[UInt8]()
    _put_u32(out, 0x04034B50)
    _put_u16(out, ZIP_VERSION)
    _put_u16(out, 0)  # flags
    _put_u16(out, 0)  # method: store
    _put_u16(out, 0)  # DOS time
    _put_u16(out, ZIP_DATE_1980_01_01)
    _put_u32(out, Int(crc))
    _put_u32(out, size)
    _put_u32(out, size)
    _put_u16(out, name.byte_length())
    _put_u16(out, 0)  # extra length
    _append_string(out, name)
    return out^


def _flush(mut file: FileHandle, mut buffer: List[UInt8]) raises:
    if len(buffer) > 0:
        file.write_bytes(buffer)
        buffer = List[UInt8]()


def _stream_u32(mut file: FileHandle, mut buffer: List[UInt8], value: UInt32) raises:
    _put_u32(buffer, Int(value))
    if len(buffer) >= STREAM_BYTES:
        _flush(file, buffer)


def _stream_u64(mut file: FileHandle, mut buffer: List[UInt8], value: UInt64) raises:
    for shift in range(0, 64, 8):
        buffer.append(UInt8((value >> UInt64(shift)) & 0xFF))
    if len(buffer) >= STREAM_BYTES:
        _flush(file, buffer)


def _write_coords_entry(
    mut file: FileHandle,
    coords: IntMatrix,
    offset: Int,
    table: List[UInt32],
) raises -> _ZipEntry:
    if coords.cols < 4:
        raise Error("npz: texture coords must include batch + xyz")
    var name = String("coords.npy")
    var header = _npy_header("<i4", [coords.rows, 3])
    var size = len(header) + coords.rows * 3 * 4
    var crc = _crc_coords(header, coords, table)
    file.write_bytes(_local_header(name, crc, size))
    file.write_bytes(header)
    var buffer = List[UInt8]()
    for row in range(coords.rows):
        for col in range(1, 4):
            _stream_u32(file, buffer, UInt32(coords.at(row, col)))
    _flush(file, buffer)
    return _ZipEntry(name^, crc, size, offset)


def _write_attrs_entry(
    mut file: FileHandle,
    attrs: Tensor[F32],
    offset: Int,
    table: List[UInt32],
) raises -> _ZipEntry:
    if len(attrs.shape) != 2:
        raise Error("npz: texture attrs must be rank 2")
    var name = String("attrs.npy")
    var header = _npy_header("<f4", attrs.shape)
    var size = len(header) + len(attrs.data) * 4
    var crc = _crc_attrs(header, attrs, table)
    file.write_bytes(_local_header(name, crc, size))
    file.write_bytes(header)
    var buffer = List[UInt8]()
    for value in attrs.data:
        _stream_u32(file, buffer, value.to_bits[DType.uint32]())
    _flush(file, buffer)
    return _ZipEntry(name^, crc, size, offset)


def _write_origin_entry(
    mut file: FileHandle, offset: Int, table: List[UInt32]
) raises -> _ZipEntry:
    var name = String("origin.npy")
    var header = _npy_header("<f4", [3])
    var size = len(header) + 12
    var crc = _crc_origin(header, table)
    file.write_bytes(_local_header(name, crc, size))
    file.write_bytes(header)
    var buffer = List[UInt8]()
    var bits = Float32(-0.5).to_bits[DType.uint32]()
    for _ in range(3):
        _stream_u32(file, buffer, bits)
    _flush(file, buffer)
    return _ZipEntry(name^, crc, size, offset)


def _write_voxel_size_entry(
    mut file: FileHandle,
    voxel_size: Float64,
    offset: Int,
    table: List[UInt32],
) raises -> _ZipEntry:
    var name = String("voxel_size.npy")
    var header = _npy_header("<f8", List[Int]())
    var size = len(header) + 8
    var crc = _crc_voxel_size(header, voxel_size, table)
    file.write_bytes(_local_header(name, crc, size))
    file.write_bytes(header)
    var buffer = List[UInt8]()
    _stream_u64(file, buffer, voxel_size.to_bits[DType.uint64]())
    _flush(file, buffer)
    return _ZipEntry(name^, crc, size, offset)


def _append_central(mut out: List[UInt8], entry: _ZipEntry):
    _put_u32(out, 0x02014B50)
    _put_u16(out, ZIP_VERSION)  # made by
    _put_u16(out, ZIP_VERSION)  # needed
    _put_u16(out, 0)  # flags
    _put_u16(out, 0)  # store
    _put_u16(out, 0)
    _put_u16(out, ZIP_DATE_1980_01_01)
    _put_u32(out, Int(entry.crc))
    _put_u32(out, entry.size)
    _put_u32(out, entry.size)
    _put_u16(out, entry.name.byte_length())
    _put_u16(out, 0)  # extra
    _put_u16(out, 0)  # comment
    _put_u16(out, 0)  # disk
    _put_u16(out, 0)  # internal attrs
    _put_u32(out, 0)  # external attrs
    _put_u32(out, entry.offset)
    _append_string(out, entry.name)


def write_tex_voxels_npz(
    path: String,
    coords: IntMatrix,
    attrs: Tensor[F32],
    resolution: Int,
) raises:
    """Write coords[T,3] i32, attrs[T,C] f32, origin[3] f32 and scalar
    voxel_size f64 in the same uncompressed .npz schema as the old runner."""
    if resolution <= 0:
        raise Error("npz: resolution must be positive")
    if attrs.shape[0] != coords.rows:
        raise Error("npz: coords/attrs row mismatch")

    var table = _crc_table()
    var file = open(path, "w")
    var entries = List[_ZipEntry]()
    var offset = 0

    var coords_entry = _write_coords_entry(file, coords, offset, table)
    entries.append(coords_entry.copy())
    offset += 30 + coords_entry.name.byte_length() + coords_entry.size

    var attrs_entry = _write_attrs_entry(file, attrs, offset, table)
    entries.append(attrs_entry.copy())
    offset += 30 + attrs_entry.name.byte_length() + attrs_entry.size

    var origin_entry = _write_origin_entry(file, offset, table)
    entries.append(origin_entry.copy())
    offset += 30 + origin_entry.name.byte_length() + origin_entry.size

    var voxel_entry = _write_voxel_size_entry(
        file, 1.0 / Float64(resolution), offset, table
    )
    entries.append(voxel_entry.copy())
    offset += 30 + voxel_entry.name.byte_length() + voxel_entry.size

    var central_offset = offset
    var central = List[UInt8]()
    for entry in entries:
        _append_central(central, entry)
    file.write_bytes(central)
    offset += len(central)

    var end = List[UInt8]()
    _put_u32(end, 0x06054B50)
    _put_u16(end, 0)
    _put_u16(end, 0)
    _put_u16(end, len(entries))
    _put_u16(end, len(entries))
    _put_u32(end, len(central))
    _put_u32(end, central_offset)
    _put_u16(end, 0)
    file.write_bytes(end)
    file.close()
