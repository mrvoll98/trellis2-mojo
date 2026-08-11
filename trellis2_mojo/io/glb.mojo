# WP15 (fase 2, ADR 0008): pure-Mojo GLB 2.0 writer for the textured
# vertex-attribute export — one mesh primitive with POSITION / NORMAL /
# COLOR_0 (VEC4 f32, clamped to [0,1] like upstream's np.clip at bake) /
# u32 indices, and a single pbrMetallicRoughness material where
# metallic/roughness ride as GLOBAL factors (glTF has no standard
# per-vertex channel for them; per-voxel PBR stays in the npz).
#
# Container layout: 12-byte header (magic "glTF", version 2, total length)
# + JSON chunk (padded to 4 with spaces) + BIN chunk (payload is 4-aligned
# by construction: 12/12/16/4-byte elements). Written with two
# f.write_bytes calls: a byte-appended head (header + JSON) and a
# pointer-filled binary payload (byte-wise appends are too slow for the
# ~33 MB golden payload).
#
# Axis convention: `to_glb_axes` mirrors upstream o_voxel postprocess
# (y,z -> z,-y for vertices AND normals); callers apply it BEFORE
# write_glb — the writer stores exactly what it is given (the parity
# roundtrip reads back bit-exact values).

from trellis2_mojo.sparse.tensor import Tensor, IntMatrix

comptime F32 = DType.float32
comptime U32 = DType.uint32


def to_glb_axes(mut t: Tensor[F32]):
    """In place on [N, 3] rows: y, z = z, -y (upstream to_glb's swap)."""
    var n = t.shape[0]
    for i in range(n):
        var y = t.data[i * 3 + 1]
        var z = t.data[i * 3 + 2]
        t.data[i * 3 + 1] = z
        t.data[i * 3 + 2] = -y


def _u32le(mut b: List[UInt8], v: Int):
    b.append(UInt8(v & 0xFF))
    b.append(UInt8((v >> 8) & 0xFF))
    b.append(UInt8((v >> 16) & 0xFF))
    b.append(UInt8((v >> 24) & 0xFF))


def _str_bytes(mut b: List[UInt8], s: String):
    for c in s.as_bytes():
        b.append(c)


def write_glb(
    path: String,
    vertices: Tensor[F32],
    normals: Tensor[F32],
    colors: Tensor[F32],
    faces: IntMatrix,
    metallic_factor: Float64,
    roughness_factor: Float64,
) raises:
    """vertices/normals [V, 3], colors [V, 4] RGBA in ~[0,1] (clamped on
    write), faces [F, 3] (0-based). Factors in [0, 1]."""
    var v = vertices.shape[0]
    if normals.shape[0] != v or colors.shape[0] != v:
        raise Error("write_glb: vertex-attribute row mismatch")
    if colors.shape[1] != 4:
        raise Error("write_glb: colors must be [V, 4]")
    var nidx = faces.rows * 3

    # --- binary payload: positions | normals | colors | indices ---
    var pos_len = v * 12
    var nrm_len = v * 12
    var col_len = v * 16
    var idx_len = nidx * 4
    var bin_len = pos_len + nrm_len + col_len + idx_len
    var bin = List[UInt8](length=bin_len, fill=0)
    var fp = bin.unsafe_ptr().unsafe_bitcast[Scalar[F32]]()
    for i in range(v * 3):
        fp.unsafe_store(i, vertices.data[i])
    var noff = v * 3
    for i in range(v * 3):
        fp.unsafe_store(noff + i, normals.data[i])
    var coff = v * 6
    for i in range(v * 4):
        var x = colors.data[i]
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        fp.unsafe_store(coff + i, x)
    var ip = (bin.unsafe_ptr().unsafe_offset(pos_len + nrm_len + col_len)).unsafe_bitcast[
        Scalar[U32]
    ]()
    for f in range(faces.rows):
        ip.unsafe_store(f * 3 + 0, UInt32(faces.at(f, 0)))
        ip.unsafe_store(f * 3 + 1, UInt32(faces.at(f, 1)))
        ip.unsafe_store(f * 3 + 2, UInt32(faces.at(f, 2)))

    # POSITION accessors require min/max
    var mn = List[Float32](length=3, fill=0)
    var mx = List[Float32](length=3, fill=0)
    for d in range(3):
        if v > 0:
            mn[d] = vertices.data[d]
            mx[d] = vertices.data[d]
    for i in range(1, v):
        for d in range(3):
            var x = vertices.data[i * 3 + d]
            if x < mn[d]:
                mn[d] = x
            if x > mx[d]:
                mx[d] = x

    # --- JSON chunk ---
    var js = String('{"asset":{"version":"2.0","generator":"trellis2-mojo"}')
    js += ',"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}]'
    js += ',"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1'
    js += ',"COLOR_0":2},"indices":3,"material":0,"mode":4}]}]'
    js += ',"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]'
    js += ',"metallicFactor":' + String(metallic_factor)
    js += ',"roughnessFactor":' + String(roughness_factor)
    js += '},"doubleSided":true}]'
    js += ',"buffers":[{"byteLength":' + String(bin_len) + "}]"
    js += ',"bufferViews":['
    js += '{"buffer":0,"byteOffset":0,"byteLength":' + String(pos_len) + ',"target":34962}'
    js += ',{"buffer":0,"byteOffset":' + String(pos_len) + ',"byteLength":' + String(nrm_len) + ',"target":34962}'
    js += ',{"buffer":0,"byteOffset":' + String(pos_len + nrm_len) + ',"byteLength":' + String(col_len) + ',"target":34962}'
    js += ',{"buffer":0,"byteOffset":' + String(pos_len + nrm_len + col_len) + ',"byteLength":' + String(idx_len) + ',"target":34963}'
    js += "]"
    js += ',"accessors":['
    js += '{"bufferView":0,"componentType":5126,"count":' + String(v) + ',"type":"VEC3"'
    js += ',"min":[' + String(mn[0]) + "," + String(mn[1]) + "," + String(mn[2]) + "]"
    js += ',"max":[' + String(mx[0]) + "," + String(mx[1]) + "," + String(mx[2]) + "]}"
    js += ',{"bufferView":1,"componentType":5126,"count":' + String(v) + ',"type":"VEC3"}'
    js += ',{"bufferView":2,"componentType":5126,"count":' + String(v) + ',"type":"VEC4"}'
    js += ',{"bufferView":3,"componentType":5125,"count":' + String(nidx) + ',"type":"SCALAR"}'
    js += "]}"

    var json_len = js.byte_length()
    var json_pad = (4 - json_len % 4) % 4
    var total = 12 + 8 + json_len + json_pad + 8 + bin_len

    var head = List[UInt8]()
    _str_bytes(head, "glTF")
    _u32le(head, 2)
    _u32le(head, total)
    _u32le(head, json_len + json_pad)
    _str_bytes(head, "JSON")
    _str_bytes(head, js)
    for _ in range(json_pad):
        head.append(0x20)
    _u32le(head, bin_len)
    head.append(0x42)  # B
    head.append(0x49)  # I
    head.append(0x4E)  # N
    head.append(0x00)
    var f = open(path, "w")
    f.write_bytes(head)
    f.write_bytes(bin)
    f.close()
