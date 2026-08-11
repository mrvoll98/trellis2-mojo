"""Small dependency-light GLB 2.0 reader used by offline diagnostics."""

import json
import struct

import numpy as np


def read_glb(path):
    """Return (JSON, positions, normals, colors, flat indices)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    magic, version, total = struct.unpack_from("<III", blob, 0)
    assert magic == 0x46546C67, "bad magic"
    assert version == 2, "bad version"
    assert total == len(blob), "length mismatch"
    jlen, jtype = struct.unpack_from("<II", blob, 12)
    assert jtype == 0x4E4F534A, "first chunk must be JSON"
    doc = json.loads(blob[20 : 20 + jlen].decode("utf-8"))
    boff = 20 + jlen
    blen, btype = struct.unpack_from("<II", blob, boff)
    assert btype == 0x004E4942, "second chunk must be BIN"
    binbuf = blob[boff + 8 : boff + 8 + blen]
    assert doc["buffers"][0]["byteLength"] == blen

    def acc_array(idx, dtype, comps):
        acc = doc["accessors"][idx]
        view = doc["bufferViews"][acc["bufferView"]]
        offset = view.get("byteOffset", 0)
        count = acc["count"]
        arr = np.frombuffer(
            binbuf, dtype=dtype, count=count * comps, offset=offset
        )
        return arr.reshape(count, comps) if comps > 1 else arr

    primitive = doc["meshes"][0]["primitives"][0]
    pos = acc_array(primitive["attributes"]["POSITION"], np.float32, 3)
    nrm = acc_array(primitive["attributes"]["NORMAL"], np.float32, 3)
    col = acc_array(primitive["attributes"]["COLOR_0"], np.float32, 4)
    idx = acc_array(primitive["indices"], np.uint32, 1)
    return doc, pos, nrm, col, idx
