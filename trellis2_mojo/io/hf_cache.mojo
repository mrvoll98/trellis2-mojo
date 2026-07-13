# Pure-Mojo mirror of ckpt_io.py's path resolution (WP12): resolve the
# local HF cache without network or Python. Same semantics: newest
# snapshot = lexicographically last entry under snapshots/ (glob("*")
# skips dotfiles, so we do too), and ckpt names come in the three
# pipeline.json forms ("ckpts/<file>", bare "<file>", and the cross-repo
# "microsoft/TRELLIS-image-large/ckpts/<file>" reference for ss_dec).

from std.os import getenv, listdir

from trellis2_mojo.io.json import JsonDoc, parse_json

comptime TRELLIS2_4B = "models--microsoft--TRELLIS.2-4B"
comptime IMAGE_LARGE = "models--microsoft--TRELLIS-image-large"
comptime CROSS_REPO_PREFIX = "microsoft/TRELLIS-image-large/"


def _substr(s: String, start: Int) raises -> String:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(start, len(b)):
        out.append(b[i])
    return String(from_utf8=Span(out))


def hf_hub_dir() raises -> String:
    var home = getenv("HF_HOME")
    if home == "":
        home = getenv("HOME") + "/.cache/huggingface"
    elif home.startswith("~"):
        home = getenv("HOME") + _substr(home, 1)
    return home + "/hub"


def snapshot_dir(repo_dirname: String) raises -> String:
    var d = hf_hub_dir() + "/" + repo_dirname + "/snapshots"
    var snaps = List[String]()
    for e in listdir(d):
        if not e.startswith("."):
            snaps.append(e.copy())
    if len(snaps) == 0:
        raise Error("no local HF snapshot for " + repo_dirname + " under " + d)
    sort(snaps)
    return d + "/" + snaps[len(snaps) - 1]


def ckpt_base(name: String) raises -> String:
    """Model name (pipeline.json form) -> checkpoint base path, no extension."""
    if name.startswith(CROSS_REPO_PREFIX):
        return snapshot_dir(IMAGE_LARGE) + "/" + _substr(name, CROSS_REPO_PREFIX.byte_length())
    var rel = name.copy()
    if rel.find("/") == -1:
        rel = "ckpts/" + rel
    return snapshot_dir(TRELLIS2_4B) + "/" + rel


def read_file_bytes(path: String) raises -> List[UInt8]:
    var fh = open(path, "r")
    var data = fh.read_bytes()
    fh.close()
    return data^


def load_config_json(name: String) raises -> JsonDoc:
    return parse_json(read_file_bytes(ckpt_base(name) + ".json"))


def pipeline_config_json() raises -> JsonDoc:
    return parse_json(read_file_bytes(snapshot_dir(TRELLIS2_4B) + "/pipeline.json"))


def model_path(model_key: String) raises -> String:
    """pipeline.json args.models[<key>] -> name for load_config_json/ckpt_base."""
    var doc = pipeline_config_json()
    var models = doc.obj_get(doc.obj_get(doc.root, "args"), "models")
    return doc.get_str(doc.obj_get(models, model_key))
