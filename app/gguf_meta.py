import os
import struct

TYPES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "?"}


def _value_bytes(f, off, vt, size):
    """Returns (value, bytes_consumed) for a value of type vt at off."""
    if vt == 8:
        f.seek(off)
        (sl,) = struct.unpack_from("<Q", f.read(8), 0)
        f.seek(off + 8)
        return f.read(sl).decode("utf-8", "replace"), 8 + sl
    if vt == 9:
        f.seek(off)
        (et, cnt) = struct.unpack_from("<IQ", f.read(12), 0)
        off += 12
        if et == 8:
            total = 12
            f.seek(off)
            for _ in range(cnt):
                (ll,) = struct.unpack_from("<Q", f.read(8), 0)
                total += 8 + ll
                f.seek(ll, 1)
            return None, total
        return None, 12 + cnt * TYPES.get(et, 0)
    if vt in TYPES:
        f.seek(off)
        return struct.unpack_from(FMT[vt], f.read(TYPES[vt]), 0)[0], TYPES[vt]
    return None, 0


def read_gguf_meta(path, max_pairs=8192):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) < 24 or head[:4] != b"GGUF":
            return {"error": "not a GGUF file"}
        (version,) = struct.unpack_from("<I", head, 4)
        (tensor_count,) = struct.unpack_from("<Q", head, 8)
        (n_kv,) = struct.unpack_from("<Q", head, 16)

    vals = {}
    off = 24
    with open(path, "rb") as f:
        for _ in range(min(n_kv, max_pairs)):
            if off + 8 > size:
                break
            f.seek(off)
            (ln,) = struct.unpack_from("<Q", f.read(8), 0)
            if off + 8 + ln + 4 > size:
                break
            f.seek(off + 8)
            key = f.read(ln).decode("utf-8", "replace")
            body = off + 8 + ln
            f.seek(body)
            (vt,) = struct.unpack_from("<I", f.read(4), 0)
            body += 4
            value, used = _value_bytes(f, body, vt, size)
            if key:
                vals[key] = value
            off = body + used

    meta = {
        "version": version,
        "tensor_count": tensor_count,
        "block_count": None,
        "architecture": None,
        "context_length": None,
        "general_name": None,
    }
    for k, v in vals.items():
        if v is None or not isinstance(v, (int, float, str)):
            continue
        if k.endswith(".block_count") and meta["block_count"] is None:
            meta["block_count"] = v
        elif k.endswith(".architecture") and meta["architecture"] is None:
            meta["architecture"] = v
        elif k == "general.name" and meta["general_name"] is None:
            meta["general_name"] = v
        elif k.endswith(".context_length") and meta["context_length"] is None:
            meta["context_length"] = v
    return meta


def file_gb(path):
    try:
        return os.path.getsize(path) / 2**30
    except OSError:
        return 0.0