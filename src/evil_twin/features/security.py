"""Standards-aware RSN decoding. Unknown/truncated fields remain unavailable."""
from __future__ import annotations

import struct


def _suite(data: bytes) -> str:
    return f"{data[:3].hex()}:{data[3]}" if len(data) == 4 else "unavailable"


def decode_rsn(data: bytes) -> dict:
    out = {"valid": False, "group_cipher": None, "pairwise": [], "akm": [],
           "rsn_capabilities": None, "pmf_capable": None, "pmf_required": None,
           "group_management_cipher": None}
    try:
        pos = 0
        version, = struct.unpack_from("<H", data, pos); pos += 2
        if version != 1:
            return out
        out["group_cipher"] = _suite(data[pos:pos+4]); pos += 4
        count, = struct.unpack_from("<H", data, pos); pos += 2
        out["pairwise"] = [_suite(data[pos+i*4:pos+(i+1)*4]) for i in range(count)]
        pos += count * 4
        count, = struct.unpack_from("<H", data, pos); pos += 2
        out["akm"] = [_suite(data[pos+i*4:pos+(i+1)*4]) for i in range(count)]
        pos += count * 4
        if len(data) >= pos + 2:
            caps, = struct.unpack_from("<H", data, pos); pos += 2
            out.update(rsn_capabilities=caps, pmf_required=bool(caps & (1 << 6)),
                       pmf_capable=bool(caps & (1 << 7)))
        # PMKID count/list are optional and precede group management cipher.
        if len(data) >= pos + 2:
            n_pmkid, = struct.unpack_from("<H", data, pos)
            candidate = pos + 2 + 16 * n_pmkid
            if candidate <= len(data):
                pos = candidate
        if len(data) >= pos + 4:
            out["group_management_cipher"] = _suite(data[pos:pos+4])
        out["valid"] = True
    except (struct.error, IndexError):
        pass
    return out


def security_profile(ies: dict[int, list[bytes]]) -> tuple:
    if 48 not in ies:
        return ("open_or_legacy",)
    rsn = decode_rsn(ies[48][0])
    return ("rsn", rsn["group_cipher"], tuple(sorted(rsn["pairwise"])),
            tuple(sorted(rsn["akm"])), rsn["pmf_capable"], rsn["pmf_required"])
