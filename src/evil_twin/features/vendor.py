"""Normalized Vendor-Specific IE features; raw identity is evidence only."""
from __future__ import annotations


def decode_vendor_ie(data: bytes) -> dict:
    if len(data) < 3:
        return {"valid": False, "oui": None, "type": None, "subtype": None,
                "length": len(data), "wps_attributes": {}}
    oui = data[:3].hex(":")
    vendor_type = data[3] if len(data) > 3 else None
    subtype = data[4] if len(data) > 4 else None
    attrs = {}
    # WFA WPS: 00:50:f2, type 4; attributes are big-endian TLVs.
    if oui == "00:50:f2" and vendor_type == 4:
        p = 4
        while p + 4 <= len(data):
            aid = int.from_bytes(data[p:p+2], "big")
            size = int.from_bytes(data[p+2:p+4], "big")
            p += 4
            if p + size > len(data):
                break
            attrs[f"0x{aid:04x}"] = data[p:p+size].hex()
            p += size
    return {"valid": True, "oui": oui, "type": vendor_type, "subtype": subtype,
            "length": len(data), "wps_attributes": attrs}


def vendor_features(payloads: list[bytes]) -> dict:
    decoded = [decode_vendor_ie(x) for x in payloads]
    lengths = [x["length"] for x in decoded]
    return {
        "vendor_ie_count": len(decoded),
        "vendor_length_mean": sum(lengths) / len(lengths) if lengths else None,
        "vendor_length_min": min(lengths) if lengths else None,
        "vendor_length_max": max(lengths) if lengths else None,
        "vendor_oui_diversity": len({x["oui"] for x in decoded if x["oui"]}),
        "vendor_type_diversity": len({(x["type"], x["subtype"]) for x in decoded}),
        "wps_present": any(bool(x["wps_attributes"]) for x in decoded),
    }
