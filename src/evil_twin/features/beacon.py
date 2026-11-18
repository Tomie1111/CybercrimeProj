"""Defensive Scapy Beacon parsing with explicit availability states."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .security import decode_rsn, security_profile
from .vendor import decode_vendor_ie, vendor_features


def collect_ies(packet: Any) -> dict[int, list[bytes]]:
    from scapy.layers.dot11 import Dot11Elt
    result: dict[int, list[bytes]] = defaultdict(list)
    elt = packet.getlayer(Dot11Elt)
    seen = 0
    while elt is not None and seen < 512:
        try:
            result[int(elt.ID)].append(bytes(elt.info))
        except (TypeError, ValueError, AttributeError):
            pass
        elt = elt.payload.getlayer(Dot11Elt) if getattr(elt, "payload", None) else None
        seen += 1
    return dict(result)


def phy_family(ies: dict[int, list[bytes]]) -> str:
    # Extension IE 255 ext IDs: HE 35/36, EHT 106/108 (parser support recorded separately).
    ext_ids = {x[0] for x in ies.get(255, []) if x}
    if ext_ids & {106, 108}:
        return "EHT"
    if ext_ids & {35, 36}:
        return "HE"
    if 191 in ies or 192 in ies:
        return "VHT"
    if 45 in ies or 61 in ies:
        return "HT"
    return "legacy"


def parse_beacon(packet: Any, source_file: str = "", frame_number: int | None = None,
                 packet_length: int | None = None, captured_length: int | None = None) -> dict | None:
    from scapy.layers.dot11 import Dot11, Dot11Beacon, RadioTap
    if not packet.haslayer(Dot11Beacon) or not packet.haslayer(Dot11):
        return None
    dot11, beacon = packet[Dot11], packet[Dot11Beacon]
    ies = collect_ies(packet)
    rsn = decode_rsn(ies[48][0]) if 48 in ies else {}
    rates = list(ies.get(1, [b""])[0]) + list(ies.get(50, [b""])[0])
    ids = [i for i, values in ies.items() for _ in values]
    info_lengths = [len(x) for values in ies.values() for x in values]
    rt = packet.getlayer(RadioTap)
    signal = getattr(rt, "dBm_AntSignal", None) if rt else None
    noise = getattr(rt, "dBm_AntNoise", None) if rt else None
    freq = getattr(rt, "ChannelFrequency", None) if rt else None
    channel = ies.get(3, [b""])[0][0] if ies.get(3, [b""])[0] else None
    cap = int(getattr(beacon, "cap", 0))
    vf = vendor_features(ies.get(221, []))
    vendor_metadata = [decode_vendor_ie(value) for value in ies.get(221, [])]
    wire_length = packet_length if packet_length is not None else len(bytes(packet))
    return {
        "source_file": source_file, "frame_number": frame_number,
        # Retained for provenance/ground-truth application; forbidden from ML predictors.
        "packet_length": wire_length, "captured_length": captured_length,
        "packet_truncated": bool(captured_length is not None and wire_length is not None and captured_length < wire_length),
        "malformed": False, "timestamp": float(packet.time), "timestamp_provenance": "pcap_epoch_timezone_unspecified",
        "bssid": dot11.addr3, "transmitter_address": dot11.addr2, "receiver_address": dot11.addr1,
        "ssid": ies.get(0, [b""])[0].hex(), "ssid_present": int(0 in ies),
        "sequence": int(getattr(dot11, "SC", 0)) >> 4, "retry": int(bool(int(dot11.FCfield) & 0x8)),
        "tsf": int(getattr(beacon, "timestamp", 0)), "beacon_interval": int(getattr(beacon, "beacon_interval", 0)),
        "capability": cap, "ess": int(bool(cap & 0x1)), "privacy": int(bool(cap & 0x10)),
        "short_preamble": int(bool(cap & 0x20)), "short_slot": int(bool(cap & 0x400)),
        "ie_count": len(ids), "unique_ie_count": len(set(ids)), "duplicate_ie_count": len(ids)-len(set(ids)),
        "ie_length_mean": sum(info_lengths)/len(info_lengths) if info_lengths else None,
        "rate_count": len(rates), "basic_rate_count": sum(bool(r & 0x80) for r in rates),
        "rsn_present": int(48 in ies), "rsn_valid": rsn.get("valid") if 48 in ies else None,
        "group_cipher": rsn.get("group_cipher"),
        "pairwise_ciphers": "|".join(sorted(rsn.get("pairwise", []))), "akm_suites": "|".join(sorted(rsn.get("akm", []))),
        "pmf_capable": rsn.get("pmf_capable"), "pmf_required": rsn.get("pmf_required"),
        "rsnxe_present": int(244 in ies), "security_profile": repr(security_profile(ies)),
        "phy_family": phy_family(ies), "vht_present": int(191 in ies or 192 in ies),
        "he_state": "present" if any(x and x[0] in {35,36} for x in ies.get(255,[])) else "absent_in_capture",
        "eht_state": "present" if any(x and x[0] in {106,108} for x in ies.get(255,[])) else "parser_unavailable",
        "advertised_channel": channel, "rssi": signal, "noise": noise,
        "snr": signal-noise if signal is not None and noise is not None else None,
        "observed_frequency": freq, "radiotap_available": int(rt is not None),
        "ie_structure": ",".join(map(str, ids)), "ie_ids": ids, "ie_lengths": info_lengths,
        "vendor_ie_metadata": vendor_metadata, **vf,
        "vendor_profile": repr((tuple(sorted(len(x) for x in ies.get(221, []))), vf["vendor_type_diversity"])),
    }
