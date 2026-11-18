"""Causal same-BSSID conflict observer over identity windows."""
from __future__ import annotations


def conflict_features(rows, min_support: int = 2) -> dict:
    def diversity(key):
        counts = {}
        for row in rows:
            value = row.get(key)
            counts[value] = counts.get(value, 0) + 1
        supported = [v for v, n in counts.items() if n >= min_support]
        return len(supported), int(len(supported) > 1)
    profiles, profile_conflict = diversity("security_profile")
    _, cap_conflict = diversity("capability")
    _, akm_conflict = diversity("akm_suites")
    _, cipher_conflict = diversity("pairwise_ciphers")
    _, pmf_conflict = diversity("pmf_required")
    _, vendor_conflict = diversity("vendor_profile")
    _, structure_conflict = diversity("ie_structure")
    return {"distinct_profiles": profiles, "same_bssid_profile_conflict": profile_conflict,
            "same_bssid_capability_conflict": cap_conflict, "same_bssid_akm_conflict": akm_conflict,
            "same_bssid_cipher_conflict": cipher_conflict, "same_bssid_pmf_conflict": pmf_conflict,
            "same_bssid_vendor_conflict": vendor_conflict, "same_bssid_ie_structure_conflict": structure_conflict,
            "conflict_duration": float(rows[-1]["timestamp"]-rows[0]["timestamp"]) if profile_conflict else 0.0,
            "supporting_beacons": len(rows)}
