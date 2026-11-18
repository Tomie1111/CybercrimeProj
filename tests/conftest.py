import pytest


@pytest.fixture
def frames():
    rows = []
    for index in range(60):
        rows.append({
            "source_file": "x.pcap", "frame_number": index + 1, "timestamp": index * 0.1,
            "bssid": "00:11:22:33:44:55", "security_profile": "a" if index < 30 else "b",
            "phy_family": "VHT", "sequence": (4090 + index) % 4096, "retry": 0,
            "tsf": index * 100000, "ie_count": 8, "unique_ie_count": 8,
            "duplicate_ie_count": 0, "ie_length_mean": 4, "rate_count": 8,
            "basic_rate_count": 4, "beacon_interval": 100, "vendor_ie_count": 1,
            "vendor_length_mean": 6, "vendor_length_min": 6, "vendor_length_max": 6,
            "vendor_oui_diversity": 1, "vendor_type_diversity": 1, "wps_present": 0,
            "capability": 17, "ess": 1, "privacy": 1, "short_preamble": 0,
            "short_slot": 0, "rsn_present": 1, "rsnxe_present": 0, "vht_present": 1,
            "akm_suites": "000fac:2", "pairwise_ciphers": "000fac:4", "group_cipher": "000fac:4",
            "pmf_capable": False, "pmf_required": False, "he_state": "absent_in_capture",
            "eht_state": "parser_unavailable", "vendor_profile": "v1", "ie_structure": "0,1,48,221",
            "rssi": -40, "noise": -90, "snr": 50, "advertised_channel": 6,
            "observed_frequency": 2437, "radiotap_available": 1, "label": 0,
        })
    return rows
