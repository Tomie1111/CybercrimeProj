"""Feature-family definitions for controlled common-membership ablations."""
FAMILIES = {
 "F1": ["ie_","unique_ie_","duplicate_ie_","rate_","basic_rate_","capability","ess","privacy","short_","phy_","vht_","he_","eht_","advertised_channel"],
 "F2": ["rsn","group_cipher","pairwise_cipher","akm","pmf","security_"],
 "F3": ["vendor_","wps_"],
 "F4": ["tsf_","sequence_","interarrival_","retry_","missing_beacon","timing_","window_duration"],
 "F5": ["rssi_","noise_","snr_","observed_frequency","radiotap_"],
 "F6": ["same_bssid_","distinct_profiles","conflict_","supporting_beacons"],
}

def select_family(columns, families):
    prefixes=[p for f in families for p in FAMILIES[f]]
    return [c for c in columns if any(c.startswith(p) or c==p for p in prefixes)]
