from evil_twin.features.temporal import wrapped_deltas
from evil_twin.features.windows import make_windows
from evil_twin.features.relational import conflict_features


def test_sequence_wrapping(): assert wrapped_deltas([4094,4095,0,1],4096).tolist()==[1,1,1]


def test_window_membership_sizes(frames):
    # Profile view splits the two profiles: 3xW10, 1xW20 per profile, no W50.
    assert len(make_windows(frames,10,10,"profile"))==6
    assert len(make_windows(frames,20,20,"profile"))==2
    assert len(make_windows(frames,50,50,"profile"))==0


def test_identity_view_observes_conflict(frames):
    out=make_windows(frames,60,60,"identity")[0]
    assert out["same_bssid_profile_conflict"]==1 and out["distinct_profiles"]==2


def test_causal_relational_features(frames):
    assert conflict_features(frames[:30])["same_bssid_profile_conflict"]==0
    assert conflict_features(frames)["same_bssid_profile_conflict"]==1


def test_label_purity(frames):
    frames[0]["label"]=1
    assert make_windows(frames[:20],20,20,"profile")[0]["label_purity"]==.95


def test_no_overlapping_membership(frames):
    windows=make_windows(frames,10,10,"profile")
    members=[set(x["frame_membership"].split("|")) for x in windows]
    assert all(not members[i]&members[j] for i in range(len(members)) for j in range(i))
