import pandas as pd
import pytest
import inspect
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from evil_twin.data.labels import is_supervised_eligible
from evil_twin.data.labels import apply_ground_truth
from evil_twin.data.inventory import sha256_file
from evil_twin.data.splits import validate_splits
from evil_twin.ml.preprocessing import infer_schema,build_preprocessor,validate_schema
from evil_twin.ml.train import train_select
import evil_twin.ml.train as train_module


def test_group_leakage_detection():
    assert validate_splits({"train":["a"],"validation":["b"],"test":["c"]})["passed"]
    assert not validate_splits({"train":["a"],"validation":["a"],"test":[]})["passed"]


def test_no_artificial_labels():
    assert not is_supervised_eligible({"supervised":True,"label_status":"filename_inferred","original_label":"attack"})
    assert is_supervised_eligible({"supervised":True,"label_status":"verified","original_label":"attack"})


def test_official_ground_truth_rule_boundaries():
    rule={"type":"awid3_official_filter","frame_start":10,"frame_end":20,
          "max_frame_length_exclusive":242,"bssid":"aa:bb:cc:dd:ee:ff","attack_class":"evil_twin"}
    base={"frame_number":10,"packet_length":241,"bssid":"aa:bb:cc:dd:ee:ff"}
    assert apply_ground_truth(base,rule)["label"]==1
    assert apply_ground_truth(base|{"packet_length":242},rule)["label"]==0
    assert apply_ground_truth(base|{"frame_number":21},rule)["label"]==0
    assert apply_ground_truth(base|{"bssid":"00:00:00:00:00:00"},rule)["label"]==0


def test_forbidden_identifiers_excluded():
    frame=pd.DataFrame({"bssid":["a","b"],"source_file":["x","y"],"label":[0,1],"rssi_mean":[-1,-2],"phy_family":["HT","VHT"]})
    schema=infer_schema(frame.drop(columns="label"))
    assert "bssid" not in schema.numeric+schema.categorical and "source_file" not in schema.numeric+schema.categorical


def test_schema_consistency():
    train=pd.DataFrame({"x":[1,2],"kind":["a","b"]}); schema=infer_schema(train)
    build_preprocessor(schema).fit(train)
    validate_schema(pd.DataFrame({"x":[3],"kind":["unseen"]}),schema)
    with pytest.raises(ValueError): validate_schema(pd.DataFrame({"x":[3]}),schema)


def test_training_refuses_single_or_unverified_class(tmp_path):
    one=pd.DataFrame({"x":[1,2,3],"label":[1,1,1]})
    with pytest.raises(ValueError,match="both independently verified classes"): train_select(one,one,tmp_path)


def test_preprocessing_is_unfitted_before_train():
    prep=build_preprocessor(infer_schema(pd.DataFrame({"x":[1,2]})))
    assert not hasattr(prep,"transformers_")


def test_validation_only_threshold_selection_and_reproducibility(tmp_path,monkeypatch):
    def tiny(seed):
        yield "cart",{"max_depth":2},DecisionTreeClassifier(max_depth=2,random_state=seed)
    monkeypatch.setattr(train_module,"candidates",tiny)
    train=pd.DataFrame({"x":list(range(20)),"kind":["a","b"]*10,"label":[0]*10+[1]*10})
    val=pd.DataFrame({"x":list(range(6,14)),"kind":["new"]*8,"label":[0]*4+[1]*4})
    _,a=train_select(train,val,tmp_path/"a",seed=7)
    _,b=train_select(train,val,tmp_path/"b",seed=7)
    assert a["selection_split"]=="validation" and a["threshold"]==b["threshold"]
    assert np.array_equal(a["pipeline"].predict(train.drop(columns="label")),b["pipeline"].predict(train.drop(columns="label")))


def test_no_final_test_driven_selection_api():
    assert "test" not in inspect.signature(train_select).parameters


def test_read_only_hash_preserves_historical_file(tmp_path):
    historical=tmp_path/"old-model.bin"; historical.write_bytes(b"frozen")
    before=(historical.stat().st_mtime_ns,historical.read_bytes())
    assert len(sha256_file(historical))==64
    assert (historical.stat().st_mtime_ns,historical.read_bytes())==before
