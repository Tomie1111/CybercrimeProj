"""Schema-checked train-only preprocessing."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FORBIDDEN = {"bssid","ssid","source_file","physical_device","label","original_label","capture_filename",
             "frame_membership","membership_hash","raw_oui","vendor_payload_hash","window_start","window_end",
             "packet_length","class_name","label_provenance","label_purity"}


@dataclass
class FeatureSchema:
    numeric: list[str]
    categorical: list[str]
    forbidden: list[str]

    def as_dict(self): return {"numeric":self.numeric,"categorical":self.categorical,"forbidden":self.forbidden}


def infer_schema(frame: pd.DataFrame) -> FeatureSchema:
    # Entirely unavailable train columns cannot be imputed; keep them in the
    # availability report, but never let validation/test make them trainable.
    columns = [c for c in frame.columns if c not in FORBIDDEN and not frame[c].isna().all()]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in columns if c not in numeric]
    return FeatureSchema(numeric, categorical, sorted(FORBIDDEN & set(frame.columns)))


def build_preprocessor(schema: FeatureSchema, scale=True):
    numeric_steps = [("imputer",SimpleImputer(strategy="median",add_indicator=True))]
    if scale: numeric_steps.append(("scaler",StandardScaler()))
    return ColumnTransformer([
        ("numeric",Pipeline(numeric_steps),schema.numeric),
        ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                 ("onehot",OneHotEncoder(handle_unknown="ignore"))]),schema.categorical),
    ], verbose_feature_names_out=False)


def validate_schema(frame, schema):
    missing = sorted(set(schema.numeric+schema.categorical)-set(frame.columns))
    if missing: raise ValueError(f"Missing feature columns: {missing}")
