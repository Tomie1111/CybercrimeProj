"""Group-exclusive splitting performed before window generation."""
from __future__ import annotations
from collections import defaultdict
import hashlib, json, random
from pathlib import Path


def make_group_splits(records, seed=20261116, ratios=(0.6,0.2,0.2)):
    groups = defaultdict(list)
    for row in records:
        group = row.get("physical_device") or row.get("campaign") or row.get("session")
        if not group: raise ValueError("Every record needs an independently meaningful group")
        groups[str(group)].append(row)
    names = sorted(groups); random.Random(seed).shuffle(names)
    targets = [round(len(names)*ratios[0]), round(len(names)*(ratios[0]+ratios[1]))]
    out = {"train":[],"validation":[],"test":[]}
    for i,name in enumerate(names): out["train" if i<targets[0] else "validation" if i<targets[1] else "test"].append(name)
    return out


def validate_splits(splits):
    sets = {k:set(v) for k,v in splits.items()}
    overlap = {f"{a}_{b}":sorted(sets[a]&sets[b]) for a,b in [("train","validation"),("train","test"),("validation","test")]}
    return {"passed": not any(overlap.values()), "group_overlap": overlap}


def write_manifest(splits, path, seed=20261116, device_feasible=False):
    audit = validate_splits(splits)
    obj = {"seed":seed,"group_hierarchy":["physical_device","campaign","session"],"memberships":splits,
           "device_held_out_status":"FEASIBLE" if device_feasible else "DEVICE_HELD_OUT_NOT_FEASIBLE",
           "audit":audit}
    Path(path).write_text(json.dumps(obj,indent=2)); return obj
