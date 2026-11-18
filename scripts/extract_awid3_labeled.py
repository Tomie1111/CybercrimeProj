"""Extract official-filter-labeled AWID3 W10/W20/W50 tables from raw PCAPs."""
from pathlib import Path
import pandas as pd, yaml
from evil_twin.inference.predict import extract_labeled_frames
from evil_twin.features.windows import make_windows

ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/"configs/dataset.yaml").read_text())
data=(ROOT/cfg["dataset_root"]).resolve(); out_dir=ROOT/"results"; out_dir.mkdir(exist_ok=True)
inventory_path=out_dir/"dataset_inventory.csv"
inventory=pd.read_csv(inventory_path) if inventory_path.exists() else None
for source,name in [(cfg["sources"][0],"rogue_ap"),(cfg["sources"][1],"evil_twin")]:
    frames=extract_labeled_frames(data/source["file"],source["ground_truth_rule"])
    if inventory is not None and frames:
        mask=inventory.filename.eq(source["file"])
        updates={"beacon_frame_count":len(frames),"capture_start":min(x["timestamp"] for x in frames),
                 "capture_end":max(x["timestamp"] for x in frames),
                 "bssid":"|".join(sorted({x["bssid"] for x in frames})),
                 "ssid":"|".join(sorted({x["ssid"] for x in frames})),
                 "phy_capability":"|".join(sorted({x["phy_family"] for x in frames})),
                 "radiotap_metadata":bool(any(x["radiotap_available"] for x in frames)),"inventory_status":"beacon_scanned"}
        for column,value in updates.items():
            if column in {"bssid","ssid","phy_capability","radiotap_metadata","inventory_status"}:
                inventory[column]=inventory[column].astype(object)
            inventory.loc[mask,column]=value
    for size in [10,20,50]:
        windows=pd.DataFrame(make_windows(frames,size,size,"profile"))
        windows=windows[windows.label_purity.eq(1)].copy()
        windows.to_csv(out_dir/f"awid3_{name}_profile_w{size}_labeled.csv",index=False)
        print(name,size,len(windows),windows.label.value_counts().to_dict())
if inventory is not None:
    inventory.to_csv(inventory_path,index=False)
