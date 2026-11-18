"""Operator-confirmed exact BSSID scope for the D00-D05 research captures."""
from __future__ import annotations

DECO_BSSID = "ba:a7:b9:ae:39:1c"
ZYXEL_BSSID = "28:28:5d:a6:1c:28"
TARGET_BSSIDS = frozenset({DECO_BSSID, ZYXEL_BSSID})


def target_identity(bssid: object) -> str | None:
    """Return the in-scope AP identity; all other BSSIDs are excluded."""
    value = str(bssid).strip().lower()
    if value == DECO_BSSID:
        return "Deco"
    if value == ZYXEL_BSSID:
        return "Zyxel"
    return None


def is_in_scope(session: object, bssid: object) -> bool:
    """Apply the capture-specific target: Deco through D03, Zyxel in D04, both in D05."""
    name = str(session)
    identity = target_identity(bssid)
    if name.startswith("D05"):
        return identity in {"Deco", "Zyxel"}
    if name == "D04":
        return identity == "Zyxel"
    if name.startswith(("D00", "D01", "D02", "D03")):
        return identity == "Deco"
    return False


def ground_truth(session: object, bssid: object) -> int | None:
    """D05 Zyxel is Evil Twin; every other in-scope observation is legitimate."""
    identity = target_identity(bssid)
    if not is_in_scope(session, bssid):
        return None
    return int(str(session).startswith("D05") and identity == "Zyxel")
