"""agent/drift_data.py — the measured drift set of world df8c55dabb35, embedded.

A READ-ONLY COPY of kit/world/df8c55dabb35/drift.json, reduced to what the gateway
and the strategy helpers need at decision time. `Gateway.decide` is pure and may do no
I/O (RULES.md section 3), so it cannot open drift.json; the numbers are baked in here
instead. They are facts about the corpus, not a secret: the same file ships to every
student under kit/world/.

  drifts  True if the working (w) and canonical (c) replicas of the deck disagree.
  w, c    content-frame counts per replica (day18: 45 vs 31 is the textbook case).

A path_id that is NOT in this table was never measured and is treated as not drifting
(the same rule `World.drifts` applies).
"""

from __future__ import annotations

WORLD_ID = "df8c55dabb35"

DRIFT: dict[str, dict] = {
    "04032b0c": {"drifts": False, "w": 47, "c": 47},
    "053195a5": {"drifts": True, "w": 79, "c": 79},
    "0e11ae43": {"drifts": True, "w": 45, "c": 11},
    "170c4d17": {"drifts": True, "w": 53, "c": 40},
    "28e68faa": {"drifts": True, "w": 42, "c": 10},
    "3326cb76": {"drifts": True, "w": 42, "c": 23},
    "45098556": {"drifts": True, "w": 42, "c": 14},
    "594ef4c1": {"drifts": False, "w": 38, "c": 38},
    "61a20716": {"drifts": False, "w": 42, "c": 42},
    "6cc8191a": {"drifts": False, "w": 26, "c": 26},
    "72f75709": {"drifts": True, "w": 37, "c": 15},
    "75811e75": {"drifts": True, "w": 25, "c": 21},
    "7a8d8046": {"drifts": True, "w": 81, "c": 21},
    "a13b20e6": {"drifts": True, "w": 52, "c": 30},
    "a284ae8b": {"drifts": True, "w": 31, "c": 15},
    "abd20c68": {"drifts": True, "w": 61, "c": 61},
    "b3616a68": {"drifts": False, "w": 81, "c": 81},
    "b55e1f40": {"drifts": False, "w": 81, "c": 81},
    "c5418ab5": {"drifts": True, "w": 38, "c": 8},
    "d8f95a7b": {"drifts": True, "w": 45, "c": 31},
    "e0614beb": {"drifts": True, "w": 84, "c": 59},
    "f2696f5f": {"drifts": True, "w": 77, "c": 54},
}


def drifts(path_id: str) -> bool:
    rec = DRIFT.get(path_id)
    return bool(rec and rec["drifts"])


def fresher_replica(path_id: str) -> str | None:
    """`"w"` / `"c"` — the replica with MORE content frames for a drifting path_id, or
    `None` when the path does not drift (either replica is equally current)."""
    rec = DRIFT.get(path_id)
    if not rec or not rec["drifts"]:
        return None
    if rec["w"] == rec["c"]:
        return None  # structure differs but content-frame count is equal (see deck/README atk_03)
    return "w" if rec["w"] > rec["c"] else "c"
