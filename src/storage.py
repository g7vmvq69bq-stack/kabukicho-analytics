"""
Dual-sink storage: SQLite (structured queries) + CSV (portable export).
One row = one aggregation window (default 30 seconds).
"""

import csv, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from config import DB_PATH, CSV_PATH

COLUMNS = [
    "id", "timestamp", "window_start",
    # crowd
    "crowd_avg", "crowd_max", "crowd_density_avg",
    # age
    "adults_avg", "children_avg",
    # tourist / local
    "tourists_avg", "locals_avg", "tourist_index",
    # accessories
    "shopping_bags_avg", "suitcases_avg", "backpacks_avg", "umbrellas_avg",
    # gesture
    "waving_avg", "friendliness_idx",
    # gender
    "male_avg", "female_avg", "gender_unknown_avg", "male_ratio",
    # flow
    "direction_balance",
    "movement_speed_avg", "speed_slow_pct", "speed_fast_pct",
    # social
    "solo_avg", "group_count_avg", "largest_group",
    # environment
    "brightness_avg",
    # derived
    "adult_ratio", "luggage_ratio",
    # meta
    "frame_count", "activity",
]

_DDL = f"""
CREATE TABLE IF NOT EXISTS observations (
    {", ".join(
        "id INTEGER PRIMARY KEY AUTOINCREMENT" if c == "id"
        else f"{c} TEXT NOT NULL" if c in ("timestamp", "window_start", "activity")
        else f"{c} INTEGER" if c in ("crowd_max", "largest_group", "frame_count")
        else f"{c} REAL"
        for c in COLUMNS
    )}
)
"""


def _activity(crowd_avg: float, tourist_index: float, brightness: float) -> str:
    if brightness < 80:
        score = crowd_avg * 0.8
        if score < 3:  return "NIGHT-QUIET"
        if score < 10: return "NIGHT-BUSY"
        return "NIGHT-PEAK"
    score = crowd_avg + tourist_index * 8
    if score < 4:  return "QUIET"
    if score < 12: return "MODERATE"
    if score < 22: return "BUSY"
    return "CROWDED"


def _mean(lst):         return sum(lst) / len(lst) if lst else 0.0
def _max(lst):          return max(lst)             if lst else 0
def _pct(lst, val):     return sum(1 for v in lst if v == val) / len(lst) if lst else 0.0


class DataStore:
    def __init__(self):
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(DB_PATH))
        self._conn.execute(_DDL)
        self._conn.commit()

        new_file    = not Path(CSV_PATH).exists()
        self._csv_f = open(CSV_PATH, "a", newline="", encoding="utf-8")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=COLUMNS)
        if new_file:
            self._csv_w.writeheader()

    def save_window(self, window_start: datetime, frames: list[dict]):
        if not frames:
            return None

        def avg(k): return round(_mean([f[k] for f in frames]), 3)
        def mx(k):  return _max([f[k]  for f in frames])

        crowd_avg     = avg("crowd_count")
        tourist_index = avg("tourist_index")
        brightness    = avg("brightness")
        speed_labels  = [f["speed_label"] for f in frames]

        row = {
            "id":                None,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "window_start":      window_start.isoformat(),
            "crowd_avg":         crowd_avg,
            "crowd_max":         mx("crowd_count"),
            "crowd_density_avg": avg("crowd_density"),
            "adults_avg":        avg("adults"),
            "children_avg":      avg("children"),
            "tourists_avg":      avg("tourists"),
            "locals_avg":        avg("locals"),
            "tourist_index":     tourist_index,
            "shopping_bags_avg": avg("shopping_bags"),
            "suitcases_avg":     avg("suitcases"),
            "backpacks_avg":     avg("backpacks"),
            "umbrellas_avg":     avg("umbrellas"),
            "waving_avg":        avg("waving"),
            "friendliness_idx":  avg("friendliness_idx"),
            "male_avg":          avg("gender_male"),
            "female_avg":        avg("gender_female"),
            "gender_unknown_avg":avg("gender_unknown"),
            "male_ratio":        avg("male_ratio"),
            "direction_balance": avg("direction_balance"),
            "movement_speed_avg":avg("movement_speed"),
            "speed_slow_pct":    round(_pct(speed_labels, "SLOW"), 3),
            "speed_fast_pct":    round(_pct(speed_labels, "FAST"), 3),
            "solo_avg":          avg("solo_count"),
            "group_count_avg":   avg("group_count"),
            "largest_group":     mx("largest_group"),
            "brightness_avg":    brightness,
            "adult_ratio":       avg("adult_ratio"),
            "luggage_ratio":     avg("luggage_ratio"),
            "frame_count":       len(frames),
            "activity":          _activity(crowd_avg, tourist_index, brightness),
        }

        vals = [row[c] for c in COLUMNS if c != "id"]
        cur  = self._conn.execute(
            f"INSERT INTO observations ({','.join(c for c in COLUMNS if c != 'id')}) "
            f"VALUES ({','.join('?' * len(vals))})",
            vals,
        )
        self._conn.commit()
        row["id"] = cur.lastrowid
        self._csv_w.writerow(row)
        self._csv_f.flush()

        dir_arrow = (">" if row["direction_balance"] > 0.1
                     else "<" if row["direction_balance"] < -0.1 else "~")
        print(
            f"[storage] [{window_start.strftime('%H:%M:%S')}]  "
            f"crowd={crowd_avg:.1f}  tourists={row['tourists_avg']:.1f}  "
            f"children={row['children_avg']:.1f}  "
            f"flow={dir_arrow}  spd={row['movement_speed_avg']:.2f}  "
            f"bright={brightness:.0f}  [{row['activity']}]"
        )
        return row

    def close(self):
        self._conn.close()
        self._csv_f.close()
