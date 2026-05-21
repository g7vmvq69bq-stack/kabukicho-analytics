"""
Event Engine — Part 3: Generate events when something relevant happens.

Monitors each 30-second window and fires named events when thresholds are
crossed or a meaningful state change is detected.

Events are printed to the console AND saved to:
  - data/kabukicho_events.csv   (portable)
  - data/kabukicho.db           (events table, same database)

Event types
-----------
CROWD_SURGE         crowd_avg jumps by +3 or more vs previous window
CROWD_DROP          crowd_avg falls by -3 or more vs previous window
CROWD_PEAK          crowd_max (single frame) reaches 12 or more people
ACTIVITY_CHANGE     activity label changes (e.g. MODERATE → BUSY)
NIGHT_TRANSITION    brightness crosses below 90 (day → night)
DAY_TRANSITION      brightness crosses above 90 (night → day)
LARGE_GROUP         largest_group exceeds threshold — fires once, then
                    only again when group size grows larger than before
GENDER_SKEW         male_ratio exceeds 0.95 (extreme male dominance)
DIRECTION_REVERSAL  walking direction flips sign (both sides must exceed 0.08)

Cooldown
--------
Each event type has a minimum cooldown in windows before it can fire again.
This prevents the same event flooding the log every 30 seconds.
"""

import csv, sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

# ── Thresholds ────────────────────────────────────────────────────────────────
CROWD_SURGE_DELTA     =  3.0   # people / frame
CROWD_DROP_DELTA      = -3.0
CROWD_PEAK_THRESHOLD  = 12     # crowd_max (single frame peak)
NIGHT_BRIGHTNESS      = 90.0
LARGE_GROUP_THRESHOLD =  8     # people in one group
GENDER_SKEW_THRESHOLD =  0.95  # male_ratio above this = extreme skew
DIRECTION_MIN_SIGNAL  =  0.08  # both windows must exceed this to count

# ── Cooldown (minimum windows between same event type firing again) ────────────
COOLDOWN = {
    "CROWD_SURGE":       2,
    "CROWD_DROP":        2,
    "CROWD_PEAK":        3,
    "ACTIVITY_CHANGE":   1,
    "NIGHT_TRANSITION":  5,
    "DAY_TRANSITION":    5,
    "LARGE_GROUP":       3,
    "GENDER_SKEW":       3,
    "DIRECTION_REVERSAL":2,
}

# ── CSV columns ───────────────────────────────────────────────────────────────
EVENT_COLUMNS = [
    "id", "timestamp", "window_start",
    "event_type", "severity", "description",
    "metric_name", "metric_value", "threshold",
]

_EVENTS_CSV = config.DATA_DIR / "kabukicho_events.csv"

_DDL_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    window_start TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    severity     TEXT NOT NULL,
    description  TEXT NOT NULL,
    metric_name  TEXT,
    metric_value REAL,
    threshold    REAL
)
"""

HIGH   = "HIGH"
MEDIUM = "MEDIUM"
INFO   = "INFO"

_COL = {HIGH: "\033[91m", MEDIUM: "\033[93m", INFO: "\033[96m"}
_RST = "\033[0m"


class EventEngine:
    """
    Call check(row) after every save_window().
    row must be the dict returned by DataStore.save_window().
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.execute(_DDL_EVENTS)
        self._conn.commit()

        new_file = not Path(_EVENTS_CSV).exists()
        self._csv_f = open(_EVENTS_CSV, "a", newline="", encoding="utf-8")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=EVENT_COLUMNS)
        if new_file:
            self._csv_w.writeheader()

        self._prev: dict | None = None
        self._is_night: bool | None = None
        self._largest_group_seen: int = 0
        self._window_index: int = 0
        self._last_fired: dict[str, int] = {}   # event_type → window index

    # ── public ────────────────────────────────────────────────────────────────
    def check(self, row: dict) -> list[dict]:
        """Evaluate row against all rules. Returns list of fired events."""
        self._window_index += 1
        events = []

        if self._prev is not None:
            events += self._check_crowd_surge(row)
            events += self._check_direction_reversal(row)
            events += self._check_activity_change(row)

        events += self._check_night_transition(row)
        events += self._check_crowd_peak(row)
        events += self._check_large_group(row)
        events += self._check_gender_skew(row)

        for ev in events:
            self._save(ev, row["window_start"])
            self._last_fired[ev["event_type"]] = self._window_index

        self._prev = row
        return events

    def close(self):
        self._csv_f.close()

    # ── cooldown check ────────────────────────────────────────────────────────
    def _off_cooldown(self, event_type: str) -> bool:
        last = self._last_fired.get(event_type, -999)
        return (self._window_index - last) > COOLDOWN.get(event_type, 1)

    # ── rules ─────────────────────────────────────────────────────────────────
    def _check_crowd_surge(self, row):
        if not self._off_cooldown("CROWD_SURGE") and not self._off_cooldown("CROWD_DROP"):
            return []
        delta = row["crowd_avg"] - self._prev["crowd_avg"]
        if delta >= CROWD_SURGE_DELTA and self._off_cooldown("CROWD_SURGE"):
            return [self._make(
                "CROWD_SURGE", HIGH,
                f"Crowd jumped +{delta:.1f} people/frame  "
                f"({self._prev['crowd_avg']:.1f} → {row['crowd_avg']:.1f})",
                "crowd_delta", delta, CROWD_SURGE_DELTA,
            )]
        if delta <= CROWD_DROP_DELTA and self._off_cooldown("CROWD_DROP"):
            return [self._make(
                "CROWD_DROP", MEDIUM,
                f"Crowd dropped {delta:.1f} people/frame  "
                f"({self._prev['crowd_avg']:.1f} → {row['crowd_avg']:.1f})",
                "crowd_delta", delta, CROWD_DROP_DELTA,
            )]
        return []

    def _check_activity_change(self, row):
        if not self._off_cooldown("ACTIVITY_CHANGE"):
            return []
        if row["activity"] != self._prev["activity"]:
            prev_a = self._prev["activity"]
            curr_a = row["activity"]
            sev = HIGH if "PEAK" in curr_a or "CROWDED" in curr_a else MEDIUM
            return [self._make(
                "ACTIVITY_CHANGE", sev,
                f"Activity changed: {prev_a} → {curr_a}",
                "activity", None, None,
            )]
        return []

    def _check_night_transition(self, row):
        bright    = row["brightness_avg"]
        now_night = bright < NIGHT_BRIGHTNESS
        events    = []
        if self._is_night is not None:
            if now_night and not self._is_night and self._off_cooldown("NIGHT_TRANSITION"):
                events.append(self._make(
                    "NIGHT_TRANSITION", HIGH,
                    f"District entered night mode — brightness dropped to {bright:.0f}",
                    "brightness_avg", bright, NIGHT_BRIGHTNESS,
                ))
            elif not now_night and self._is_night and self._off_cooldown("DAY_TRANSITION"):
                events.append(self._make(
                    "DAY_TRANSITION", INFO,
                    f"District returned to daytime — brightness rose to {bright:.0f}",
                    "brightness_avg", bright, NIGHT_BRIGHTNESS,
                ))
        self._is_night = now_night
        return events

    def _check_crowd_peak(self, row):
        if not self._off_cooldown("CROWD_PEAK"):
            return []
        if row["crowd_max"] >= CROWD_PEAK_THRESHOLD:
            return [self._make(
                "CROWD_PEAK", HIGH,
                f"Peak crowd in a single frame: {row['crowd_max']} people  "
                f"(avg this window: {row['crowd_avg']:.1f})",
                "crowd_max", row["crowd_max"], CROWD_PEAK_THRESHOLD,
            )]
        return []

    def _check_direction_reversal(self, row):
        if not self._off_cooldown("DIRECTION_REVERSAL"):
            return []
        prev_dir = self._prev["direction_balance"]
        curr_dir = row["direction_balance"]
        if abs(prev_dir) > DIRECTION_MIN_SIGNAL and abs(curr_dir) > DIRECTION_MIN_SIGNAL:
            if (prev_dir > 0) != (curr_dir > 0):
                prev_label = "→ into district" if prev_dir > 0 else "← toward station"
                curr_label = "→ into district" if curr_dir > 0 else "← toward station"
                return [self._make(
                    "DIRECTION_REVERSAL", MEDIUM,
                    f"Walking direction reversed: {prev_label} → {curr_label}",
                    "direction_balance", curr_dir, DIRECTION_MIN_SIGNAL,
                )]
        return []

    def _check_large_group(self, row):
        if not self._off_cooldown("LARGE_GROUP"):
            return []
        lg = row["largest_group"]
        # only fire when a new maximum is reached
        if lg >= LARGE_GROUP_THRESHOLD and lg > self._largest_group_seen:
            self._largest_group_seen = lg
            return [self._make(
                "LARGE_GROUP", INFO,
                f"New largest group detected: {lg} people moving together",
                "largest_group", lg, LARGE_GROUP_THRESHOLD,
            )]
        return []

    def _check_gender_skew(self, row):
        if not self._off_cooldown("GENDER_SKEW"):
            return []
        if row.get("male_ratio", 0) >= GENDER_SKEW_THRESHOLD:
            return [self._make(
                "GENDER_SKEW", INFO,
                f"Extreme male dominance: {row['male_ratio']*100:.0f}% male in this window",
                "male_ratio", row["male_ratio"], GENDER_SKEW_THRESHOLD,
            )]
        return []

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _make(event_type, severity, description,
              metric_name, metric_value, threshold) -> dict:
        return {
            "event_type":   event_type,
            "severity":     severity,
            "description":  description,
            "metric_name":  metric_name,
            "metric_value": round(metric_value, 3) if metric_value is not None else None,
            "threshold":    threshold,
        }

    def _save(self, ev: dict, window_start: str):
        ts = datetime.now(timezone.utc).isoformat()
        sev_col = _COL.get(ev["severity"], "")
        print(
            f"{sev_col}[EVENT] [{ev['severity']:6s}] {ev['event_type']}: "
            f"{ev['description']}{_RST}"
        )

        self._conn.execute(
            "INSERT INTO events "
            "(timestamp, window_start, event_type, severity, description, "
            " metric_name, metric_value, threshold) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ts, window_start, ev["event_type"], ev["severity"],
             ev["description"], ev["metric_name"],
             ev["metric_value"], ev["threshold"]),
        )
        self._conn.commit()

        self._csv_w.writerow({
            "id":           None,
            "timestamp":    ts,
            "window_start": window_start,
            "event_type":   ev["event_type"],
            "severity":     ev["severity"],
            "description":  ev["description"],
            "metric_name":  ev["metric_name"],
            "metric_value": ev["metric_value"],
            "threshold":    ev["threshold"],
        })
        self._csv_f.flush()
