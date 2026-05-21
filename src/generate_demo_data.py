"""
Generate a realistic 14-hour demo dataset for Kabukicho (no stream required).
Simulates: quiet morning → tourist afternoon peak → night entertainment surge.

Run:  python src/generate_demo_data.py
Then: python src/analysis.py
"""

import random, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from storage import DataStore

random.seed(42)


def _profile(jst_hour: int) -> dict:
    """Return activity profile for a given Tokyo hour."""
    if   jst_hour < 9:  f, night = 0.10, False
    elif jst_hour < 12: f, night = 0.35, False
    elif jst_hour < 15: f, night = 0.75, False
    elif jst_hour < 18: f, night = 0.90, False
    elif jst_hour < 20: f, night = 0.85, False
    elif jst_hour < 23: f, night = 1.00, True   # peak — neon district
    else:               f, night = 0.65, True

    brightness = (40  + random.gauss(15, 5)  if night
                  else 130 + random.gauss(30, 10))
    return dict(f=f, night=night,
                brightness=max(20, min(255, brightness)),
                tourist_boost=1.3 if 14 <= jst_hour <= 20 else 1.0)


def _frame(p: dict) -> dict:
    f  = p["f"]
    tb = p["tourist_boost"]
    g  = lambda mu, sd: max(0.0, random.gauss(mu, sd))

    crowd      = int(g(f * 22, 5))
    children   = int(g(crowd * 0.08, 1))
    adults     = crowd - children

    suitcases  = int(g(f * tb * 2.0, 1))
    backpacks  = int(g(f * tb * 4.0, 2))
    shopping   = int(g(f * 3.0, 2))
    umbrellas  = int(g(0.4, 0.4))

    t_score    = min(1.0, (suitcases * 0.9 + backpacks * 0.4) / max(crowd, 1))
    tourists   = int(crowd * t_score)
    locals_    = crowd - tourists

    # Gender — Kabukicho skews male, especially at night
    male_ratio  = 0.65 if p["night"] else 0.55
    detectable  = int(crowd * random.uniform(0.5, 0.8))  # not all faces visible
    gender_male = int(detectable * male_ratio + random.gauss(0, 1))
    gender_male = max(0, min(detectable, gender_male))
    gender_female = detectable - gender_male
    gender_unknown = crowd - detectable

    waving     = int(g(f * 0.6, 0.4))
    direction  = random.gauss(-0.3 if p["night"] else 0.1, 0.8)
    speed      = g(f * 1.8, 0.5)
    speed_lbl  = ("SLOW"     if speed < config.SPEED_SLOW_MAX
                  else "FAST" if speed >= config.SPEED_FAST_MIN
                  else "MODERATE")
    solo       = int(g(crowd * 0.35, 2))
    groups     = int(g(f * 1.8, 0.6))
    lg         = int(g(4 * f, 1)) if groups > 0 else 0

    return {
        "crowd_count":      crowd,
        "crowd_density":    round(min(0.5, crowd / 200.0), 4),
        "adults":           adults,
        "children":         children,
        "tourists":         tourists,
        "locals":           locals_,
        "tourist_index":    round(t_score, 3),
        "shopping_bags":    shopping,
        "suitcases":        suitcases,
        "backpacks":        backpacks,
        "umbrellas":        umbrellas,
        "gender_male":      gender_male,
        "gender_female":    gender_female,
        "gender_unknown":   gender_unknown,
        "male_ratio":       round(gender_male / max(detectable, 1), 3),
        "waving":           waving,
        "friendliness_idx": round(waving / max(crowd, 1), 3),
        "direction_balance":round(direction, 3),
        "movement_speed":   round(speed, 3),
        "speed_label":      speed_lbl,
        "solo_count":       solo,
        "group_count":      groups,
        "largest_group":    lg,
        "adult_ratio":      round(adults / max(crowd, 1), 3),
        "luggage_ratio":    round((suitcases + backpacks) / max(crowd, 1), 3),
        "brightness":       round(p["brightness"], 1),
    }


def main():
    for p in [config.DB_PATH, config.CSV_PATH]:
        if p.exists(): p.unlink()

    store = DataStore()
    # 09:00–23:00 JST = 00:00–14:00 UTC
    start = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
    t, end, count = start, start + timedelta(hours=14), 0

    while t < end:
        jst_hour = (t.hour + 9) % 24
        profile  = _profile(jst_hour)
        frames   = [_frame(profile) for _ in range(random.randint(8, 14))]
        store.save_window(t, frames)
        t    += timedelta(seconds=config.AGGREGATION_WINDOW_SECONDS)
        count += 1

    store.close()
    print(f"\n[demo] {count} windows written.")
    print(f"[demo] DB  : {config.DB_PATH}")
    print(f"[demo] CSV : {config.CSV_PATH}")
    print("\nRun  python src/analysis.py  to see the charts.\n")


if __name__ == "__main__":
    main()
