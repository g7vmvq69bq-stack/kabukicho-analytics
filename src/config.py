"""
Configuration for the Kabukicho, Shinjuku — Urban Pulse Monitor.
All tunable parameters live here. Edit STREAM_URL if the link changes.
"""

import pathlib

# ── Stream ────────────────────────────────────────────────────────────────────
STREAM_URL = "https://www.youtube.com/watch?v=DjdUEyjx8GM"
# Kabukicho Ichibankai, Shinjuku, Tokyo — 24-hour pedestrian street cam

# ── Recording ─────────────────────────────────────────────────────────────────
AGGREGATION_WINDOW_SECONDS = 30   # one DB row per this many seconds
FRAME_SKIP = 3                    # process every Nth frame (speed vs accuracy)

# ── Models ────────────────────────────────────────────────────────────────────
YOLO_OBJ_MODEL  = "yolov8n.pt"        # objects: people, bags, vehicles
YOLO_POSE_MODEL = "yolov8n-pose.pt"   # body keypoints: gestures, proportions
CONFIDENCE      = 0.35                 # detection confidence threshold

# ── COCO class IDs ────────────────────────────────────────────────────────────
PERSON_CLASS     = 0
BICYCLE_CLASS    = 1
CAR_CLASS        = 2
MOTORCYCLE_CLASS = 3
BUS_CLASS        = 5
TRUCK_CLASS      = 7
BACKPACK_CLASS   = 24
UMBRELLA_CLASS   = 25
HANDBAG_CLASS    = 26
SUITCASE_CLASS   = 28

VEHICLE_CLASSES = {CAR_CLASS, TRUCK_CLASS, BUS_CLASS, MOTORCYCLE_CLASS, BICYCLE_CLASS}

# Accessory → tourist likelihood weight (combined score ≥ 0.35 → tourist)
TOURIST_WEIGHTS = {
    SUITCASE_CLASS: 0.90,   # rolling luggage → very strong tourist signal
    BACKPACK_CLASS: 0.40,   # backpack       → moderate (tourist or student)
    UMBRELLA_CLASS: 0.10,   # umbrella       → weather-related, weak signal
    HANDBAG_CLASS:  0.05,   # handbag        → mostly locals, very weak signal
}

# ── Adult / child classification (bounding-box height / frame height) ─────────
ADULT_MIN_RATIO = 0.22   # bbox height > 22 % of frame → adult
CHILD_MAX_RATIO = 0.14   # bbox height < 14 % of frame → child

# ── Waving gesture (pose model) ───────────────────────────────────────────────
WAVING_PIXEL_THRESHOLD = 15   # wrist must be this many px above shoulder

# ── Group detection ───────────────────────────────────────────────────────────
GROUP_DISTANCE_PX = 110   # horizontal distance (px) to be in the same group
GROUP_MIN_SIZE    = 3     # minimum people to form a group

# ── Movement speed (optical-flow magnitude) ───────────────────────────────────
SPEED_SLOW_MAX = 0.8    # below → SLOW  (browsing / standing)
SPEED_FAST_MIN = 2.5    # above → FAST  (rushing)

# ── Gender estimation (DeepFace) ──────────────────────────────────────────────
# Only attempt gender on person bboxes taller than this many pixels
GENDER_MIN_BBOX_HEIGHT = 80
# Confidence threshold: only count if DeepFace is this % sure (0-100)
GENDER_CONFIDENCE_MIN  = 60
# Max people to run gender estimation on per frame (keeps speed acceptable)
GENDER_MAX_PER_FRAME   = 6
# Run gender estimation every Nth processed frame (1 = every frame)
GENDER_FRAME_SKIP      = 2

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "output"
DB_PATH      = DATA_DIR / "kabukicho.db"
CSV_PATH     = DATA_DIR / "kabukicho.csv"
