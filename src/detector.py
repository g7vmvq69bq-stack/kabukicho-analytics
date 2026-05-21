"""
Rich feature extractor — tuned for the Kabukicho pedestrian street cam.

Models used
-----------
  yolov8n.pt       — people, bags, vehicles (COCO object detection)
  yolov8n-pose.pt  — 17-point body skeleton per person
  DeepFace         — gender estimation on upper-body crops

Per-frame output dict
─────────────────────
crowd_count        total people visible
crowd_density      fraction of frame area covered by person bboxes (0–1)
adults             estimated adults  (bbox height > ADULT_MIN_RATIO of frame)
children           estimated children (bbox height < CHILD_MAX_RATIO of frame)
tourists           people with suitcase / backpack nearby
locals             remaining people
tourist_index      tourists / crowd_count
shopping_bags      handbags detected
suitcases          rolling luggage detected
backpacks          backpacks detected
umbrellas          umbrellas detected
waving             people with wrist above shoulder (pose model)
friendliness_idx   waving / crowd_count
gender_male        estimated males (confident detections only)
gender_female      estimated females (confident detections only)
gender_unknown     people where gender could not be determined
male_ratio         gender_male / (gender_male + gender_female)  0–1
direction_balance  mean horizontal optical flow (+ rightward, - leftward)
movement_speed     mean optical-flow magnitude inside person bboxes
speed_label        SLOW / MODERATE / FAST
solo_count         people not in a group
group_count        number of groups (>= GROUP_MIN_SIZE within GROUP_DISTANCE_PX)
largest_group      size of the largest group
brightness         mean frame brightness 0–255 (day/night proxy)
annotated          BGR frame with overlays
"""

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    YOLO_OBJ_MODEL, YOLO_POSE_MODEL, CONFIDENCE,
    PERSON_CLASS, VEHICLE_CLASSES, TOURIST_WEIGHTS,
    ADULT_MIN_RATIO, CHILD_MAX_RATIO,
    WAVING_PIXEL_THRESHOLD,
    GROUP_DISTANCE_PX, GROUP_MIN_SIZE,
    SPEED_SLOW_MAX, SPEED_FAST_MIN,
    SUITCASE_CLASS, BACKPACK_CLASS, UMBRELLA_CLASS, HANDBAG_CLASS,
    GENDER_MIN_BBOX_HEIGHT, GENDER_CONFIDENCE_MIN,
    GENDER_MAX_PER_FRAME, GENDER_FRAME_SKIP,
)

_LS, _RS = 5, 6    # left/right shoulder keypoint indices
_LW, _RW = 9, 10   # left/right wrist keypoint indices


# ── Gender estimator ──────────────────────────────────────────────────────────

class GenderEstimator:
    """
    Estimates gender from upper-body crops using DeepFace.
    Falls back gracefully if DeepFace is not installed.

    Note: gender is estimated from visual appearance only and is used
    as a statistical approximation, not as an individual classification.
    """

    def __init__(self):
        self.enabled = False
        try:
            from deepface import DeepFace
            self._df = DeepFace
            self.enabled = True
            print("[gender] DeepFace loaded — gender estimation enabled.")
        except ImportError:
            print("[gender] DeepFace not found — gender estimation disabled.")
            print("[gender] Install with: pip install deepface tf-keras")

    def estimate(
        self, frame: np.ndarray, person_boxes: list[tuple]
    ) -> dict:
        """
        Returns dict with keys: male, female, unknown.
        Only runs on boxes large enough to give reliable results.
        """
        unknown = len(person_boxes)
        if not self.enabled or not person_boxes:
            return {"male": 0, "female": 0, "unknown": unknown}

        male = female = unknown = 0

        # Sort by bbox area descending — prioritise larger (closer) people
        sorted_boxes = sorted(
            person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )[:GENDER_MAX_PER_FRAME]

        # People not analysed at all → unknown
        unknown = len(person_boxes) - len(sorted_boxes)

        for (x1, y1, x2, y2) in sorted_boxes:
            bbox_h = y2 - y1
            if bbox_h < GENDER_MIN_BBOX_HEIGHT:
                unknown += 1
                continue

            # Crop the upper 45% of the person bbox (head + upper body)
            crop_y2 = y1 + int(bbox_h * 0.45)
            crop    = frame[y1:crop_y2, x1:x2]

            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                unknown += 1
                continue

            try:
                result     = self._df.analyze(
                    crop,
                    actions=["gender"],
                    enforce_detection=False,
                    silent=True,
                )
                dominant   = result[0]["dominant_gender"]   # "Man" or "Woman"
                confidence = result[0]["gender"][dominant]  # 0–100

                if confidence >= GENDER_CONFIDENCE_MIN:
                    if dominant == "Man":
                        male += 1
                    else:
                        female += 1
                else:
                    unknown += 1

            except Exception:
                unknown += 1

        return {"male": male, "female": female, "unknown": unknown}


# ── Main detector ─────────────────────────────────────────────────────────────

class Detector:
    def __init__(self):
        print(f"[detector] Loading {YOLO_OBJ_MODEL} …")
        self._obj    = YOLO(YOLO_OBJ_MODEL)
        print(f"[detector] Loading {YOLO_POSE_MODEL} …")
        self._pose   = YOLO(YOLO_POSE_MODEL)
        self._gender = GenderEstimator()
        self._prev_gray: np.ndarray | None = None
        self._frame_count = 0
        self._last_gender = {"male": 0, "female": 0, "unknown": 0}
        print("[detector] Ready.\n")

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        self._frame_count += 1
        h, w = frame.shape[:2]

        obj_res  = self._obj( frame, conf=CONFIDENCE, verbose=False)[0]
        pose_res = self._pose(frame, conf=CONFIDENCE, verbose=False)[0]

        # ── parse detections ──────────────────────────────────────────────────
        person_boxes: list[tuple] = []
        acc_boxes:    dict[int, list] = {}

        for box in obj_res.boxes:
            cls  = int(box.cls[0])
            xyxy = tuple(map(int, box.xyxy[0]))
            if cls == PERSON_CLASS:
                person_boxes.append(xyxy)
            elif cls in TOURIST_WEIGHTS:
                acc_boxes.setdefault(cls, []).append(xyxy)

        # ── per-person: age class + tourist score ─────────────────────────────
        adults = children = tourists = locals_ = 0
        person_cx:        list[float] = []
        total_person_area = 0

        for (x1, y1, x2, y2) in person_boxes:
            person_cx.append((x1 + x2) / 2)
            total_person_area += (x2 - x1) * (y2 - y1)

            hr = (y2 - y1) / h
            if   hr >= ADULT_MIN_RATIO: adults   += 1
            elif hr <= CHILD_MAX_RATIO: children += 1
            else:                       adults   += 1

            if self._tourist_score((x1, y1, x2, y2), acc_boxes) >= 0.35:
                tourists += 1
            else:
                locals_ += 1

        # ── waving ────────────────────────────────────────────────────────────
        waving = 0
        if pose_res.keypoints is not None:
            for kpts in pose_res.keypoints.xy.cpu().numpy():
                if self._is_waving(kpts):
                    waving += 1

        # ── gender estimation (throttled) ─────────────────────────────────────
        if self._frame_count % GENDER_FRAME_SKIP == 0:
            self._last_gender = self._gender.estimate(frame, person_boxes)
        gender = self._last_gender

        male    = gender["male"]
        female  = gender["female"]
        g_unk   = gender["unknown"]
        detected = male + female
        male_ratio = male / detected if detected > 0 else 0.5  # default 50/50

        # ── optical flow ──────────────────────────────────────────────────────
        direction_balance, movement_speed = self._compute_flow(frame, person_boxes)

        # ── grouping ──────────────────────────────────────────────────────────
        solo_count, group_count, largest_group = self._detect_groups(person_cx)

        # ── accessories ───────────────────────────────────────────────────────
        suitcases     = len(acc_boxes.get(SUITCASE_CLASS, []))
        backpacks     = len(acc_boxes.get(BACKPACK_CLASS, []))
        umbrellas     = len(acc_boxes.get(UMBRELLA_CLASS, []))
        shopping_bags = len(acc_boxes.get(HANDBAG_CLASS,  []))

        # ── derived ───────────────────────────────────────────────────────────
        crowd_count   = len(person_boxes)
        crowd_density = total_person_area / (h * w) if (h * w) > 0 else 0.0
        tourist_index = tourists / crowd_count       if crowd_count > 0 else 0.0
        friendliness  = waving   / crowd_count       if crowd_count > 0 else 0.0
        adult_ratio   = adults   / crowd_count       if crowd_count > 0 else 0.0
        luggage_ratio = (suitcases + backpacks) / crowd_count if crowd_count > 0 else 0.0
        brightness    = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

        if   movement_speed < SPEED_SLOW_MAX: speed_label = "SLOW"
        elif movement_speed < SPEED_FAST_MIN: speed_label = "MODERATE"
        else:                                 speed_label = "FAST"

        return {
            "crowd_count":       crowd_count,
            "crowd_density":     round(crowd_density,     4),
            "adults":            adults,
            "children":          children,
            "tourists":          tourists,
            "locals":            locals_,
            "tourist_index":     round(tourist_index,     3),
            "shopping_bags":     shopping_bags,
            "suitcases":         suitcases,
            "backpacks":         backpacks,
            "umbrellas":         umbrellas,
            "waving":            waving,
            "friendliness_idx":  round(friendliness,      3),
            "gender_male":       male,
            "gender_female":     female,
            "gender_unknown":    g_unk,
            "male_ratio":        round(male_ratio,         3),
            "direction_balance": round(direction_balance,  3),
            "movement_speed":    round(movement_speed,     3),
            "speed_label":       speed_label,
            "solo_count":        solo_count,
            "group_count":       group_count,
            "largest_group":     largest_group,
            "adult_ratio":       round(adult_ratio,        3),
            "luggage_ratio":     round(luggage_ratio,      3),
            "brightness":        round(brightness,         1),
            "annotated":         self._annotate(
                frame, obj_res, pose_res, person_boxes, h,
                tourists, children, waving, male, female,
                tourist_index, friendliness, direction_balance, speed_label,
            ),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _tourist_score(self, person_box: tuple, acc_boxes: dict) -> float:
        px1, py1, px2, py2 = person_box
        pw   = px2 - px1
        zx1  = px1 - pw * 0.5
        zx2  = px2 + pw * 0.5
        score = 0.0
        for cls_id, weight in TOURIST_WEIGHTS.items():
            for (ax1, ay1, ax2, ay2) in acc_boxes.get(cls_id, []):
                if zx1 <= (ax1 + ax2) / 2 <= zx2:
                    score += weight
        return min(score, 1.0)

    def _is_waving(self, kpts: np.ndarray) -> bool:
        for wi, si in [(_LW, _LS), (_RW, _RS)]:
            wx, wy = kpts[wi];  sx, sy = kpts[si]
            if (wx == 0 and wy == 0) or (sx == 0 and sy == 0):
                continue
            if sy - wy > WAVING_PIXEL_THRESHOLD:
                return True
        return False

    def _compute_flow(
        self, frame: np.ndarray, person_boxes: list[tuple]
    ) -> tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._prev_gray  = gray
        direction_balance = float(flow[..., 0].mean())

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        if not person_boxes:
            movement_speed = float(mag.mean())
        else:
            fh, fw = gray.shape
            sx, sy = fw / frame.shape[1], fh / frame.shape[0]
            vals = []
            for (x1, y1, x2, y2) in person_boxes:
                r1 = max(0, int(y1 * sy));  r2 = min(fh, int(y2 * sy))
                c1 = max(0, int(x1 * sx));  c2 = min(fw, int(x2 * sx))
                if r2 > r1 and c2 > c1:
                    vals.append(mag[r1:r2, c1:c2].mean())
            movement_speed = float(np.mean(vals)) if vals else float(mag.mean())

        return direction_balance, movement_speed

    def _detect_groups(self, cx: list[float]) -> tuple[int, int, int]:
        if not cx:
            return 0, 0, 0
        clusters, current = [], [sorted(cx)[0]]
        for x in sorted(cx)[1:]:
            if x - current[-1] <= GROUP_DISTANCE_PX:
                current.append(x)
            else:
                clusters.append(current);  current = [x]
        clusters.append(current)
        solo   = sum(1 for c in clusters if len(c) == 1)
        groups = [c for c in clusters if len(c) >= GROUP_MIN_SIZE]
        return solo, len(groups), max((len(g) for g in groups), default=0)

    def _annotate(
        self, frame, obj_res, pose_res, person_boxes, h_frame,
        tourists, children, waving, male, female,
        tourist_index, friendliness, direction_balance, speed_label,
    ) -> np.ndarray:
        out = frame.copy()

        for box in obj_res.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_CLASS:
                is_child = (y2 - y1) / h_frame <= CHILD_MAX_RATIO
                color    = (255, 140, 0) if is_child else (50, 220, 50)
                label    = f"{'child' if is_child else 'adult'} {conf:.0%}"
            elif cls in TOURIST_WEIGHTS:
                color = (0, 180, 255)
                label = {24:"backpack", 25:"umbrella",
                         26:"bag",      28:"suitcase"}.get(cls, "bag")
            else:
                continue

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        if pose_res.keypoints is not None:
            for kpts in pose_res.keypoints.xy.cpu().numpy():
                for idx in [_LS, _RS, _LW, _RW]:
                    x, y = int(kpts[idx][0]), int(kpts[idx][1])
                    if x > 0 or y > 0:
                        col = (0, 255, 255) if idx in (_LW, _RW) else (255, 255, 0)
                        cv2.circle(out, (x, y), 4, col, -1)

        # Direction arrow
        cx_mid, cy_mid = out.shape[1] // 2, out.shape[0] - 30
        alen = int(min(abs(direction_balance) * 40, 80))
        if direction_balance > 0.1:
            cv2.arrowedLine(out, (cx_mid, cy_mid), (cx_mid + alen, cy_mid), (0, 200, 255), 3)
        elif direction_balance < -0.1:
            cv2.arrowedLine(out, (cx_mid, cy_mid), (cx_mid - alen, cy_mid), (0, 200, 255), 3)

        # HUD
        crowd  = len(person_boxes)
        dir_t  = (">> RIGHT" if direction_balance > 0.1
                  else "<< LEFT" if direction_balance < -0.1 else "MIXED")
        total_g = male + female
        g_str  = (f"M:{male} F:{female} ({male/total_g:.0%} male)"
                  if total_g > 0 else "gender: estimating…")
        hud = [
            f"Crowd:{crowd}  tourists:{tourists}  children:{children}  waving:{waving}",
            f"Gender — {g_str}",
            f"Flow:{dir_t}  Speed:{speed_label}  Tourist:{tourist_index:.0%}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(out, line, (10, 26 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        return out
