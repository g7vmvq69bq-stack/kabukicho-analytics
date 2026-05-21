# Kabukicho Urban Pulse Monitor
### Assignment: Data Extraction from Livestreams
**Postgraduate Programme in IoT & Big Data**

---

## Overview

This application connects to the 24-hour live camera at **Kabukicho Ichibankai, Shinjuku, Tokyo** and extracts rich behavioural data from the video stream in real time.

Kabukicho is Tokyo's most famous entertainment and tourist district. The camera provides a continuous view of a busy pedestrian street, making it ideal for urban crowd analytics.

---

## Features extracted

| Feature | Method | Why it is useful |
|---|---|---|
| **Crowd count & density** | YOLOv8 person detection | Quantifies footfall; useful for event planning and retail staffing |
| **Tourist vs Local** | Suitcases / backpacks detected near people | Tourism boards and local businesses can track visitor ratios |
| **Adult vs Child** | Bounding-box height relative to frame | Family visitor detection; spikes on weekends and school holidays |
| **Gender distribution** | DeepFace on upper-body crops | Estimates male/female ratio across the crowd (statistical approximation) |
| **Walking direction** | Optical flow horizontal component | Shows whether people are arriving into or leaving the district |
| **Movement speed** | Optical flow magnitude inside person regions | Distinguishes browsing visitors from commuters passing through |
| **Friendliness index** | YOLOv8-pose wrist-above-shoulder detection | Creative metric: how welcoming does the crowd feel over time |
| **Social grouping** | Horizontal clustering of person centres | Tourist groups tend to be larger than solo local commuters |
| **Accessories** | YOLOv8 COCO class detection | Suitcases, backpacks, umbrellas, shopping bags per frame |
| **Day vs Night** | Mean frame brightness | Kabukicho transforms at night — neon lights up, crowds surge |

Every 30 seconds the aggregated window is stored in **SQLite** and **CSV**.

---

## Installation

### Requirements
- Python 3.9 or higher
- Internet connection (for stream access and first-run model downloads)
- No GPU required — all models run on CPU

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/kabukicho-analytics.git
cd kabukicho-analytics
```

### Step 2 — Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

This installs:
| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Video frame capture and optical flow |
| `ultralytics` | ≥ 8.0 | YOLOv8 object detection and pose estimation |
| `yt-dlp` | ≥ 2024.1 | Resolves YouTube live stream to direct HLS URL |
| `deepface` | ≥ 0.0.93 | Gender estimation from upper-body crops |
| `tf-keras` | ≥ 2.16 | Required backend for DeepFace |
| `pandas` | ≥ 2.0 | Data loading and aggregation |
| `matplotlib` | ≥ 3.7 | Chart generation |
| `seaborn` | ≥ 0.13 | Heatmap and styled charts |
| `numpy` | ≥ 1.24 | Numerical operations |

> **First run note:** YOLOv8 models (~6 MB each) and the DeepFace gender model (~537 MB) are downloaded automatically on first use. Ensure you have a stable internet connection.

---

## Usage

### Record from the live stream
```bash
python src/main.py                    # runs until you press q
python src/main.py --duration 600     # stop after 10 minutes
python src/main.py --no-display       # headless — no video window
```

Press **q** in the video window to stop early.

### Generate analysis charts
```bash
python src/analysis.py                # charts from today's session
python src/analysis.py --all          # all sessions in the database
```

### Try without a stream (offline demo)
```bash
python src/generate_demo_data.py      # generates 14 hours of realistic data
python src/analysis.py
```

---

## Output charts

Both sessions (daytime and evening) produce the same 6 charts, saved to `output/daytime/` and `output/evening/`:

| File | Contents |
|---|---|
| `01_dashboard.png` | KPI cards (crowd, tourist index, gender split, friendliness) + crowd timeline + activity pie |
| `03_tourist_local.png` | Tourist vs local stacked timeline + accessory bar chart + tourist index trend |
| `04_demographics.png` | Adult/child timeline + age pie + solo vs group behaviour |
| `05_crowd_comparison.png` | Crowd metrics side-by-side: daytime vs evening (avg, peak, adults, children, activity labels) |
| `06_gender_comparison.png` | Gender distribution comparison: daytime vs evening (pie charts, stacked ratio, counts) |
| `07_correlations.png` | Metric correlation heatmap (Big Data pattern discovery) |

---

## Project structure

```
kabukicho-analytics/
├── src/
│   ├── config.py              all settings — edit STREAM_URL here if needed
│   ├── stream_reader.py       resolves YouTube / RTSP / HLS to OpenCV
│   ├── detector.py            YOLOv8 object + pose + DeepFace gender + optical flow
│   ├── storage.py             SQLite + CSV dual-sink writer
│   ├── event_engine.py        event detection — fires alerts when thresholds are crossed
│   ├── main.py                recording loop
│   ├── analysis.py            6 charts + text report
│   └── generate_demo_data.py  offline demo data generator
├── data/
│   ├── kabukicho.csv          Raw data — daytime session (14 May 2026, 12:39–12:49 JST)
│   ├── kabukicho_evening.csv  Raw data — evening session (14 May 2026, 18:57–19:06 JST)
│   └── kabukicho_events.csv   Event log — events fired during daytime session
├── output/
│   ├── daytime/               6 charts from the daytime recording session
│   │   ├── 01_dashboard.png
│   │   ├── 03_tourist_local.png
│   │   ├── 04_demographics.png
│   │   ├── 05_crowd_comparison.png
│   │   ├── 06_gender_comparison.png
│   │   └── 07_correlations.png
│   └── evening/               6 charts from the evening recording session
│       ├── 01_dashboard.png
│       ├── 03_tourist_local.png
│       ├── 04_demographics.png
│       ├── 05_crowd_comparison.png
│       ├── 06_gender_comparison.png
│       └── 07_correlations.png
├── BigData_Pipeline_Report.docx
└── requirements.txt
```

## Raw data

Two real recording sessions are included in the `data/` folder:

| File | Session | Time (JST) | Windows | Avg Crowd |
|---|---|---|---|---|
| `kabukicho.csv` | Daytime | 14 May 2026, 12:39–12:49 | 20 rows | 6.6 people/frame |
| `kabukicho_evening.csv` | Evening | 14 May 2026, 18:57–19:06 | 20 rows | 11.7 people/frame |

Each CSV row represents one 30-second aggregation window and contains 32 columns covering crowd count, tourist index, gender ratio, movement speed, walking direction, accessories, brightness, and activity label. See the [Database schema](#database-schema) section for full column definitions.

---

## File explanations — what each file does and why it exists

### `src/config.py` — All settings in one place

This file holds every tunable parameter for the whole project: the YouTube stream URL, model filenames, detection thresholds, folder paths, and aggregation timing. No other file contains hard-coded values — they all import from here. This means if you ever want to point the application at a different camera, change the recording interval, or tweak a confidence threshold, you only edit one file.

### `src/stream_reader.py` — Connects to the live camera

YouTube does not give you a direct video address. It wraps the real stream inside a webpage. This file uses a tool called **yt-dlp** to look inside that webpage and extract the actual HLS (`.m3u8`) video address. It then hands that address to OpenCV so the rest of the application can read frames from it as if it were a normal camera. Without this file, the app could not connect to a YouTube live stream.

### `src/detector.py` — Analyses every video frame

This is the brain of the project. For every frame it receives, it runs:

- **YOLOv8 object model** — detects people, suitcases, backpacks, umbrellas, shopping bags
- **YOLOv8 pose model** — detects body keypoints (wrists, shoulders) to identify waving
- **DeepFace gender estimator** — crops the upper body of each detected person and estimates male/female
- **Optical flow** — compares two consecutive frames to measure how fast and in which direction people are moving
- **Geometric logic** — uses bounding-box heights for adult/child classification, and spatial proximity of luggage to people for tourist/local estimation

All these results are packaged into a single Python dictionary per frame and returned to `main.py`. The annotated frame (with bounding boxes drawn on it) is also returned so it can be displayed on screen.

### `src/storage.py` — Saves the data every 30 seconds

The detector produces one dictionary per frame (roughly 10–14 frames per 30-second window). This file takes all those frame dictionaries, averages them into a single summary row, and writes that row to two places simultaneously:

- **SQLite database** (`data/kabukicho.db`) — structured, queryable, permanent storage
- **CSV file** (`data/kabukicho.csv`) — portable, openable in Excel or pandas

It also assigns an **activity label** (QUIET / MODERATE / BUSY / CROWDED / NIGHT-QUIET / NIGHT-BUSY / NIGHT-PEAK) based on crowd size, tourist ratio, and brightness. Every time a row is saved, a one-line summary is printed to the terminal.

### `src/event_engine.py` — Generates events when something relevant happens

After every 30-second window is saved, this file checks the aggregated data against a set of rules and fires named events when thresholds are crossed. Events are printed to the terminal in colour and saved to `data/kabukicho_events.csv` and the SQLite database.

| Event | Trigger | Severity |
|---|---|---|
| `CROWD_SURGE` | Crowd jumps +3 people vs previous window | HIGH |
| `CROWD_DROP` | Crowd falls -3 people vs previous window | MEDIUM |
| `CROWD_PEAK` | Peak crowd in a single frame reaches 12+ people | HIGH |
| `ACTIVITY_CHANGE` | Activity label changes (e.g. MODERATE → BUSY) | MEDIUM |
| `NIGHT_TRANSITION` | Brightness drops below 90 (day → night) | HIGH |
| `DAY_TRANSITION` | Brightness rises above 90 (night → day) | INFO |
| `LARGE_GROUP` | New maximum group size detected (8+ people) | INFO |
| `GENDER_SKEW` | Male ratio exceeds 95% in a window | INFO |
| `DIRECTION_REVERSAL` | Walking direction flips left ↔ right | MEDIUM |

A cooldown system prevents the same event from firing repeatedly — each event type has a minimum gap before it can fire again.

### `src/main.py` — The recording loop (run this to start)

This is the entry point you actually run. It ties everything together:

1. Opens the stream using `stream_reader.py`
2. Creates a `Detector` and a `DataStore`
3. Reads frames in a loop, skipping every N frames (configurable) to save CPU
4. Sends each kept frame to the detector
5. Collects results for 30 seconds, then calls `storage.save_window()`
6. Shows the annotated frame in a window (unless `--no-display` is used)
7. Stops when the duration expires or you press **q**

### `src/analysis.py` — Generates the 6 charts (run this after recording)

After recording, this file reads all rows from the SQLite database and produces six PNG chart files in the `output/` folder. It automatically loads both the daytime and evening databases to produce comparison charts:

| Chart | What it shows |
|---|---|
| `01_dashboard.png` | High-level KPI overview — crowd, tourist index, gender split, friendliness |
| `03_tourist_local.png` | Tourist vs local balance, accessories, tourist trend |
| `04_demographics.png` | Adult/child breakdown and solo vs group behaviour |
| `05_crowd_comparison.png` | Crowd metrics side-by-side: daytime vs evening |
| `06_gender_comparison.png` | Gender distribution comparison: daytime vs evening |
| `07_correlations.png` | Heatmap showing which metrics are statistically related |

It also prints a short text report to the terminal summarising the session.

### `src/generate_demo_data.py` — Creates fake test data (for offline testing only)

This file generates 14 hours of realistic-looking but completely **made-up** data. It simulates Kabukicho patterns: a quiet morning, a tourist-heavy afternoon, and a busy neon-lit night peak. It is useful for testing that the charts and database work correctly **without needing to connect to a real stream**. The data it creates must be deleted before you do a real recording session — otherwise your charts will mix fake rows with real ones.

---

## Configuration (`src/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `STREAM_URL` | Kabukicho YouTube cam | Any YouTube live, RTSP, or HLS URL |
| `AGGREGATION_WINDOW_SECONDS` | `30` | One database row per N seconds |
| `FRAME_SKIP` | `3` | Process every Nth frame |
| `YOLO_OBJ_MODEL` | `yolov8n.pt` | Object detection model size |
| `YOLO_POSE_MODEL` | `yolov8n-pose.pt` | Pose estimation model |
| `CONFIDENCE` | `0.35` | Minimum detection confidence |
| `ADULT_MIN_RATIO` | `0.22` | Bbox height fraction → adult |
| `CHILD_MAX_RATIO` | `0.14` | Bbox height fraction → child |
| `GENDER_MIN_BBOX_HEIGHT` | `80` px | Minimum person height for gender estimation |
| `GENDER_CONFIDENCE_MIN` | `60` % | DeepFace confidence threshold |
| `GENDER_MAX_PER_FRAME` | `6` | Max people estimated per frame (speed limit) |
| `GENDER_FRAME_SKIP` | `2` | Run gender estimation every Nth processed frame |

---

## Database schema

```sql
CREATE TABLE observations (
    id                  INTEGER PRIMARY KEY,
    timestamp           TEXT,      -- UTC time the row was written
    window_start        TEXT,      -- UTC start of the 30-second window
    crowd_avg           REAL,      -- mean people per frame
    crowd_max           INTEGER,   -- peak people in any single frame
    crowd_density_avg   REAL,      -- fraction of frame area covered by people
    adults_avg          REAL,
    children_avg        REAL,
    tourists_avg        REAL,
    locals_avg          REAL,
    tourist_index       REAL,      -- tourists / crowd_avg
    shopping_bags_avg   REAL,
    suitcases_avg       REAL,
    backpacks_avg       REAL,
    umbrellas_avg       REAL,
    waving_avg          REAL,
    friendliness_idx    REAL,      -- waving / crowd_avg
    male_avg            REAL,      -- estimated males per frame
    female_avg          REAL,      -- estimated females per frame
    gender_unknown_avg  REAL,      -- people without confident gender estimate
    male_ratio          REAL,      -- male / (male + female)
    direction_balance   REAL,      -- + rightward  /  - leftward
    movement_speed_avg  REAL,
    speed_slow_pct      REAL,      -- fraction of frames labelled SLOW
    speed_fast_pct      REAL,
    solo_avg            REAL,
    group_count_avg     REAL,
    largest_group       INTEGER,
    brightness_avg      REAL,      -- 0-255, proxy for day vs night
    adult_ratio         REAL,
    luggage_ratio       REAL,
    frame_count         INTEGER,
    activity            TEXT       -- QUIET / MODERATE / BUSY / CROWDED / NIGHT-*
);
```

---

## Gender estimation — how it works

For each person detected by YOLO:
1. The upper 45% of the bounding box is cropped (head + upper body)
2. The crop is passed to **DeepFace** (VGG-Face based gender classifier)
3. If the model returns ≥ 60% confidence, the person is counted as male or female
4. Smaller or distant people (bbox height < 80 px) go to "unknown"
5. To keep the pipeline fast, max 6 people are estimated per frame, every 2nd processed frame

> **Important:** gender is estimated from visual appearance only. It is a statistical approximation at crowd level, not an individual classification.

---

## Big Data context

This application is designed to scale from one camera to a city-wide network:

| Concern | Approach |
|---|---|
| **Ingest** | Replace CSV sink with Apache Kafka — each 30-second window becomes a message |
| **Storage** | Swap SQLite for Apache Cassandra (time-series) or BigQuery |
| **Processing** | Apache Spark Structured Streaming across hundreds of cameras simultaneously |
| **Serving** | REST API or Grafana dashboard for real-time city monitoring |
| **Use cases** | Tourist flow prediction, crowd surge alerts, retail staffing, gender-based marketing insights, event impact measurement |

---

*Submitted for the IoT & Big Data postgraduate programme.*
