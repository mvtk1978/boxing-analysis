# Boxing Analysis Tool

AI-powered boxing video analysis using MediaPipe pose estimation. Detects and visualizes technique faults in real time — guard drops, chin exposure, elbow flares, stance problems, and more.

---

## Features

- **8 fault detectors** with severity levels (critical / warning)
- **Single-fighter** and **two-fighter simultaneous** analysis
- **Highlight reel** generation with freeze frames, slow motion, and coaching callouts
- **Social media vertical reel** (9:16, 720×1280) for Threads / Instagram / TikTok
- **YouTube download** support via yt-dlp (auto re-encodes AV1 to H.264)
- Per-session fault frequency report with % breakdowns

---

## Architecture

```
boxing/
├── analyze.py              # Single-fighter entry point
├── analyze_two.py          # Two-fighter simultaneous analysis
├── highlight_reel.py       # Highlight reel with freeze/slowmo + callout strip
├── social_reel.py          # Vertical 9:16 reel for social media
├── demo_synthetic.py       # Synthetic test without a real video
├── requirements.txt
├── pose_landmarker_full.task  # MediaPipe model (not in repo — download separately)
└── boxing_analyzer/
    ├── landmarks.py        # MediaPipe landmark indices + geometry helpers
    ├── pattern_detector.py # Core fault detection engine (stateful, temporal)
    ├── visualizer.py       # OpenCV overlay, skeleton drawing, coaching panel
    └── video_io.py         # YouTube download + H.264 re-encoding
```

### Data flow

```
Video / YouTube URL
      │
      ▼
video_io.py — yt-dlp download → ffmpeg H.264 re-encode (if AV1/VP9)
      │
      ▼
MediaPipe PoseLandmarker (Tasks API, VIDEO mode)
  33-point body skeleton per frame
      │
      ▼
pattern_detector.py — PatternDetector.analyze()
  Geometric ratios normalized by body_height
  Temporal deque (45-frame / 1.5s window)
  Returns FrameAnalysis { faults, metrics, stance, action }
      │
      ▼
visualizer.py / highlight_reel.py / social_reel.py
  OpenCV skeleton overlay
  Fault annotations
  Coaching panel / callout strip
      │
      ▼
Output .mp4
```

---

## Installation

### Prerequisites

- Python 3.9+
- `ffmpeg` and `ffprobe` in PATH
- `yt-dlp` in PATH (for YouTube downloads)

### Setup

```bash
cd boxing
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download MediaPipe model

```bash
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
     -O pose_landmarker_full.task
```

---

## Usage

### Single-fighter analysis

```bash
# Analyze a local video file
python3 analyze.py /path/to/fight.mp4 --output result.mp4

# Analyze from YouTube
python3 analyze.py 'https://youtu.be/VIDEO_ID' --output result.mp4

# Clip: start at 30s, analyze 60 seconds, half resolution
python3 analyze.py fight.mp4 --start 30 --duration 60 --scale 0.5

# Print summary only (no video output)
python3 analyze.py fight.mp4 --no-save
```

### Two-fighter analysis

```bash
python3 analyze_two.py 'https://youtu.be/VIDEO_ID' --output two_fighter.mp4

# With time window
python3 analyze_two.py fight.mp4 --start 10 --duration 120 --output analysis.mp4
```

Left fighter is shown with blue/gold skeleton; right fighter with green skeleton.
Each fighter gets an independent `PatternDetector` instance — fault tracking does not bleed between fighters.

### Highlight reel

```bash
# Generate a highlight reel from a YouTube video
python3 highlight_reel.py 'https://youtu.be/VIDEO_ID' --output highlights.mp4

# Control density
python3 highlight_reel.py fight.mp4 \
    --output highlights.mp4 \
    --max-highlights 10 \
    --freeze 25 \
    --slowmo 2
```

The reel uses a two-pass approach:
1. **Pass 1** — analyze all frames, collect fault moments
2. **Pick** — divide clip into thirds, pick highest-confidence fault per bucket per fault type, enforce minimum gap between highlights
3. **Pass 2** — render: normal play → freeze (N frames) → hold → slow motion (±22 frames × factor) → continue

Callouts appear in a **130px bottom strip** — the video area above stays unobstructed.

### Social media vertical reel (9:16)

```bash
python3 social_reel.py 'https://youtu.be/VIDEO_ID' \
    --output social.mp4 \
    --title "OPPONENT BREAKDOWN" \
    --subtitle "AI Fault Detection" \
    --highlights 6 \
    --slowmo 3 \
    --freeze 45

# From local file
python3 social_reel.py fight.mp4 --output social.mp4 --start 30 --duration 90
```

Output structure:
- 2.5s animated intro card (title + subtitle)
- Top 6 fault highlight moments (freeze → callout → slowmo)
- 4s summary card ("EXPLOITABLE PATTERNS" with frequency bars)

Format: 720×1280, H.264, yuv420p — ready for Threads / Instagram / TikTok upload.

### Synthetic demo (no video required)

```bash
python3 demo_synthetic.py
```

Generates a synthetic video with injected faults in the first half and clean technique in the second half. Useful for verifying the detection pipeline without downloading any video.

---

## Fault Detection Reference

All metrics are normalized by **body height** (shoulder-to-hip distance) to be scale-invariant.

| Fault | Trigger | Severity |
|---|---|---|
| Lead Hand Too Low | lead_guard_ratio < -0.05 | warning; critical if < -0.15 |
| Rear Hand Too Low | rear_guard_ratio < -0.08 | warning; critical if < -0.18 |
| Chin Up – Head Exposed | chin_elevation > 0.12 | warning; critical if > 0.20 |
| Lead/Rear Elbow Flared | elbow_angle > 110° | warning; critical if > 135° |
| Stance Too Wide | ankle_dist > 2.2× shoulder_width | warning |
| Stance Too Narrow | ankle_dist < 0.9× shoulder_width | warning |
| Trunk Leaning Forward/Backward | abs(trunk_lean) > 20° | warning; critical if > 35° |
| Static Head – No Head Movement | mean head displacement < 0.015 normalized (over 30-frame window) | warning |
| Feet Crossing | feet cross relative to shoulder orientation + 20px threshold | critical |

Punch detection (no fault, just classification):
- `jab` — lead wrist velocity > 0.15 body_height/frame
- `cross` — rear wrist velocity > 0.15 body_height/frame

### Stance detection

Orthodox vs southpaw is determined from shoulder orientation (which shoulder faces the opponent) and relative foot positions. Re-evaluated every 30 frames.

---

## Output Layout

### Single-fighter (`analyze.py`)

```
┌─────────────────────────────────┬─────────────────────┐
│                                 │  COACHING PANEL     │
│   Video with skeleton overlay   │  Stance: orthodox   │
│   Fault badges on joints        │  Live faults        │
│   Pulsing rings on critical     │  Metrics bars       │
│                                 │  Session summary %  │
└─────────────────────────────────┴─────────────────────┘
  video_width × video_height          400px
```

### Two-fighter (`analyze_two.py`)

```
┌──────────┬──────────────────────┬──────────┐
│  FIGHTER │                      │  FIGHTER │
│  LEFT    │  Video (both skels)  │  RIGHT   │
│  panel   │                      │  panel   │
└──────────┴──────────────────────┴──────────┘
   220px        video_width            220px
```

### Social reel (`social_reel.py`)

```
┌──────────────────────┐  720px wide
│                      │
│   Video (cropped     │  860px tall
│   center to 9:16)    │
│                      │
├──────────────────────┤
│  Callout strip:      │  420px tall
│  fault name +        │
│  description +       │
│  mode badge          │
└──────────────────────┘  1280px total
```

---

## Requirements

```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
yt-dlp>=2024.1.0
scipy>=1.11.0
```

**Note:** `mediapipe 0.10+` uses the Tasks API (`PoseLandmarker`). The legacy `mp.solutions.pose` API was removed in 0.10. `analyze.py` auto-detects and falls back to the legacy API if needed.

**OpenCV limitation:** OpenCV cannot decode AV1 video. `video_io.py` detects the codec with `ffprobe` and automatically re-encodes to H.264 via `ffmpeg` if needed.

---

## Security Notes

- YouTube URLs are validated using `urllib.parse` (not substring matching) in `video_io.py`
- Downloaded files are verified to remain within the download directory (path traversal prevention)
- Intermediate files during social reel generation use a secure temp directory (`tempfile.mkdtemp`) that is always cleaned up
- Numeric CLI arguments (`--scale`, `--slowmo`, `--freeze`, etc.) are range-validated before use
- Output paths are resolved and validated before being passed to subprocesses

---

## Limitations

- **Single plane detection:** MediaPipe pose works best when the full body is visible. Occlusion (fighters overlapping) degrades accuracy.
- **Two-fighter tracking:** Fighters are assigned left/right by horizontal center each frame — rapid position switches may cause brief mis-assignment.
- **Camera angle:** Faults like trunk lean depend on camera being roughly perpendicular to the fighters. Side-on camera angles will skew readings.
- **No audio:** Output videos are video-only (no audio track).

---

## Examples

```bash
# Full fight analysis with 30s clip from Lomachenko bout
python3 highlight_reel.py 'https://www.youtube.com/watch?v=BJe9f4ooOfQ' \
    --output loma_highlights.mp4 \
    --start 60 --duration 120

# Generate Instagram reel
python3 social_reel.py 'https://www.youtube.com/watch?v=BJe9f4ooOfQ' \
    --output loma_social.mp4 \
    --title "KOASICHA BREAKDOWN" \
    --subtitle "AI Fault Detection" \
    --start 60 --duration 90 --highlights 6

# Two-fighter comparison (shorts)
python3 analyze_two.py 'https://www.youtube.com/shorts/xLz6VsVKOwE' \
    --output two_fighter.mp4
```
