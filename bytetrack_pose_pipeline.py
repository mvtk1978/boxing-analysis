#!/usr/bin/env python3
"""
ByteTrack + Pose Pipeline for Guardado-Only Boxing Analysis
============================================================
Lightweight production-grade alternative to SAM 2:

  1. YOLOv8 detects people per frame (with persistent track IDs from
     BoT-SORT / ByteTrack).
  2. We identify which track ID is GUARDADO once, in a seed frame
     (manual bbox or color-heuristic), then trust the tracker to
     keep that ID across the fight.
  3. If the Guardado track is lost (occlusion / fast motion), we
     re-identify using IoU + color similarity to the last known box.
  4. Pose is run only on Guardado's cropped bbox each frame.
  5. Same fault detector + per-fault reel renderer as before.

Memory profile: ~500 MB peak (no full-frame tensor preload).
Speed: ~30 fps on RTX 4080, ~5-8 fps CPU-only.

Usage:
  python bytetrack_pose_pipeline.py video.mp4 \\
      --start 98 --duration 419 \\
      --seed-frame-time 130 --seed-box 208,163,638,715 \\
      --outdir site/reels
"""

import os, sys, argparse, time, json
import numpy as np
import cv2
import torch
from typing import Optional, List, Dict, Tuple

sys.stdout.reconfigure(line_buffering=True)

from ultralytics import YOLO

# Pose: MediaPipe (drop-in; can be swapped for ViTPose later)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point
import per_fault_reels as pfr


# ─── Detection + Tracking via YOLOv8 ────────────────────────────────────────

def build_yolo_tracker(model_size: str = "x"):
    """Return a YOLOv8 model with built-in tracking."""
    weights = f"yolov8{model_size}.pt"
    print(f"  [YOLO] loading {weights}", flush=True)
    return YOLO(weights)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_guardado_color(frame_bgr, box_xyxy, w, h) -> bool:
    """
    Hard color gate: even if the tracker says it's Guardado, verify the
    bbox is on a DARK-trunked boxer with NO green leakage.
    Returns True only if confident this is Guardado.
    """
    green, dark = shorts_color_score(frame_bgr, box_xyxy, w, h)
    # Must be substantially dark AND no significant green pixels
    return dark > 25.0 and green < 8.0


def shorts_color_score(frame_bgr, box_xyxy, w, h):
    """
    Sample the lower-third of the bbox (shorts area).
    Returns (green_score, dark_score) — higher green = Conceição,
    higher dark = Guardado-likely.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    bh = y2 - y1
    # Shorts area: from 55% to 80% down the body
    shorts_y1 = y1 + int(bh * 0.55)
    shorts_y2 = y1 + int(bh * 0.80)
    if shorts_y2 <= shorts_y1:
        return 0.0, 0.0
    patch = frame_bgr[shorts_y1:shorts_y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Green pixels
    green_mask = (h_ >= 40) & (h_ <= 85) & (s_ > 60)
    green_score = float(green_mask.mean()) * 100
    # Dark pixels (not green)
    dark_mask = (v_ < 80) & ~green_mask
    dark_score = float(dark_mask.mean()) * 100
    return green_score, dark_score


# ─── Identify Guardado from track candidates ────────────────────────────────

def find_guardado_track_id(seed_results, frame_bgr, seed_box: Tuple[int,int,int,int]):
    """
    From YOLO results on the seed frame, find the track ID whose bbox best
    overlaps the user-provided seed_box. Return that track ID.
    """
    if seed_results.boxes.id is None:
        raise RuntimeError("Tracker did not assign IDs on seed frame — try running for a few frames first")
    boxes = seed_results.boxes.xyxy.cpu().numpy()
    ids   = seed_results.boxes.id.cpu().numpy().astype(int)
    cls   = seed_results.boxes.cls.cpu().numpy().astype(int)
    # Person class in COCO = 0
    best_id, best_iou = None, 0.0
    print(f"  [seed] {len(boxes)} detections in seed frame", flush=True)
    for box, tid, c in zip(boxes, ids, cls):
        if c != 0: continue
        iou = iou_xyxy(box, seed_box)
        print(f"    track #{tid} bbox={box.astype(int).tolist()} IoU_with_seed={iou:.2f}", flush=True)
        if iou > best_iou:
            best_iou = iou
            best_id = int(tid)
    if best_id is None or best_iou < 0.3:
        raise RuntimeError(f"No track sufficiently overlapped seed bbox (best IoU={best_iou:.2f})")
    print(f"  [seed] GUARDADO = track #{best_id} (IoU {best_iou:.2f})", flush=True)
    return best_id


def re_id_guardado(results, frame_bgr, last_box, w, h):
    """
    Track lost — try to recover but ONLY if a candidate confidently
    passes the Guardado color test. Otherwise return None (skip frame).
    """
    if results.boxes.id is None or len(results.boxes) == 0:
        return None, None
    boxes = results.boxes.xyxy.cpu().numpy()
    ids   = results.boxes.id.cpu().numpy().astype(int)
    cls   = results.boxes.cls.cpu().numpy().astype(int)
    candidates = []
    for box, tid, c in zip(boxes, ids, cls):
        if c != 0: continue
        if not is_guardado_color(frame_bgr, box, w, h):
            continue   # HARD reject any non-Guardado coloring
        iou = iou_xyxy(box, last_box) if last_box is not None else 0.0
        candidates.append((box, int(tid), iou))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: -x[2])
    best = candidates[0]
    return best[0], best[1]


# ─── Pose on cropped Guardado bbox ──────────────────────────────────────────

def build_pose_landmarker(model_path: str):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def pose_on_bbox(landmarker, frame_bgr, box_xyxy, ts_ms,
                 w, h, padding=0.10):
    """
    Crop the bbox (with padding), zero-out everything else in a full-size
    canvas, run pose. Returns landmarks in the FULL-FRAME coordinate
    system so downstream fault detection works unchanged.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    bw = x2 - x1; bh = y2 - y1
    pad_x = int(bw * padding); pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    # Full-frame canvas with everything outside the bbox zeroed
    canvas = np.zeros_like(frame_bgr)
    canvas[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    r = landmarker.detect_for_video(mp_img, ts_ms)
    return r.pose_landmarks[0] if r.pose_landmarks else None


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--outdir",          default="site/reels")
    p.add_argument("--start",           type=float, default=98.0)
    p.add_argument("--duration",        type=float, default=419.0)
    p.add_argument("--seed-frame-time", type=float, default=130.0,
                   help="Time (sec) of the SEED frame where Guardado is identified")
    p.add_argument("--seed-box",        type=str, required=True,
                   help="Manual seed bbox 'x1,y1,x2,y2' for Guardado at seed time")
    p.add_argument("--per-fault",       type=int, default=4)
    p.add_argument("--slowmo",          type=int, default=4)
    p.add_argument("--yolo",            default="x", choices=["n","s","m","l","x"])
    p.add_argument("--tracker-cfg",     default="botsort.yaml",
                   help="Ultralytics tracker config (botsort.yaml or bytetrack.yaml)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = int(args.start * fps)
    end_f = min(total, start_f + int(args.duration * fps))
    seed_f_global = int(args.seed_frame_time * fps)
    seed_box = tuple(int(x) for x in args.seed_box.split(","))

    print(f"[Video] {args.source} {W}x{H} @ {fps:.0f}fps", flush=True)
    print(f"[Window] frames {start_f}..{end_f} ({(end_f-start_f)/fps:.0f}s)", flush=True)
    print(f"[Seed] Guardado bbox {seed_box} at frame {seed_f_global} (t={args.seed_frame_time}s)", flush=True)

    # ── Setup ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}", flush=True)

    yolo = build_yolo_tracker(args.yolo)
    pose_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "pose_landmarker_full.task")
    landmarker = build_pose_landmarker(pose_model_path)
    detector = PatternDetector()

    # ── Pass 1: seed via short video segment ending at seed frame ───────
    # YOLOv8 tracking needs sequential frames; we run from start_f and
    # capture at the seed frame.
    print("\n[Pass 1] tracking + pose on Guardado", flush=True)

    metas: List[pfr.FrameMeta] = []
    last_box = None
    guardado_tid: Optional[int] = None

    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fi = start_f
    t0 = time.time()
    last_status = t0

    track_lost_count = 0
    track_recovered_count = 0
    color_rejected_count = 0
    pose_found_count = 0

    while cap.isOpened() and fi < end_f:
        ret, frame = cap.read()
        if not ret: break

        # YOLO + tracker
        results = yolo.track(
            frame, classes=[0], persist=True, verbose=False,
            tracker=args.tracker_cfg, device=device,
        )[0]

        # On seed frame, identify Guardado track ID
        if fi == seed_f_global:
            try:
                guardado_tid = find_guardado_track_id(results, frame, seed_box)
            except Exception as e:
                print(f"  [WARN] seed identification failed: {e}", flush=True)

        # Find Guardado box in this frame
        gbox = None
        if guardado_tid is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            mask = ids == guardado_tid
            if mask.any():
                gbox = results.boxes.xyxy.cpu().numpy()[mask][0]

        if gbox is None and guardado_tid is not None:
            # Track lost — try to re-identify (with strict color verification)
            track_lost_count += 1
            recovered_box, recovered_id = re_id_guardado(
                results, frame, last_box, W, H)
            if recovered_box is not None:
                if recovered_id != guardado_tid:
                    guardado_tid = recovered_id
                    track_recovered_count += 1
                gbox = recovered_box

        # ── HARD COLOR GATE ──
        # Even if the tracker says it's Guardado, verify the bbox is on a
        # dark-trunked boxer with no green leakage. If it fails, drop the
        # frame entirely (better no data than wrong-boxer data).
        if gbox is not None and not is_guardado_color(frame, gbox, W, H):
            color_rejected_count += 1
            gbox = None

        if gbox is not None:
            last_box = gbox
            ts_ms = int(fi * 1000 / fps)
            lms = pose_on_bbox(landmarker, frame, gbox, ts_ms, W, H)
            if lms is not None:
                pose_found_count += 1
            analysis = detector.analyze(lms, W, H) if lms else FrameAnalysis()
            metas.append(pfr.FrameMeta(idx=fi, lms=lms, analysis=analysis))
        else:
            metas.append(pfr.FrameMeta(idx=fi, lms=None, analysis=FrameAnalysis()))

        fi += 1
        if time.time() - last_status > 5:
            done = fi - start_f
            rate = done / max(0.1, time.time() - t0)
            eta = (end_f - fi) / max(0.1, rate)
            print(f"  frame {fi}/{end_f} ({done/(end_f-start_f)*100:.0f}%) | {rate:.1f} fps | ETA {eta:.0f}s | pose={pose_found_count} lost={track_lost_count} recovered={track_recovered_count} color_rejected={color_rejected_count}", flush=True)
            last_status = time.time()

    cap.release()
    landmarker.close()

    print(f"\n[Pass 1] done in {time.time()-t0:.0f}s", flush=True)
    print(f"  total frames analyzed: {len(metas)}", flush=True)
    print(f"  Guardado pose detected: {pose_found_count}", flush=True)
    print(f"  track lost events: {track_lost_count} (recovered: {track_recovered_count})", flush=True)
    print(f"  color-gate rejections (wrong-boxer leakage prevented): {color_rejected_count}", flush=True)

    summary = detector.get_session_summary()
    print("\n[Top faults]", flush=True)
    for name, pct in list(summary.items())[:8]:
        print(f"  {name:<40} {pct:5.1f}%", flush=True)

    # Save summary
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "method": "YOLOv8 + ByteTrack/BoT-SORT + MediaPipe pose on bbox crop",
            "frames_analyzed": len(metas),
            "guardado_pose_frames": pose_found_count,
            "track_lost": track_lost_count,
            "track_recovered": track_recovered_count,
            "color_rejected": color_rejected_count,
            "fault_percentages": summary,
            "seed_box": list(seed_box),
            "seed_frame_time": args.seed_frame_time,
        }, f, indent=2)

    # ── Pass 2: per-fault reels ─────────────────────────────────────────
    print(f"\n[Pass 2] rendering per-fault reels → {args.outdir}", flush=True)
    rendered = 0
    for fault_name in summary.keys():
        slug = pfr.slugify(fault_name)
        out = os.path.join(args.outdir, f"{slug}.mp4")
        idxs = pfr.select_top_per_fault(metas, fault_name, args.per_fault, min_gap_frames=80)
        print(f"  → {fault_name} ({len(idxs)} moments)", flush=True)
        if pfr.render_fault_reel(args.source, metas, fault_name, idxs,
                                 out, fps, W, H, scale=1.0, slowmo=args.slowmo):
            rendered += 1

    print(f"\n[DONE] {rendered} per-fault reels written to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
