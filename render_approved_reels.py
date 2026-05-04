#!/usr/bin/env python3
"""
Render Approved Reels — Path C, Step 3
=======================================
Takes the user's review_selections.json (from review/index.html)
and renders the final per-fault reels using only approved moments.

Reuses the per_fault_reels.render_fault_reel infrastructure but with
manually-curated frame indices. Re-runs pose detection on each
approved frame from the source video so the skeleton matches what
the user saw in the review UI.

Usage:
  python render_approved_reels.py jose_guardado_h264.mp4 \\
      --selections review_selections.json \\
      --candidates site/review/candidates.json \\
      --outdir site/reels --slowmo 4
"""

import os, sys, argparse, json, time
import numpy as np
import cv2
import torch

sys.stdout.reconfigure(line_buffering=True)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
import per_fault_reels as pfr


def build_pose_landmarker(model_path):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def pose_on_bbox(landmarker, frame_bgr, box_xyxy, ts_ms, w, h, padding=0.10):
    if box_xyxy is None: return None
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    bw = x2 - x1; bh = y2 - y1
    pad_x = int(bw * padding); pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1: return None
    canvas = np.zeros_like(frame_bgr)
    canvas[y1:y2, x1:x2] = frame_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    r = landmarker.detect(mp_img)
    return r.pose_landmarks[0] if r.pose_landmarks else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--selections", required=True,
                   help="review_selections.json from the review UI")
    p.add_argument("--candidates", required=True,
                   help="candidates.json from dinov2_reid_pipeline")
    p.add_argument("--outdir", default="site/reels")
    p.add_argument("--slowmo", type=int, default=4)
    args = p.parse_args()

    with open(args.selections) as f:
        selections = json.load(f)["selections"]
    with open(args.candidates) as f:
        candidates = json.load(f)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    pose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pose_landmarker_full.task")
    landmarker = build_pose_landmarker(pose_path)
    detector = PatternDetector()

    print(f"[Render] {args.source} {W}x{H} @ {fps:.0f}fps", flush=True)
    print(f"[Render] {len(selections)} faults to render", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    for fault_name, sel in selections.items():
        approved = sel["approved"]
        if not approved:
            print(f"  → {fault_name}: 0 approved, skipping", flush=True)
            continue

        # Build a metas list spanning all approved frames + windows around them.
        # Need pose for surrounding frames too because the renderer does
        # lead-in + freeze + slowmo around each picked moment.
        approved_idxs = sorted([a["frame_idx"] for a in approved])
        slug = sel["slug"]

        # Collect all frame indices we need (approved + ±25 frames around each)
        WINDOW = 30
        needed_frames = set()
        for idx in approved_idxs:
            for off in range(-WINDOW, WINDOW + 1):
                if off >= -25:  # match pfr render's SLOWMO_R window
                    needed_frames.add(idx + off)
        needed_frames = sorted(f for f in needed_frames if f >= 0)
        if not needed_frames: continue

        # Look up each approved candidate's bbox from candidates.json
        bbox_map = {}
        for c in candidates[fault_name]["candidates"]:
            bbox_map[c["frame_idx"]] = c["bbox"]
        # Use the nearest known bbox for surrounding frames
        def nearest_bbox(idx):
            if not bbox_map: return None
            keys = list(bbox_map.keys())
            return bbox_map[min(keys, key=lambda k: abs(k - idx))]

        # Run pose for all needed frames
        cap = cv2.VideoCapture(args.source)
        metas = []
        print(f"  → {fault_name}: {len(approved_idxs)} approved, building "
              f"{len(needed_frames)} frame metas", flush=True)
        last_idx = -1
        for fi in needed_frames:
            if fi != last_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            last_idx = fi
            if not ret:
                metas.append(pfr.FrameMeta(idx=fi, lms=None, analysis=FrameAnalysis()))
                continue
            bbox = nearest_bbox(fi)
            ts_ms = int(fi * 1000 / fps)
            lms = pose_on_bbox(landmarker, frame, bbox, ts_ms, W, H)
            analysis = detector.analyze(lms, W, H) if lms else FrameAnalysis()
            metas.append(pfr.FrameMeta(idx=fi, lms=lms, analysis=analysis))
        cap.release()

        # The renderer needs FrameMeta for every consecutive index in
        # the range, so fill gaps with None metas.
        idx_set = {m.idx for m in metas}
        full_metas = []
        min_i, max_i = min(needed_frames), max(needed_frames)
        meta_by_idx = {m.idx: m for m in metas}
        for i in range(min_i, max_i + 1):
            if i in meta_by_idx:
                full_metas.append(meta_by_idx[i])
            else:
                full_metas.append(pfr.FrameMeta(idx=i, lms=None, analysis=FrameAnalysis()))

        # render_fault_reel takes a simple list of global frame indices
        # (used as keys into meta_by_idx inside the renderer).
        # If a fault detection didn't fire on the user's approved frame,
        # we synthesize one so the renderer can still pick it up.
        for global_idx in approved_idxs:
            fm = meta_by_idx.get(global_idx)
            if fm and fm.lms:
                has_fault = any(f.name == fault_name for f in fm.analysis.faults)
                if not has_fault:
                    # Inject a synthetic Fault so the renderer can find it
                    fm.analysis.faults.append(Fault(
                        name=fault_name, severity="warning",
                        description="Manually approved",
                        affected_landmarks=[], confidence=0.5))

        out_path = os.path.join(args.outdir, f"{slug}.mp4")
        pfr.render_fault_reel(
            args.source, full_metas, fault_name, approved_idxs,
            out_path, fps, W, H, scale=1.0, slowmo=args.slowmo,
        )
        print(f"  → wrote {out_path}", flush=True)

    landmarker.close()
    print("\n[DONE] approved reels rendered", flush=True)


if __name__ == "__main__":
    main()
