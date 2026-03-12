#!/usr/bin/env python3
"""
Two-Fighter Boxing Analysis
Detects BOTH fighters separately, assigns left/right roles,
runs independent fault analysis, and renders a split coaching panel.

Usage:
  python analyze_two.py <video_path> [--output result.mp4] [--start SEC] [--duration SEC]
"""

import sys
import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis
from boxing_analyzer.landmarks import LM, get_point, get_point_norm
from boxing_analyzer.video_io import download_youtube, ensure_h264, is_youtube_url

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Skeleton connections
POSE_CONNECTIONS = [
    (LM.LEFT_EAR, LM.NOSE), (LM.RIGHT_EAR, LM.NOSE),
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]

COLORS_A = {  # Left fighter — blue tones
    "skeleton": (255, 180, 50),
    "fault": (50, 50, 255),
    "header": (255, 200, 50),
    "panel": (10, 10, 25),
}
COLORS_B = {  # Right fighter — red tones
    "skeleton": (50, 230, 120),
    "fault": (50, 50, 255),
    "header": (50, 220, 50),
    "panel": (10, 25, 10),
}


def build_pose_detector(model_path: str, num_poses: int = 2):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def torso_center_x(lms, w):
    """Get horizontal center of a pose (avg of shoulders + hips)."""
    pts = []
    for idx in [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP]:
        pts.append(lms[idx].x * w)
    return float(np.mean(pts))


def draw_skeleton(frame, lms, w, h, fault_set, colors, alpha=0.9):
    overlay = frame.copy()
    for a, b in POSE_CONNECTIONS:
        try:
            pa = get_point(lms, a, w, h).astype(int)
            pb = get_point(lms, b, w, h).astype(int)
            is_fault = (a in fault_set) or (b in fault_set)
            color = colors["fault"] if is_fault else colors["skeleton"]
            cv2.line(overlay, tuple(pa), tuple(pb), color, 3 if is_fault else 2, cv2.LINE_AA)
        except Exception:
            pass
    for idx in range(33):
        try:
            pt = get_point(lms, idx, w, h).astype(int)
            is_fault = idx in fault_set
            color = colors["fault"] if is_fault else colors["skeleton"]
            r = 7 if is_fault else 4
            cv2.circle(overlay, tuple(pt), r, color, -1, cv2.LINE_AA)
        except Exception:
            pass
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def build_fighter_panel(label: str, analysis: FrameAnalysis, summary: dict,
                        h: int, pw: int, colors: dict) -> np.ndarray:
    panel = np.zeros((h, pw, 3), dtype=np.uint8)
    panel[:] = colors["panel"]
    cv2.line(panel, (0, 0), (0, h), (60, 60, 60), 2)

    y = 18
    cv2.putText(panel, label, (10, y), FONT, 0.62, colors["header"], 2, cv2.LINE_AA)
    y += 20
    cv2.line(panel, (8, y), (pw - 8, y), (50, 50, 50), 1)
    y += 14

    stance_color = (200, 180, 80) if "orthodox" in analysis.stance else (80, 200, 180)
    cv2.putText(panel, f"Stance: {analysis.stance.upper()}", (10, y), FONT, 0.46, stance_color, 1, cv2.LINE_AA)
    y += 22
    cv2.line(panel, (8, y), (pw - 8, y), (40, 40, 40), 1)
    y += 12

    # Live faults
    cv2.putText(panel, "LIVE FAULTS", (10, y), FONT, 0.46, (140, 140, 140), 1, cv2.LINE_AA)
    y += 20

    if not analysis.faults:
        cv2.putText(panel, "  Clean", (10, y), FONT, 0.44, (50, 180, 50), 1, cv2.LINE_AA)
        y += 18
    else:
        for f in analysis.faults[:5]:
            icon = "!!" if f.severity == "critical" else "! "
            fc = (50, 50, 220) if f.severity == "critical" else (50, 140, 220)
            cv2.putText(panel, f"{icon} {f.name}", (10, y), FONT, 0.42, fc, 1, cv2.LINE_AA)
            y += 17

    y += 6
    cv2.line(panel, (8, y), (pw - 8, y), (40, 40, 40), 1)
    y += 12

    # Session summary bars
    cv2.putText(panel, "SESSION %", (10, y), FONT, 0.46, (140, 140, 140), 1, cv2.LINE_AA)
    y += 20
    for name, pct in list(summary.items())[:7]:
        bar_w = int((pct / 100) * (pw - 30))
        bar_color = (50, 50, 180) if pct > 60 else (50, 120, 50)
        cv2.rectangle(panel, (10, y - 9), (10 + bar_w, y + 2), bar_color, -1)
        short = name[:24]
        cv2.putText(panel, f"{short} {pct:.0f}%", (12, y), FONT, 0.36, (210, 210, 210), 1, cv2.LINE_AA)
        y += 17
        if y > h - 20:
            break

    return panel


def draw_fault_labels(frame, lms, analysis, w, h, side: str):
    """Pin fault labels on the body."""
    shown = set()
    offset_x = 8 if side == "right" else -8
    for f in analysis.faults[:3]:
        if f.name in shown:
            continue
        shown.add(f.name)
        try:
            lm_idx = f.affected_landmarks[0]
            pt = get_point(lms, lm_idx, w, h).astype(int)
            label = f.name
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            lx = pt[0] + 8 if side == "right" else pt[0] - tw - 8
            lx = max(4, min(lx, w - tw - 6))
            ly = max(th + 4, min(pt[1] - 6, h - 4))
            bg = (30, 30, 180) if f.severity == "critical" else (30, 100, 180)
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), bg, -1)
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (255, 255, 255), 1)
            cv2.putText(frame, label, (lx, ly), FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass


def run(args):
    video_path = args.source
    if is_youtube_url(video_path):
        video_path = download_youtube(video_path)
    video_path = ensure_h264(video_path)

    model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    landmarker = build_pose_detector(model_path, num_poses=2)
    print("[Pose] 2-person detector ready")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {os.path.basename(video_path)}  {orig_w}x{orig_h} @ {fps:.0f}fps  {total_frames} frames")

    start_frame = int(args.start * fps)
    end_frame = min(total_frames, start_frame + int(args.duration * fps) if args.duration else total_frames)

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scale = args.scale
    W = int(orig_w * scale)
    H = int(orig_h * scale)
    PANEL_W = 220

    # Two independent detectors — one per fighter
    det_left  = PatternDetector()
    det_right = PatternDetector()

    out_w = W + PANEL_W * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, H))
    print(f"[Output] {args.output}  ({out_w}x{H})")

    frame_idx = start_frame
    processed = 0
    pulse = 0

    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pulse = (pulse + 4) % 360

        if scale != 1.0:
            frame = cv2.resize(frame, (W, H))

        # ---- Pose detection (up to 2 people) ----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_img, ts_ms)
        all_lms = result.pose_landmarks  # list of up to 2 pose landmark lists

        # Sort by horizontal center → index 0 = left fighter, 1 = right fighter
        if len(all_lms) >= 2:
            centers = [(torso_center_x(lms, W), lms) for lms in all_lms]
            centers.sort(key=lambda x: x[0])
            lms_left  = centers[0][1]
            lms_right = centers[1][1]
        elif len(all_lms) == 1:
            cx = torso_center_x(all_lms[0], W)
            lms_left  = all_lms[0] if cx < W * 0.5 else None
            lms_right = all_lms[0] if cx >= W * 0.5 else None
        else:
            lms_left = lms_right = None

        # ---- Analyze each fighter ----
        analysis_left = analysis_right = None
        summary_left = summary_right = {}

        if lms_left:
            analysis_left  = det_left.analyze(lms_left, W, H)
            summary_left   = det_left.get_session_summary()
            fault_set_left = {lm for f in analysis_left.faults for lm in f.affected_landmarks}
            draw_skeleton(frame, lms_left, W, H, fault_set_left, COLORS_A)
            draw_fault_labels(frame, lms_left, analysis_left, W, H, "right")

        if lms_right:
            analysis_right  = det_right.analyze(lms_right, W, H)
            summary_right   = det_right.get_session_summary()
            fault_set_right = {lm for f in analysis_right.faults for lm in f.affected_landmarks}
            draw_skeleton(frame, lms_right, W, H, fault_set_right, COLORS_B)
            draw_fault_labels(frame, lms_right, analysis_right, W, H, "left")

        # Animated pulse on critical fault joints
        pr = int(14 + 5 * abs(np.sin(np.radians(pulse))))
        for lms_x, analysis_x in [(lms_left, analysis_left), (lms_right, analysis_right)]:
            if lms_x and analysis_x:
                for f in analysis_x.faults:
                    if f.severity == "critical":
                        for lm_idx in f.affected_landmarks[:1]:
                            try:
                                pt = get_point(lms_x, lm_idx, W, H).astype(int)
                                cv2.circle(frame, tuple(pt), pr, (0, 0, 255), 2, cv2.LINE_AA)
                            except Exception:
                                pass

        # ---- Panels ----
        dummy = FrameAnalysis()
        dummy.stance = "unknown"
        panel_left  = build_fighter_panel(
            "FIGHTER LEFT",
            analysis_left  or dummy, summary_left,  H, PANEL_W, COLORS_A)
        panel_right = build_fighter_panel(
            "FIGHTER RIGHT",
            analysis_right or dummy, summary_right, H, PANEL_W, COLORS_B)

        combined = np.hstack([panel_left, frame, panel_right])
        writer.write(combined)

        if args.show:
            cv2.imshow("Two-Fighter Analysis", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        processed += 1
        if processed % 30 == 0:
            pct = (frame_idx - start_frame) / max(1, end_frame - start_frame) * 100
            print(f"  {frame_idx}/{end_frame} ({pct:.0f}%)")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    landmarker.close()

    # ---- Final reports ----
    for label, det in [("FIGHTER LEFT", det_left), ("FIGHTER RIGHT", det_right)]:
        summary = det.get_session_summary()
        print(f"\n{'='*55}")
        print(f"  {label} — FAULT REPORT")
        print(f"{'='*55}")
        print(f"  Stance: {det.stance or 'unknown'}")
        for name, pct in summary.items():
            bar = "█" * int(pct / 2)
            print(f"  {name:<35} {pct:5.1f}% {bar}")
        if not summary:
            print("  No faults detected")

    print(f"\n  Saved: {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--output", default="two_fighter_analysis.mp4")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
