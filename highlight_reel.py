#!/usr/bin/env python3
"""
Boxing Highlight Reel Generator
================================
Callout-ы вынесены в полосу снизу — видео не перекрывается.
Структура кадра:
  ┌──────────────────┐
  │                  │  ← чистое видео (только скелет + маркеры)
  │    VIDEO AREA    │
  │                  │
  ├──────────────────┤
  │  CALLOUT STRIP   │  ← текст, иконки, описание
  └──────────────────┘

Usage:
  python highlight_reel.py <video_path> [--output reel.mp4] [--slowmo 4]
"""

import os, argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import List, Optional

from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point
from boxing_analyzer.video_io import ensure_h264, download_youtube

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

CONNECTIONS = [
    (LM.LEFT_EAR, LM.NOSE), (LM.RIGHT_EAR, LM.NOSE),
    (LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER,  LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW,     LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER,  LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP,       LM.RIGHT_HIP),
    (LM.LEFT_HIP,       LM.LEFT_KNEE),
    (LM.LEFT_KNEE,      LM.LEFT_ANKLE),
    (LM.RIGHT_HIP,      LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE,     LM.RIGHT_ANKLE),
]

STRIP_H = 130   # height of callout strip below video

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    frame:    np.ndarray
    lms:      object
    analysis: FrameAnalysis

@dataclass
class Highlight:
    frame_idx: int
    fault:     Fault
    lms:       object
    frame_img: np.ndarray


# ─── Skeleton drawing ─────────────────────────────────────────────────────────

def draw_skeleton(img, lms, w, h, fault_lm_set):
    """Skeleton only — no text on video."""
    ov = img.copy()
    for a, b in CONNECTIONS:
        try:
            pa = get_point(lms, a, w, h).astype(int)
            pb = get_point(lms, b, w, h).astype(int)
            fault = (a in fault_lm_set) or (b in fault_lm_set)
            col   = (0, 0, 200)   if fault else (50, 200, 50)
            glow  = (60, 60, 255) if fault else (120, 255, 120)
            cv2.line(ov, tuple(pa), tuple(pb), col,  5, cv2.LINE_AA)
            cv2.line(ov, tuple(pa), tuple(pb), glow, 2, cv2.LINE_AA)
        except Exception:
            pass
    for idx in range(33):
        try:
            pt    = get_point(lms, idx, w, h).astype(int)
            fault = idx in fault_lm_set
            outer = (0, 0, 160)    if fault else (20, 140, 20)
            inner = (100, 100, 255) if fault else (160, 255, 160)
            r = 9 if fault else 5
            cv2.circle(ov, tuple(pt), r + 3, outer, -1, cv2.LINE_AA)
            cv2.circle(ov, tuple(pt), r,     inner, -1, cv2.LINE_AA)
        except Exception:
            pass
    cv2.addWeighted(ov, 0.88, img, 0.12, 0, img)


def draw_fault_markers(img, lms, w, h, faults, pulse):
    """
    Small numbered badges on fault joints + pulsing ring.
    Numbers reference entries in the callout strip below.
    """
    shown = {}
    n = 1
    for f in faults[:4]:
        for lm_idx in f.affected_landmarks[:1]:
            if lm_idx in shown:
                continue
            shown[lm_idx] = n
            n += 1

    pr = int(16 + 6 * abs(np.sin(np.radians(pulse))))
    for lm_idx, num in shown.items():
        try:
            pt = get_point(lms, lm_idx, w, h).astype(int)
            # Pulsing ring
            cv2.circle(img, tuple(pt), pr,     (0, 0, 220), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pt), pr + 4, (0, 0, 120), 1, cv2.LINE_AA)
            # Number badge
            bx, by = pt[0] + 14, pt[1] - 14
            cv2.circle(img, (bx, by), 11, (0, 0, 180), -1, cv2.LINE_AA)
            cv2.circle(img, (bx, by), 11, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(num), (bx - 4, by + 5),
                        FONT_BOLD, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass


# ─── Callout strip (below video) ─────────────────────────────────────────────

def make_callout_strip(w, faults: list, mode: str, progress: float,
                       slowmo_factor: int = 1) -> np.ndarray:
    """
    Build the callout strip (w × STRIP_H).
    mode: "normal" | "freeze" | "slowmo"
    progress: 0→1 slide-in animation
    """
    strip = np.zeros((STRIP_H, w, 3), dtype=np.uint8)
    strip[:] = (12, 12, 18)

    # Top divider line
    cv2.line(strip, (0, 0), (w, 0), (40, 40, 60), 2)

    if mode == "normal" and not faults:
        cv2.putText(strip, "Analyzing...", (10, 30),
                    FONT, 0.45, (60, 60, 80), 1, cv2.LINE_AA)
        return strip

    p = min(1.0, progress)

    # Mode badge top-right
    if mode == "freeze":
        badge_txt = "PAUSED"
        badge_col = (40, 40, 160)
        badge_tc  = (150, 150, 255)
    elif mode == "slowmo":
        badge_txt = f"SLOW x1/{slowmo_factor}"
        badge_col = (10, 60, 80)
        badge_tc  = (0, 210, 255)
    else:
        badge_txt = ""
        badge_col = None

    if badge_col:
        (bw, bh), _ = cv2.getTextSize(badge_txt, FONT_BOLD, 0.5, 1)
        bx = w - bw - 20
        cv2.rectangle(strip, (bx - 6, 6), (w - 6, 26), badge_col, -1)
        cv2.rectangle(strip, (bx - 6, 6), (w - 6, 26), badge_tc, 1)
        cv2.putText(strip, badge_txt, (bx, 22), FONT_BOLD, 0.5, badge_tc, 1, cv2.LINE_AA)

    if not faults:
        return strip

    # Slide-in offset
    slide = int((1 - p) * 30)

    # Layout: up to 3 faults side by side in columns
    n        = min(len(faults), 3)
    col_w    = w // n
    y_top    = 14 + slide

    for i, f in enumerate(faults[:n]):
        cx = i * col_w
        # Separator between columns
        if i > 0:
            cv2.line(strip, (cx, 8), (cx, STRIP_H - 8), (40, 40, 55), 1)

        # Number circle
        num_x, num_y = cx + 16, y_top + 12
        num_col = (0, 0, 180) if f.severity == "critical" else (0, 100, 180)
        cv2.circle(strip, (num_x, num_y), 11, num_col, -1, cv2.LINE_AA)
        cv2.putText(strip, str(i + 1), (num_x - 4, num_y + 5),
                    FONT_BOLD, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Severity icon
        sev_txt = "[!!]" if f.severity == "critical" else "[!]"
        sev_col = (80, 80, 255) if f.severity == "critical" else (80, 160, 255)
        cv2.putText(strip, sev_txt, (cx + 32, y_top + 14),
                    FONT_BOLD, 0.44, sev_col, 1, cv2.LINE_AA)

        # Fault name
        cv2.putText(strip, f.name, (cx + 8, y_top + 34),
                    FONT_BOLD, 0.46, (240, 240, 240), 1, cv2.LINE_AA)

        # Description wrapped
        words = f.description.split()
        line, ty = "", y_top + 54
        max_chars = col_w // 7
        for word in words:
            test = line + ("" if not line else " ") + word
            if len(test) > max_chars:
                if line:
                    cv2.putText(strip, line, (cx + 8, ty),
                                FONT, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
                    ty += 16
                line = word
            else:
                line = test
        if line:
            cv2.putText(strip, line, (cx + 8, ty),
                        FONT, 0.35, (170, 170, 170), 1, cv2.LINE_AA)

    return strip


def make_summary_strip(w, session_summary: dict) -> np.ndarray:
    """Compact summary strip shown during normal playback."""
    strip = np.zeros((STRIP_H, w, 3), dtype=np.uint8)
    strip[:] = (10, 10, 16)
    cv2.line(strip, (0, 0), (w, 0), (40, 40, 60), 2)

    cv2.putText(strip, "SESSION FAULTS", (8, 22),
                FONT_BOLD, 0.50, (80, 80, 120), 1, cv2.LINE_AA)

    y = 42
    for name, pct in list(session_summary.items())[:4]:
        bar_w = int((pct / 100) * (w - 100))
        bar_c = (50, 50, 180) if pct > 60 else (40, 110, 50)
        cv2.rectangle(strip, (8, y - 10), (8 + bar_w, y + 2), bar_c, -1)
        short = name[:26]
        cv2.putText(strip, f"{short}  {pct:.0f}%",
                    (10, y), FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
        if y > STRIP_H - 10:
            break
    return strip


def draw_top_banner(img, text, sub, progress):
    """Thin top banner — fault name title."""
    w = img.shape[1]
    p  = min(1.0, progress)
    bh = 46
    by = int(-bh + p * bh)

    ov = img.copy()
    cv2.rectangle(ov, (0, by), (w, by + bh), (8, 8, 12), -1)
    cv2.rectangle(ov, (0, by + bh - 2), (w, by + bh), (0, 0, 180), 2)
    (tw, _), _ = cv2.getTextSize(text, FONT_BOLD, 0.78, 2)
    cv2.putText(ov, text, ((w - tw) // 2, by + 32),
                FONT_BOLD, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ov, sub, (8, by + bh - 6),
                FONT, 0.34, (100, 100, 140), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.93, img, 0.07, 0, img)


def draw_progress_bar(img, current, total, color=(50, 160, 80)):
    h, w = img.shape[:2]
    bw = int(current / max(1, total) * w)
    cv2.rectangle(img, (0, h - 3), (w, h),  (20, 20, 20), -1)
    cv2.rectangle(img, (0, h - 3), (bw, h), color, -1)


# ─── Pose helpers ─────────────────────────────────────────────────────────────

def build_landmarker(model_path):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def detect(landmarker, frame_bgr, ts_ms):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    r      = landmarker.detect_for_video(mp_img, ts_ms)
    return r.pose_landmarks[0] if r.pose_landmarks else None


# ─── Highlight selection ──────────────────────────────────────────────────────

def pick_highlights(frame_data: List[FrameData], min_gap=20) -> List[Highlight]:
    total = len(frame_data)
    by_fault: dict[str, list] = {}
    for i, fd in enumerate(frame_data):
        if fd.lms is None:
            continue
        for f in fd.analysis.faults:
            by_fault.setdefault(f.name, []).append((i, f, fd.lms, fd.frame))

    def priority(name):
        entries = by_fault[name]
        crit = any(f.severity == "critical" for _, f, _, _ in entries)
        return (0 if crit else 1, -len(entries))

    candidates = []
    for name in sorted(by_fault.keys(), key=priority):
        entries = by_fault[name]
        third = max(1, total // 3)
        for bucket in [
            [e for e in entries if e[0] < third],
            [e for e in entries if third <= e[0] < 2 * third],
            [e for e in entries if e[0] >= 2 * third],
        ]:
            if bucket:
                candidates.append(max(bucket, key=lambda e: e[1].confidence))

    candidates.sort(key=lambda e: e[0])
    highlights, last = [], -min_gap
    for idx, fault, lms, frame_img in candidates:
        gap = min_gap // 2 if fault.severity == "critical" else min_gap
        if idx - last >= gap:
            highlights.append(Highlight(idx, fault, lms, frame_img.copy()))
            last = idx

    highlights.sort(key=lambda h: h.frame_idx)
    return highlights


# ─── Renderer ─────────────────────────────────────────────────────────────────

def render(frame_data: List[FrameData],
           highlights: List[Highlight],
           output: str, fps: float, w: int, h: int,
           slowmo: int, freeze_n: int, hold_n: int):
    """
    Output resolution: w × (h + STRIP_H)
    Video area is always clean.
    """
    out_h  = h + STRIP_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, out_h))

    total    = len(frame_data)
    hl_map   = {hl.frame_idx: hl for hl in highlights}
    SLOWMO_R = 22   # frames around highlight shown in slowmo
    session  = {}
    pulse    = 0
    i        = 0
    written  = 0

    while i < total:
        fd   = frame_data[i]
        pulse = (pulse + 5) % 360

        # Rebuild session summary up to this point
        from boxing_analyzer.pattern_detector import PatternDetector as _PD
        # We already have analysis, just collect from history
        session = _build_session(frame_data, i)

        base = fd.frame.copy()

        if fd.lms is not None:
            fault_lms = {lm for f in fd.analysis.faults for lm in f.affected_landmarks}
            draw_skeleton(base, fd.lms, w, h, fault_lms)
            if fd.analysis.faults:
                draw_fault_markers(base, fd.lms, w, h, fd.analysis.faults, pulse)

        if i in hl_map:
            hl = hl_map[i]

            # ── Freeze + callout ──────────────────────────────────────────
            for t in range(freeze_n + hold_n):
                img = base.copy()
                prog = min(1.0, t / max(1, freeze_n * 0.35))

                sub = f"Frame {i}  |  {hl.fault.severity.upper()}  |  conf {hl.fault.confidence:.0%}"
                draw_top_banner(img, hl.fault.name.upper(), sub, min(1.0, t / 6))
                draw_progress_bar(img, i, total, color=(0, 0, 200))

                # Re-draw markers with animation
                if hl.lms is not None:
                    draw_fault_markers(img, hl.lms, w, h,
                                       hl.fault_list if hasattr(hl, 'fault_list')
                                       else [hl.fault], pulse + t * 3)

                strip = make_callout_strip(w, [hl.fault], "freeze", prog)
                combined = np.vstack([img, strip])
                writer.write(combined)
                written += 1

            # ── Slowmo replay ─────────────────────────────────────────────
            slo_s = max(0, i - SLOWMO_R)
            slo_e = min(total, i + SLOWMO_R + 1)

            for j in range(slo_s, slo_e):
                sfd  = frame_data[j]
                simg = sfd.frame.copy()

                if sfd.lms is not None:
                    sfault_lms = {lm for f in sfd.analysis.faults
                                  for lm in f.affected_landmarks}
                    draw_skeleton(simg, sfd.lms, w, h, sfault_lms)
                    if sfd.analysis.faults:
                        draw_fault_markers(simg, sfd.lms, w, h,
                                           sfd.analysis.faults, pulse)

                draw_progress_bar(simg, j, total, color=(0, 180, 220))

                faults_here = sfd.analysis.faults if sfd.lms else []
                strip = make_callout_strip(w, faults_here[:3], "slowmo", 1.0,
                                           slowmo_factor=slowmo)

                combined = np.vstack([simg, strip])
                for _ in range(slowmo):
                    writer.write(combined)
                    written += 1

            i = slo_e
            continue

        else:
            # Normal playback — summary strip
            draw_progress_bar(base, i, total)
            strip = make_summary_strip(w, session)
            combined = np.vstack([base, strip])
            writer.write(combined)
            written += 1

        i += 1

    writer.release()
    print(f"[Render] {written} frames → {output}")


def _build_session(frame_data, up_to):
    counts: dict[str, int] = {}
    n = min(up_to + 1, len(frame_data))
    for fd in frame_data[:n]:
        seen = set()
        for f in fd.analysis.faults:
            if f.name not in seen:
                counts[f.name] = counts.get(f.name, 0) + 1
                seen.add(f.name)
    if n == 0:
        return {}
    return {k: round(v / n * 100, 1)
            for k, v in sorted(counts.items(), key=lambda x: -x[1])}


# ─── Main ─────────────────────────────────────────────────────────────────────

def is_url(s):
    return s.startswith("http") and ("youtube.com" in s or "youtu.be" in s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--output",   default="highlight_reel.mp4")
    p.add_argument("--slowmo",   type=int,   default=4)
    p.add_argument("--freeze",   type=int,   default=50)
    p.add_argument("--start",    type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--scale",    type=float, default=1.0)
    p.add_argument("--min-gap",  type=int,   default=15)
    args = p.parse_args()

    src = args.source
    if is_url(src):
        src = download_youtube(src)
    src = ensure_h264(src)

    cap   = cv2.VideoCapture(src)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W     = int(ow * args.scale)
    H     = int(oh * args.scale)

    start_f = int(args.start * fps)
    end_f   = min(total, start_f + int(args.duration * fps) if args.duration else total)
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    print(f"[Video] {os.path.basename(src)}  {W}x{H} @ {fps:.0f}fps")

    model = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    lmrk  = build_landmarker(model)
    pat   = PatternDetector()

    all_frames: List[FrameData] = []
    fi = start_f
    while cap.isOpened() and fi < end_f:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if args.scale != 1.0:
            frame = cv2.resize(frame, (W, H))
        ts_ms    = int(fi * 1000 / fps)
        lms      = detect(lmrk, frame, ts_ms)
        analysis = pat.analyze(lms, W, H) if lms else FrameAnalysis()
        all_frames.append(FrameData(frame, lms, analysis))
        if len(all_frames) % 30 == 0:
            pct = (fi - start_f) / max(1, end_f - start_f) * 100
            print(f"  {fi}/{end_f} ({pct:.0f}%)")

    cap.release()
    lmrk.close()

    summary = pat.get_session_summary()
    print("\n[Top faults]")
    for name, pct in list(summary.items())[:5]:
        print(f"  {name:<35} {pct:.1f}%")

    highlights = pick_highlights(all_frames, args.min_gap)
    print(f"\n[Highlights] {len(highlights)} moments:")
    for hl in highlights:
        print(f"  frame {hl.frame_idx:3d}  [{hl.fault.severity.upper()}] {hl.fault.name}")

    print(f"\n[Rendering] → {args.output}")
    render(all_frames, highlights, args.output,
           fps, W, H,
           slowmo=args.slowmo,
           freeze_n=args.freeze,
           hold_n=60)

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
