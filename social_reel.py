#!/usr/bin/env python3
"""
Social Media Reel Generator — Threads / Instagram / TikTok
============================================================
Output: 720x1280 (9:16 vertical), ~60-90 sec, H.264
Structure:
  [2s]  Intro card
  [N×]  Top fault moments: freeze 1.5s → callout → slowmo 2s
  [3s]  Summary card — top 5 faults ranked
"""

import os, argparse, textwrap
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import List

from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point
from boxing_analyzer.video_io import ensure_h264, download_youtube

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

OUT_W, OUT_H = 720, 1280
VIDEO_H      = 860    # video crop area height
STRIP_H      = OUT_H - VIDEO_H   # 420px callout strip

CONNECTIONS = [
    (LM.LEFT_EAR, LM.NOSE), (LM.RIGHT_EAR, LM.NOSE),
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),   (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),     (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE),         (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),       (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]

@dataclass
class FrameData:
    frame: np.ndarray   # already cropped to OUT_W × VIDEO_H
    lms: object
    analysis: FrameAnalysis
    orig_scale: float   # scale factor applied when cropping

@dataclass
class Highlight:
    frame_idx: int
    fault: Fault
    lms: object


# ── Frame prep ────────────────────────────────────────────────────────────────

def crop_to_vertical(frame: np.ndarray) -> tuple:
    """
    Crop horizontal (or any) frame to OUT_W × VIDEO_H (9:16 ratio for video area).
    Returns (cropped_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    # Scale so height = VIDEO_H
    scale = VIDEO_H / h
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, VIDEO_H))
    # Center crop to OUT_W
    if new_w >= OUT_W:
        x0 = (new_w - OUT_W) // 2
        cropped = resized[:, x0:x0 + OUT_W]
    else:
        # Pad sides
        pad = (OUT_W - new_w) // 2
        cropped = cv2.copyMakeBorder(resized, 0, 0, pad, OUT_W - new_w - pad,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cropped, scale


def remap_lms(lms, orig_w, orig_h):
    """
    Wrap landmarks so get_point() returns coords in OUT_W × VIDEO_H space.
    Applies same crop_to_vertical transform.
    """
    scale = VIDEO_H / orig_h
    new_w = int(orig_w * scale)
    x_off = (new_w - OUT_W) // 2 if new_w >= OUT_W else -(OUT_W - new_w) // 2

    class RemappedLM:
        def __init__(self, x, y, z, vis):
            self.x = x; self.y = y; self.z = z; self.visibility = vis

    class RemappedList:
        def __init__(self, orig, scale, x_off):
            self._orig = orig
            self._s = scale
            self._xo = x_off
        def __getitem__(self, i):
            lm = self._orig[i]
            px = lm.x * orig_w * scale - x_off
            py = lm.y * orig_h * scale
            return RemappedLM(px / OUT_W, py / VIDEO_H, lm.z, lm.visibility)

    return RemappedList(lms, scale, x_off)


# ── Skeleton ──────────────────────────────────────────────────────────────────

def draw_skeleton(img, lms, fault_set):
    w, h = OUT_W, VIDEO_H
    ov = img.copy()
    for a, b in CONNECTIONS:
        try:
            pa = get_point(lms, a, w, h).astype(int)
            pb = get_point(lms, b, w, h).astype(int)
            fault = (a in fault_set) or (b in fault_set)
            col  = (0, 0, 210)   if fault else (40, 200, 40)
            glow = (80, 80, 255) if fault else (100, 255, 100)
            cv2.line(ov, tuple(pa), tuple(pb), col,  6, cv2.LINE_AA)
            cv2.line(ov, tuple(pa), tuple(pb), glow, 2, cv2.LINE_AA)
        except Exception:
            pass
    for idx in range(33):
        try:
            pt    = get_point(lms, idx, w, h).astype(int)
            fault = idx in fault_set
            outer = (0, 0, 170)    if fault else (20, 140, 20)
            inner = (120, 120, 255) if fault else (150, 255, 150)
            r = 11 if fault else 6
            cv2.circle(ov, tuple(pt), r + 3, outer, -1, cv2.LINE_AA)
            cv2.circle(ov, tuple(pt), r,     inner, -1, cv2.LINE_AA)
        except Exception:
            pass
    cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)


def draw_markers(img, lms, faults, pulse):
    w, h = OUT_W, VIDEO_H
    pr = int(20 + 7 * abs(np.sin(np.radians(pulse))))
    for n, f in enumerate(faults[:3], 1):
        for lm_idx in f.affected_landmarks[:1]:
            try:
                pt = get_point(lms, lm_idx, w, h).astype(int)
                cv2.circle(img, tuple(pt), pr,     (0, 0, 230), 3, cv2.LINE_AA)
                cv2.circle(img, tuple(pt), pr + 5, (0, 0, 130), 1, cv2.LINE_AA)
                cv2.circle(img, tuple(pt), 16, (0, 0, 200), -1, cv2.LINE_AA)
                cv2.putText(img, str(n), (pt[0] - 5, pt[1] + 6),
                            FONT_BOLD, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                pass


# ── Callout strip ─────────────────────────────────────────────────────────────

def make_strip(faults: list, mode: str, progress: float,
               session: dict, slowmo: int = 1) -> np.ndarray:
    strip = np.zeros((STRIP_H, OUT_W, 3), dtype=np.uint8)
    strip[:] = (10, 10, 18)
    cv2.line(strip, (0, 0), (OUT_W, 0), (50, 50, 80), 3)

    p = min(1.0, progress)
    slide = int((1 - p) * 40)

    # Mode badge
    if mode == "freeze":
        badge, bcol, btc = "PAUSED", (35, 35, 130), (130, 130, 255)
    elif mode == "slowmo":
        badge, bcol, btc = f"SLOW x1/{slowmo}", (10, 55, 75), (0, 200, 255)
    else:
        badge, bcol, btc = "", None, None

    if bcol:
        (bw, _), _ = cv2.getTextSize(badge, FONT_BOLD, 0.55, 1)
        bx = OUT_W - bw - 24
        cv2.rectangle(strip, (bx - 8, 8), (OUT_W - 8, 30), bcol, -1)
        cv2.rectangle(strip, (bx - 8, 8), (OUT_W - 8, 30), btc, 1)
        cv2.putText(strip, badge, (bx, 26), FONT_BOLD, 0.55, btc, 1, cv2.LINE_AA)

    if not faults:
        # Show session summary in normal mode
        y = 40
        cv2.putText(strip, "FAULT TRACKING", (14, y),
                    FONT_BOLD, 0.55, (60, 60, 100), 1, cv2.LINE_AA)
        y += 28
        for name, pct in list(session.items())[:5]:
            bw = int(pct / 100 * (OUT_W - 120))
            bc = (50, 50, 180) if pct > 60 else (40, 100, 50)
            cv2.rectangle(strip, (14, y - 12), (14 + bw, y + 2), bc, -1)
            cv2.putText(strip, f"{name[:28]}  {pct:.0f}%", (16, y),
                        FONT, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            y += 22
        return strip

    # Fault callouts — vertical stack (better for portrait)
    y = 18 + slide
    for i, f in enumerate(faults[:3]):
        if y > STRIP_H - 60:
            break
        # Number + severity
        num_col = (0, 0, 180) if f.severity == "critical" else (0, 90, 180)
        cv2.circle(strip, (22, y + 8), 14, num_col, -1, cv2.LINE_AA)
        cv2.putText(strip, str(i + 1), (17, y + 14),
                    FONT_BOLD, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        sev = "[!!]" if f.severity == "critical" else "[!]"
        sev_c = (100, 100, 255) if f.severity == "critical" else (80, 150, 255)
        cv2.putText(strip, sev, (42, y + 8), FONT_BOLD, 0.48, sev_c, 1, cv2.LINE_AA)

        # Fault name — big and bold
        cv2.putText(strip, f.name, (42, y + 28),
                    FONT_BOLD, 0.60, (240, 240, 240), 1, cv2.LINE_AA)

        # Description wrapped
        words = f.description.split()
        line, ty = "", y + 50
        for word in words:
            test = (line + " " + word).strip()
            if len(test) > 44:
                cv2.putText(strip, line, (42, ty), FONT, 0.40,
                            (160, 160, 160), 1, cv2.LINE_AA)
                ty += 18
                line = word
            else:
                line = test
        if line:
            cv2.putText(strip, line, (42, ty), FONT, 0.40,
                        (160, 160, 160), 1, cv2.LINE_AA)
            y = ty + 30
        else:
            y += 90

        if i < len(faults) - 1 and y < STRIP_H - 10:
            cv2.line(strip, (10, y - 8), (OUT_W - 10, y - 8), (35, 35, 50), 1)

    return strip


# ── Cards ─────────────────────────────────────────────────────────────────────

def make_intro_card(title: str, subtitle: str, n_frames: int,
                    fps: float) -> List[np.ndarray]:
    frames = []
    for t in range(n_frames):
        img = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        # Gradient background
        for y in range(OUT_H):
            ratio = y / OUT_H
            b = int(8 + ratio * 20)
            img[y, :] = (b, b, b + 15)

        p = min(1.0, t / max(1, n_frames * 0.3))

        # Boxing gloves icon (simple circles)
        cx = OUT_W // 2
        cy = OUT_H // 2 - 180
        r  = int(60 * p)
        cv2.circle(img, (cx - 45, cy), r, (0, 0, 180), -1, cv2.LINE_AA)
        cv2.circle(img, (cx + 45, cy), r, (0, 0, 180), -1, cv2.LINE_AA)
        cv2.circle(img, (cx - 45, cy), r, (60, 60, 255), 3, cv2.LINE_AA)
        cv2.circle(img, (cx + 45, cy), r, (60, 60, 255), 3, cv2.LINE_AA)

        # Title
        alpha = p
        slide = int((1 - p) * 60)
        ov = img.copy()
        (tw, th), _ = cv2.getTextSize(title, FONT_BOLD, 1.1, 2)
        cv2.putText(ov, title, ((OUT_W - tw) // 2, OUT_H // 2 - 40 + slide),
                    FONT_BOLD, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        # Subtitle
        for i, line in enumerate(subtitle.split("\n")):
            (lw, lh), _ = cv2.getTextSize(line, FONT, 0.58, 1)
            cv2.putText(ov, line, ((OUT_W - lw) // 2,
                        OUT_H // 2 + 30 + i * 32 + slide),
                        FONT, 0.58, (180, 180, 255), 1, cv2.LINE_AA)

        # Red accent line
        lw = int(p * 200)
        cv2.rectangle(ov, ((OUT_W - lw) // 2, OUT_H // 2 - 60),
                      ((OUT_W + lw) // 2, OUT_H // 2 - 55), (0, 0, 220), -1)

        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

        # Progress line at bottom
        bw = int(t / n_frames * OUT_W)
        cv2.rectangle(img, (0, OUT_H - 4), (bw, OUT_H), (0, 0, 180), -1)
        frames.append(img)
    return frames


def make_summary_card(session: dict, n_frames: int) -> List[np.ndarray]:
    frames = []
    items = list(session.items())[:5]
    for t in range(n_frames):
        img = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        for y in range(OUT_H):
            ratio = y / OUT_H
            img[y, :] = (int(8 + ratio * 10), int(5 + ratio * 5),
                         int(20 + ratio * 30))

        p = min(1.0, t / max(1, n_frames * 0.3))

        cv2.putText(img, "EXPLOITABLE", (40, 120), FONT_BOLD, 0.95,
                    (200, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "PATTERNS", (40, 170), FONT_BOLD, 1.3,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (40, 190), (40 + int(p * 320), 196),
                      (0, 0, 220), -1)

        y = 260
        for i, (name, pct) in enumerate(items):
            slide = int((1 - min(1.0, t / max(1, n_frames * 0.5))) * 80)
            bar_max = OUT_W - 80
            bar_w   = int(pct / 100 * bar_max * min(1.0, t / max(1, n_frames * 0.6)))
            bc = (50, 50, 200) if pct > 60 else (40, 110, 60)

            cv2.rectangle(img, (40, y - 2 + slide),
                          (40 + bar_w, y + 20 + slide), bc, -1)
            cv2.putText(img, f"{i+1}. {name}", (44, y + 16 + slide),
                        FONT_BOLD, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(img, f"{pct:.0f}%", (OUT_W - 70, y + 16 + slide),
                        FONT_BOLD, 0.55, (180, 180, 255), 1, cv2.LINE_AA)
            y += 72

        cv2.putText(img, "AI Boxing Analysis", (40, OUT_H - 60),
                    FONT, 0.45, (70, 70, 100), 1, cv2.LINE_AA)
        cv2.putText(img, "MediaPipe + OpenCV", (40, OUT_H - 35),
                    FONT, 0.40, (50, 50, 80), 1, cv2.LINE_AA)
        frames.append(img)
    return frames


# ── Top banner on video area ──────────────────────────────────────────────────

def draw_banner(img, text, progress):
    p  = min(1.0, progress)
    bh = 52
    by = int(-bh + p * bh)
    ov = img.copy()
    cv2.rectangle(ov, (0, by), (OUT_W, by + bh), (6, 6, 12), -1)
    cv2.rectangle(ov, (0, by + bh - 3), (OUT_W, by + bh), (0, 0, 200), -1)
    (tw, _), _ = cv2.getTextSize(text, FONT_BOLD, 0.85, 2)
    cv2.putText(ov, text, ((OUT_W - tw) // 2, by + 36),
                FONT_BOLD, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.92, img, 0.08, 0, img)


# ── Highlight selection ───────────────────────────────────────────────────────

def pick_top_highlights(frame_data, n=6, min_gap=20):
    total = len(frame_data)
    by_fault = {}
    for i, fd in enumerate(frame_data):
        if fd.lms is None:
            continue
        for f in fd.analysis.faults:
            by_fault.setdefault(f.name, []).append((i, f, fd.lms))

    def priority(name):
        entries = by_fault[name]
        crit = any(f.severity == "critical" for _, f, _ in entries)
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
    for idx, fault, lms in candidates:
        gap = min_gap // 2 if fault.severity == "critical" else min_gap
        if idx - last >= gap:
            highlights.append(Highlight(idx, fault, lms))
            last = idx
        if len(highlights) >= n:
            break

    return sorted(highlights, key=lambda h: h.frame_idx)


# ── Pose ──────────────────────────────────────────────────────────────────────

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
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    r   = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
    return r.pose_landmarks[0] if r.pose_landmarks else None


# ── Render ────────────────────────────────────────────────────────────────────

def render(frame_data, highlights, output, fps, orig_w, orig_h,
           slowmo=3, freeze_n=45, hold_n=45):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (OUT_W, OUT_H))

    total  = len(frame_data)
    hl_map = {h.frame_idx: h for h in highlights}
    SCTX   = 18   # slowmo context frames
    pulse  = 0
    i      = 0
    written = 0

    def compose(video_frame, strip):
        return np.vstack([video_frame, strip])

    while i < total:
        fd    = frame_data[i]
        pulse = (pulse + 6) % 360

        base = fd.frame.copy()
        if fd.lms is not None:
            fault_lms = {lm for f in fd.analysis.faults for lm in f.affected_landmarks}
            draw_skeleton(base, fd.lms, fault_lms)
            if fd.analysis.faults:
                draw_markers(base, fd.lms, fd.analysis.faults, pulse)

        if i in hl_map:
            hl = hl_map[i]

            # Freeze
            for t in range(freeze_n + hold_n):
                img = base.copy()
                prog = min(1.0, t / max(1, freeze_n * 0.4))
                draw_banner(img, hl.fault.name.upper(), min(1.0, t / 8))
                if hl.lms:
                    draw_markers(img, hl.lms, [hl.fault], pulse + t * 4)
                strip = make_strip([hl.fault], "freeze", prog, {})
                writer.write(compose(img, strip))
                written += 1

            # Slowmo
            for j in range(max(0, i - SCTX), min(total, i + SCTX + 1)):
                sfd  = frame_data[j]
                simg = sfd.frame.copy()
                if sfd.lms:
                    sfl = {lm for f in sfd.analysis.faults for lm in f.affected_landmarks}
                    draw_skeleton(simg, sfd.lms, sfl)
                    if sfd.analysis.faults:
                        draw_markers(simg, sfd.lms, sfd.analysis.faults, pulse)
                faults_here = sfd.analysis.faults[:2] if sfd.lms else []
                strip = make_strip(faults_here, "slowmo", 1.0, {}, slowmo)
                combined = compose(simg, strip)
                for _ in range(slowmo):
                    writer.write(combined)
                    written += 1

            i = min(total, i + SCTX + 1)
            continue

        # Normal
        session = _session_up_to(frame_data, i)
        strip   = make_strip(fd.analysis.faults[:2] if fd.lms else [],
                             "normal", 1.0, session)
        writer.write(compose(base, strip))
        written += 1
        i += 1

    writer.release()
    print(f"[Render] {written} frames → {output}")


def _session_up_to(frame_data, up_to):
    counts = {}
    for fd in frame_data[:up_to + 1]:
        seen = set()
        for f in fd.analysis.faults:
            if f.name not in seen:
                counts[f.name] = counts.get(f.name, 0) + 1
                seen.add(f.name)
    n = up_to + 1
    return {k: round(v / n * 100, 1)
            for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def is_url(s):
    return s.startswith("http") and ("youtube.com" in s or "youtu.be" in s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--output",    default="social_reel.mp4")
    p.add_argument("--title",     default="OPPONENT ANALYSIS")
    p.add_argument("--subtitle",  default="AI Boxing Breakdown\nFault Detection")
    p.add_argument("--slowmo",    type=int,   default=3)
    p.add_argument("--freeze",    type=int,   default=45)
    p.add_argument("--highlights",type=int,   default=6)
    p.add_argument("--start",     type=float, default=0.0)
    p.add_argument("--duration",  type=float, default=None)
    p.add_argument("--fps",       type=float, default=30.0)
    args = p.parse_args()

    src = args.source
    if is_url(src):
        src = download_youtube(src)
    src = ensure_h264(src)

    cap   = cv2.VideoCapture(src)
    fps   = cap.get(cv2.CAP_PROP_FPS) or args.fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = int(args.start * fps)
    end_f   = min(total, start_f + int(args.duration * fps) if args.duration else total)
    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    print(f"[Video] {os.path.basename(src)}  {ow}x{oh} @ {fps:.0f}fps")
    print(f"[Output] {OUT_W}x{OUT_H} vertical (video {OUT_W}x{VIDEO_H} + strip {STRIP_H}px)")

    model    = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    lmrk     = build_landmarker(model)
    pat      = PatternDetector()

    all_frames: List[FrameData] = []
    fi = start_f
    while cap.isOpened() and fi < end_f:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1

        cropped, scale = crop_to_vertical(frame)

        # Detect on original resolution for accuracy
        ts_ms    = int(fi * 1000 / fps)
        lms_orig = detect(lmrk, frame, ts_ms)

        if lms_orig:
            lms_remap = remap_lms(lms_orig, ow, oh)
            analysis  = pat.analyze(lms_remap, OUT_W, VIDEO_H)
        else:
            lms_remap = None
            analysis  = FrameAnalysis()

        all_frames.append(FrameData(cropped, lms_remap, analysis, scale))

        if len(all_frames) % 60 == 0:
            pct = (fi - start_f) / max(1, end_f - start_f) * 100
            print(f"  {fi}/{end_f} ({pct:.0f}%)  faults={len(analysis.faults)}")

    cap.release()
    lmrk.close()

    session = pat.get_session_summary()
    print("\n[Top faults]")
    for name, pct in list(session.items())[:6]:
        print(f"  {name:<35} {pct:.1f}%")

    highlights = pick_top_highlights(all_frames, n=args.highlights)
    print(f"\n[Highlights] {len(highlights)} moments selected")
    for h in highlights:
        print(f"  frame {h.frame_idx:4d}  [{h.fault.severity.upper()}] {h.fault.name}")

    # Build output: intro + reel + summary
    print(f"\n[Rendering] → {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (OUT_W, OUT_H))

    # Intro card
    intro_frames = make_intro_card(args.title, args.subtitle,
                                   int(fps * 2.5), fps)
    for f in intro_frames:
        writer.write(f)

    writer.release()

    # Main reel (appended via separate render call)
    tmp_reel = args.output.replace(".mp4", "_tmp_reel.mp4")
    render(all_frames, highlights, tmp_reel, fps, ow, oh,
           slowmo=args.slowmo, freeze_n=args.freeze, hold_n=args.freeze)

    # Summary card
    tmp_summary = args.output.replace(".mp4", "_tmp_summary.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_summary, fourcc, fps, (OUT_W, OUT_H))
    for f in make_summary_card(session, int(fps * 4)):
        writer.write(f)
    writer.release()

    # Concat all parts with ffmpeg
    import subprocess
    list_file = args.output.replace(".mp4", "_parts.txt")
    with open(list_file, "w") as f:
        intro_tmp = args.output.replace(".mp4", "_tmp_intro.mp4")
        # Write intro from writer above — re-render
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w2 = cv2.VideoWriter(intro_tmp, fourcc, fps, (OUT_W, OUT_H))
        for fr in make_intro_card(args.title, args.subtitle, int(fps * 2.5), fps):
            w2.write(fr)
        w2.release()
        f.write(f"file '{os.path.abspath(intro_tmp)}'\n")
        f.write(f"file '{os.path.abspath(tmp_reel)}'\n")
        f.write(f"file '{os.path.abspath(tmp_summary)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        args.output
    ], check=True, capture_output=True)

    # Cleanup temp files
    for f in [intro_tmp, tmp_reel, tmp_summary, list_file]:
        try:
            os.remove(f)
        except Exception:
            pass

    size_mb = os.path.getsize(args.output) / 1e6
    dur     = (len(intro_frames) + len(all_frames) + int(fps * 4)) / fps
    print(f"\nDone: {args.output}  ({size_mb:.1f}MB, ~{dur:.0f}s)")
    print(f"Format: {OUT_W}x{OUT_H} 9:16 — ready for Threads / Instagram / TikTok")


if __name__ == "__main__":
    main()
