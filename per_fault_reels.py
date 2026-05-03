#!/usr/bin/env python3
"""
Per-Fault Highlight Reels
=========================
Pass 1 — analyze video (skips intro via --start, filters out blue-shirt
referee, multi-pose detection, picks largest non-ref pose per frame).
Pass 2 — for each fault TYPE, render a SEPARATE short MP4:
  • Title card with fault name (1.5s)
  • Top N best moments, each:
      ~0.4s normal lead-in
      ~0.6s freeze frame with fault label
      ~2.5s slow-motion replay (1/4 speed) with persistent subtitle
  • Skeleton overlay on every frame
  • Persistent bottom subtitle naming the fault

Output: one MP4 per fault, sized for embedding side-by-side with the
fault item on the scouting site.

Usage:
  python per_fault_reels.py jose_guardado_h264.mp4 \\
      --outdir site/reels/ --start 60 --duration 390 --per-fault 4
"""

import os, sys, argparse, time, re
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from typing import List

from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point
from boxing_analyzer.video_io import ensure_h264

sys.stdout.reconfigure(line_buffering=True)

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

SUBTITLE_H = 110

# Map fault names → short tactical subtitle copy
FAULT_BLURBS = {
    "Chin Up - Head Exposed":
        "Chin elevated, jaw exposed — open target for hooks and uppercuts.",
    "Stance Too Narrow":
        "Feet too close together — no balance, no power, easy to push off line.",
    "Rear Elbow Flared":
        "Rear elbow drifts away from body — corridor to the liver/ribs.",
    "Lead Elbow Flared":
        "Lead elbow wings out — left side body open, counter-right path clear.",
    "Lead Hand Too Low":
        "Lead hand below guard line — straight right has clean path to face.",
    "Stance Too Wide":
        "Over-extended, planted feet — pivot off line and counter.",
    "Static Head - No Head Movement":
        "Head glued to center line — no slipping, no defense, just absorbs.",
    "Trunk Leaning Backward":
        "Weight on back foot — power gone, balance broken, vulnerable.",
    "Trunk Leaning Forward":
        "Over-committed forward — uppercut path open, off-balance for counters.",
    "Feet Crossing - Dangerous Footwork":
        "Feet cross — fundamental error, momentary loss of balance and base.",
}


@dataclass
class FrameMeta:
    idx:      int
    lms:      object
    analysis: FrameAnalysis


# ─── Ref filter & multi-pose detection ───────────────────────────────────────

_FILTER_STATS = {
    "ref_skipped": 0,
    "conceicao_skipped": 0,
    "no_pose_total": 0,
    "kept_guardado": 0,
}


def _sample_hsv(frame_bgr, lms, w, h, ratio_along_torso: float):
    """
    Sample HSV at a point between shoulders and ankles.
    ratio_along_torso = 0.0 (shoulders) → 1.0 (ankles).
    Returns (h, s, v) or None.
    """
    try:
        ls = get_point(lms, LM.LEFT_SHOULDER,  w, h)
        rs = get_point(lms, LM.RIGHT_SHOULDER, w, h)
        lh = get_point(lms, LM.LEFT_HIP,       w, h)
        rh = get_point(lms, LM.RIGHT_HIP,      w, h)
        lk = get_point(lms, LM.LEFT_KNEE,      w, h)
        rk = get_point(lms, LM.RIGHT_KNEE,     w, h)
    except Exception:
        return None
    sh_mid = (ls + rs) / 2
    hp_mid = (lh + rh) / 2
    kn_mid = (lk + rk) / 2
    if ratio_along_torso <= 0.6:
        # shoulders → hips
        t = ratio_along_torso / 0.6
        pt = sh_mid * (1 - t) + hp_mid * t
    else:
        # hips → knees (shorts area)
        t = (ratio_along_torso - 0.6) / 0.4
        pt = hp_mid * (1 - t) + kn_mid * t
    cx, cy = int(pt[0]), int(pt[1])
    cx = max(3, min(w - 4, cx)); cy = max(3, min(h - 4, cy))
    patch = frame_bgr[cy-3:cy+4, cx-3:cx+4]
    if patch.size == 0:
        return None
    avg = patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(np.uint8([[avg]]), cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def is_blue_shirt(frame_bgr, lms, w, h) -> bool:
    """Referee in blue/light-blue shirt — sample chest area."""
    hsv = _sample_hsv(frame_bgr, lms, w, h, 0.35)  # upper torso
    if hsv is None:
        return False
    h_, s_, v_ = hsv
    return (95 <= h_ <= 140) and s_ > 70 and v_ > 35


def green_score(frame_bgr, lms, w, h) -> float:
    """
    Score how 'green-shorted' a pose is. Higher = more likely Conceição.
    Samples 3 points along the thigh region for robustness.
    """
    score = 0.0
    samples = 0
    for r in (0.70, 0.78, 0.85):
        hsv = _sample_hsv(frame_bgr, lms, w, h, r)
        if hsv is None:
            continue
        h_, s_, v_ = hsv
        samples += 1
        # Green hue 40-85 (broad). Score scales with saturation + presence
        # in the green hue band. Even shadowy green still has hue in band.
        if 40 <= h_ <= 85:
            # weight: saturation matters more than brightness for hue ID
            score += (s_ / 255.0) * 100 * (1 + (v_ / 255.0))
    return score / max(1, samples)


def bbox_area(lms) -> float:
    xs = [lms[i].x for i in range(33)]
    ys = [lms[i].y for i in range(33)]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def build_landmarker(model_path):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=3,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def detect(landmarker, frame_bgr, ts_ms):
    """
    Detect up to N poses, identify GUARDADO specifically:
      1. Skip referee (blue shirt at chest)
      2. From remaining poses, score each by 'green-shortedness'
      3. If 2+ candidates: pick the LEAST-green one (Guardado, since
         Conceição has bright green trunks — works regardless of lighting)
      4. If only 1 candidate AND it's strongly green: skip (it's Conceição
         alone in frame); else accept (likely Guardado)
    """
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    r      = landmarker.detect_for_video(mp_img, ts_ms)
    if not r.pose_landmarks:
        _FILTER_STATS["no_pose_total"] += 1
        return None

    h, w = frame_bgr.shape[:2]
    candidates = []  # (green_score, bbox_area, pose)

    for pose in r.pose_landmarks:
        if is_blue_shirt(frame_bgr, pose, w, h):
            _FILTER_STATS["ref_skipped"] += 1
            continue
        gs = green_score(frame_bgr, pose, w, h)
        candidates.append((gs, bbox_area(pose), pose))

    if not candidates:
        _FILTER_STATS["no_pose_total"] += 1
        return None

    # Sort by green-score (lowest first = most likely Guardado)
    candidates.sort(key=lambda x: x[0])

    if len(candidates) >= 2:
        # Multiple non-ref poses — pick least-green = Guardado
        # Highest-green one is Conceição (skipped by selection)
        _FILTER_STATS["conceicao_skipped"] += 1
        chosen = candidates[0][2]
    else:
        # Only one non-ref pose. If it's strongly green, it's Conceição alone
        # (possible during knockdown / between-rounds). Skip in that case.
        if candidates[0][0] > 30.0:  # strong green signal
            _FILTER_STATS["conceicao_skipped"] += 1
            return None
        chosen = candidates[0][2]

    _FILTER_STATS["kept_guardado"] += 1
    return chosen


# ─── Drawing ─────────────────────────────────────────────────────────────────

def draw_skeleton(img, lms, w, h, fault_lm_set):
    ov = img.copy()
    for a, b in CONNECTIONS:
        try:
            pa = get_point(lms, a, w, h).astype(int)
            pb = get_point(lms, b, w, h).astype(int)
            fault = (a in fault_lm_set) or (b in fault_lm_set)
            col   = (0, 0, 220) if fault else (50, 200, 50)
            glow  = (60, 60, 255) if fault else (120, 255, 120)
            cv2.line(ov, tuple(pa), tuple(pb), col,  5, cv2.LINE_AA)
            cv2.line(ov, tuple(pa), tuple(pb), glow, 2, cv2.LINE_AA)
        except Exception:
            pass
    for idx in range(33):
        try:
            pt    = get_point(lms, idx, w, h).astype(int)
            fault = idx in fault_lm_set
            outer = (0, 0, 180) if fault else (20, 140, 20)
            inner = (100, 100, 255) if fault else (160, 255, 160)
            r = 9 if fault else 5
            cv2.circle(ov, tuple(pt), r + 3, outer, -1, cv2.LINE_AA)
            cv2.circle(ov, tuple(pt), r,     inner, -1, cv2.LINE_AA)
        except Exception:
            pass
    cv2.addWeighted(ov, 0.88, img, 0.12, 0, img)


def draw_pulse_marker(img, lms, w, h, lm_idx, pulse):
    try:
        pt = get_point(lms, lm_idx, w, h).astype(int)
    except Exception:
        return
    pr = int(18 + 8 * abs(np.sin(np.radians(pulse))))
    cv2.circle(img, tuple(pt), pr,     (0, 0, 230), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(pt), pr + 5, (0, 0, 120), 1, cv2.LINE_AA)


def make_subtitle_strip(w, fault_name, blurb, mode_badge=None):
    s = np.zeros((SUBTITLE_H, w, 3), dtype=np.uint8)
    s[:] = (8, 8, 14)
    cv2.line(s, (0, 0), (w, 0), (0, 0, 200), 3)

    # Fault name (big)
    (tw, _), _ = cv2.getTextSize(fault_name.upper(), FONT_BOLD, 0.85, 2)
    x_name = max(20, (w - tw) // 2)
    cv2.putText(s, fault_name.upper(), (x_name, 38),
                FONT_BOLD, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Subtitle blurb (smaller, centered, may wrap)
    max_chars = w // 10
    words = blurb.split()
    lines, line = [], ""
    for word in words:
        test = line + ("" if not line else " ") + word
        if len(test) > max_chars:
            if line: lines.append(line)
            line = word
        else:
            line = test
    if line: lines.append(line)
    y = 64
    for ln in lines[:2]:
        (lw, _), _ = cv2.getTextSize(ln, FONT, 0.55, 1)
        cv2.putText(s, ln, ((w - lw) // 2, y),
                    FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        y += 22

    # Mode badge in top-right
    if mode_badge:
        text, col, tc = mode_badge
        (bw, bh), _ = cv2.getTextSize(text, FONT_BOLD, 0.5, 1)
        bx = w - bw - 20
        cv2.rectangle(s, (bx - 8, 6), (w - 8, 28), col, -1)
        cv2.putText(s, text, (bx, 23), FONT_BOLD, 0.5, tc, 1, cv2.LINE_AA)
    return s


def make_title_card(w, h, fault_name, blurb, count_text):
    card = np.zeros((h + SUBTITLE_H, w, 3), dtype=np.uint8)
    card[:] = (20, 18, 28)
    # Top accent
    cv2.rectangle(card, (0, 0), (w, 6), (0, 0, 200), -1)
    # Big fault name
    (tw, _), _ = cv2.getTextSize(fault_name.upper(), FONT_BOLD, 1.6, 3)
    cv2.putText(card, fault_name.upper(), ((w - tw) // 2, h // 2 - 30),
                FONT_BOLD, 1.6, (240, 240, 250), 3, cv2.LINE_AA)
    # Subtitle
    (sw, _), _ = cv2.getTextSize(blurb, FONT, 0.7, 1)
    cv2.putText(card, blurb, ((w - sw) // 2, h // 2 + 10),
                FONT, 0.7, (180, 180, 200), 1, cv2.LINE_AA)
    # Count
    (cw, _), _ = cv2.getTextSize(count_text, FONT_BOLD, 0.6, 1)
    cv2.putText(card, count_text, ((w - cw) // 2, h // 2 + 50),
                FONT_BOLD, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
    return card


# ─── Per-fault selection ─────────────────────────────────────────────────────

def select_top_per_fault(metas: List[FrameMeta], fault_name: str,
                         top_n: int, min_gap_frames: int) -> List[int]:
    """Return up to top_n frame indices for this fault, well-spaced."""
    candidates = []
    for fm in metas:
        if fm.lms is None: continue
        for f in fm.analysis.faults:
            if f.name == fault_name:
                candidates.append((fm.idx, f.confidence))
    candidates.sort(key=lambda x: -x[1])  # highest confidence first
    picked, last_used = [], []
    for idx, _ in candidates:
        if all(abs(idx - u) >= min_gap_frames for u in last_used):
            picked.append(idx)
            last_used.append(idx)
            if len(picked) >= top_n:
                break
    picked.sort()
    return picked


# ─── Per-fault reel renderer ─────────────────────────────────────────────────

def render_fault_reel(src_path, metas, fault_name, frame_indices,
                      output, fps, w, h, scale, slowmo=4):
    """
    Each fault gets its own MP4:
      title card (1.5s) → for each moment: lead-in → freeze → slow-mo
    """
    if not frame_indices:
        print(f"  [skip] {fault_name}: no moments")
        return False

    blurb = FAULT_BLURBS.get(fault_name, "Detected by AI pose analysis.")
    out_h = h + SUBTITLE_H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, out_h))

    # Lookup helper
    meta_by_idx = {fm.idx: fm for fm in metas}

    # Title card (1.5s = 38 frames at 25fps)
    title = make_title_card(w, h, fault_name, blurb,
                            f"{len(frame_indices)} key moments")
    for _ in range(int(fps * 1.5)):
        writer.write(title)

    cap = cv2.VideoCapture(src_path)

    LEAD_IN = int(fps * 0.4)   # 0.4s normal lead-in (= ~10 frames at 25)
    FREEZE  = int(fps * 0.7)   # 0.7s freeze with big label
    SLOW_R  = int(fps * 0.35)  # ±0.35s window for slow-mo (= ±9 source frames)
    pulse = 0

    for moment_i, target_idx in enumerate(frame_indices):
        fm = meta_by_idx.get(target_idx)
        if fm is None: continue
        fault_lms = {lm for f in fm.analysis.faults
                     if f.name == fault_name
                     for lm in f.affected_landmarks}

        # ── LEAD-IN (normal speed, no freeze) ─────────────────────────────
        lead_start = max(0, target_idx - LEAD_IN)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lead_start)
        for j in range(lead_start, target_idx):
            ret, frame = cap.read()
            if not ret: break
            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))
            sfm = meta_by_idx.get(j)
            if sfm and sfm.lms is not None:
                sf = {lm for f in sfm.analysis.faults
                      if f.name == fault_name
                      for lm in f.affected_landmarks}
                draw_skeleton(frame, sfm.lms, w, h, sf)
            strip = make_subtitle_strip(w, fault_name, blurb,
                                        ("LEAD-IN", (60, 30, 30), (180, 180, 220)))
            writer.write(np.vstack([frame, strip]))

        # ── FREEZE on the fault frame ─────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, freeze_frame = cap.read()
        if not ret: continue
        if scale != 1.0:
            freeze_frame = cv2.resize(freeze_frame, (w, h))

        for t in range(FREEZE):
            pulse = (pulse + 8) % 360
            f = freeze_frame.copy()
            if fm.lms is not None:
                draw_skeleton(f, fm.lms, w, h, fault_lms)
                for lm_idx in list(fault_lms)[:2]:
                    draw_pulse_marker(f, fm.lms, w, h, lm_idx, pulse)
            # Big "FAULT" overlay on top
            ov = f.copy()
            cv2.rectangle(ov, (0, 0), (w, 56), (0, 0, 0), -1)
            cv2.rectangle(ov, (0, 54), (w, 58), (0, 0, 200), -1)
            cv2.putText(ov, "← FAULT DETECTED →", (20, 38),
                        FONT_BOLD, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.85, f, 0.15, 0, f)
            strip = make_subtitle_strip(w, fault_name, blurb,
                                        ("FREEZE", (40, 40, 160), (255, 255, 255)))
            writer.write(np.vstack([f, strip]))

        # ── SLOW-MO REPLAY (each source frame written `slowmo` times) ─────
        slo_start = max(0, target_idx - SLOW_R)
        slo_end   = target_idx + SLOW_R + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, slo_start)
        for j in range(slo_start, slo_end):
            ret, sframe = cap.read()
            if not ret: break
            if scale != 1.0:
                sframe = cv2.resize(sframe, (w, h))
            sfm = meta_by_idx.get(j)
            sf = set()
            if sfm and sfm.lms is not None:
                sf = {lm for fl in sfm.analysis.faults
                      if fl.name == fault_name
                      for lm in fl.affected_landmarks}
                draw_skeleton(sframe, sfm.lms, w, h, sf)
            strip = make_subtitle_strip(
                w, fault_name, blurb,
                (f"SLOW 1/{slowmo}", (10, 70, 90), (0, 220, 255)))
            combined = np.vstack([sframe, strip])
            for _ in range(slowmo):
                writer.write(combined)

    cap.release()
    writer.release()
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--outdir",    default="site/reels")
    p.add_argument("--start",     type=float, default=0.0)
    p.add_argument("--duration",  type=float, default=None)
    p.add_argument("--scale",     type=float, default=1.0)
    p.add_argument("--per-fault", type=int,   default=4,
                   help="moments per fault reel")
    p.add_argument("--slowmo",    type=int,   default=4)
    p.add_argument("--min-gap",   type=int,   default=80,
                   help="min frames between two moments in the same fault")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    src = ensure_h264(args.source)

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

    print(f"[Video] {os.path.basename(src)}  {W}x{H} @ {fps:.0f}fps", flush=True)
    print(f"[Window] frames {start_f}..{end_f} (skip first {args.start:.0f}s)",
          flush=True)

    model = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    lmrk  = build_landmarker(model)
    pat   = PatternDetector()

    # ── PASS 1 ────────────────────────────────────────────────────────────
    print(f"[Pass 1] analyzing with ref filter (multi-pose, blue-shirt skip)",
          flush=True)
    metas: List[FrameMeta] = []
    fi = start_f
    t0 = time.time()
    last_status = t0
    while cap.isOpened() and fi < end_f:
        ret, frame = cap.read()
        if not ret: break
        if args.scale != 1.0:
            frame = cv2.resize(frame, (W, H))
        ts_ms    = int(fi * 1000 / fps)
        lms      = detect(lmrk, frame, ts_ms)
        analysis = pat.analyze(lms, W, H) if lms else FrameAnalysis()
        metas.append(FrameMeta(fi, lms, analysis))
        fi += 1
        if time.time() - last_status > 5:
            done = fi - start_f
            pct  = done / max(1, end_f - start_f) * 100
            rate = done / max(0.1, time.time() - t0)
            eta  = (end_f - fi) / max(0.1, rate)
            print(f"  frame {fi}/{end_f} ({pct:.0f}%) | {rate:.1f} fps | ETA {eta:.0f}s "
                  f"| ref={_FILTER_STATS['ref_skipped']} concei={_FILTER_STATS['conceicao_skipped']} guard={_FILTER_STATS['kept_guardado']}",
                  flush=True)
            last_status = time.time()
    cap.release()
    lmrk.close()

    print(f"[Pass 1] done in {time.time()-t0:.0f}s — {len(metas)} frames analyzed",
          flush=True)
    print(f"[Filter] ref_skipped={_FILTER_STATS['ref_skipped']}  "
          f"conceicao_skipped={_FILTER_STATS['conceicao_skipped']}  "
          f"guardado_kept={_FILTER_STATS['kept_guardado']}  "
          f"no_pose={_FILTER_STATS['no_pose_total']}",
          flush=True)

    summary = pat.get_session_summary()
    print("\n[Session faults]", flush=True)
    for name, pct in list(summary.items())[:8]:
        print(f"  {name:<40} {pct:5.1f}%", flush=True)

    # Save the cleaned summary as JSON for the site
    import json
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "frames_analyzed": len(metas),
            "ref_filter_stats": _FILTER_STATS,
            "fault_percentages": summary,
        }, f, indent=2)

    # ── PASS 2: per-fault reels ───────────────────────────────────────────
    print(f"\n[Pass 2] rendering per-fault reels → {args.outdir}/", flush=True)
    rendered = {}
    for fault_name in summary.keys():
        slug = slugify(fault_name)
        out  = os.path.join(args.outdir, f"{slug}.mp4")
        idxs = select_top_per_fault(metas, fault_name,
                                    args.per_fault, args.min_gap)
        print(f"  → {fault_name}  ({len(idxs)} moments)", flush=True)
        ok = render_fault_reel(src, metas, fault_name, idxs,
                               out, fps, W, H, args.scale, args.slowmo)
        if ok:
            rendered[fault_name] = out

    print(f"\n[Done] {len(rendered)} per-fault reels in {args.outdir}/",
          flush=True)
    for name, path in rendered.items():
        size = os.path.getsize(path) / (1024*1024)
        print(f"  {name:<40} {size:6.1f} MB  {path}", flush=True)


if __name__ == "__main__":
    main()
