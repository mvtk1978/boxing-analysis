#!/usr/bin/env python3
"""
SAM 2 + Pose Pipeline for Guardado-Only Boxing Analysis
========================================================
Three stages:
  Stage 1 — Identity: Grounding DINO finds "person wearing black shorts"
            in seed frame → bounding box → SAM 2 video propagation
            outputs Guardado pixel mask for every frame.

  Stage 2 — Pose: For each frame, mask out everything except Guardado,
            run pose estimation on the masked frame. No more
            wrong-boxer detections, no more broken legs from
            background clutter.

  Stage 3 — Faults: Same fault detection as before, but now operating
            on guaranteed-Guardado data. Output per-fault reels.

Usage:
  python sam2_pose_pipeline.py video.mp4 \\
      --start 98 --duration 419 \\
      --seed-frame 0 \\
      --outdir site/reels
"""

import os, sys, argparse, time, re, json
import numpy as np
import cv2
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# Force the script to be run from a non-conflicting directory; we already
# cd into a safe dir before invoking.
from sam2.build_sam import build_sam2_video_predictor
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection
)

# Pose: use MediaPipe for now (we already have it; ViTPose can drop in later)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, '/home/max/boxingnew1')
from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point


# ─── Stage 1: Identity (Grounding DINO + SAM 2) ──────────────────────────────

def find_guardado_box(seed_frame_bgr, prompt: str = "boxer wearing black shorts."):
    """
    Use Grounding DINO to find the Guardado bounding box.
    Returns (x1, y1, x2, y2) or raises if not found.
    """
    device = "cuda"
    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"  [Grounding DINO] loading {model_id}...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    rgb = cv2.cvtColor(seed_frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Newer transformers renamed kwargs
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.30,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.30,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    print(f"  [Grounding DINO] found {len(boxes)} candidates with prompt '{prompt}'",
          flush=True)
    for box, score, lbl in zip(boxes, scores, labels):
        print(f"    box={box.astype(int).tolist()} score={score:.2f} label={lbl}", flush=True)

    if len(boxes) == 0:
        raise RuntimeError("No Guardado found in seed frame — try a different prompt or seed frame")

    # Pick highest-score box
    best = int(np.argmax(scores))
    return tuple(boxes[best].astype(int))


def extract_frames(video_path, start_f, end_f, scale_w, scale_h, out_dir):
    """Extract frames as JPEGs (SAM 2 video predictor needs jpeg dir)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fi = start_f
    written = 0
    t0 = time.time()
    while fi < end_f:
        ret, frame = cap.read()
        if not ret: break
        if scale_w and (frame.shape[1] != scale_w or frame.shape[0] != scale_h):
            frame = cv2.resize(frame, (scale_w, scale_h))
        # SAM 2 expects zero-padded sequential filenames
        out_name = os.path.join(out_dir, f"{written:05d}.jpg")
        cv2.imwrite(out_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1
        fi += 1
        if written % 500 == 0:
            print(f"  [extract] {written} frames written ({(time.time()-t0):.0f}s)", flush=True)
    cap.release()
    print(f"  [extract] DONE: {written} frames -> {out_dir}", flush=True)
    return written


def propagate_with_sam2(predictor, frames_dir, seed_box, ann_frame_idx=0):
    """
    Use SAM 2 video predictor: seed with bounding box on annotated frame,
    propagate through all frames. Returns dict frame_idx -> mask (HxW bool).
    """
    print(f"  [SAM 2] init video state on {frames_dir}", flush=True)
    inference_state = predictor.init_state(
        video_path=frames_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )

    # Add the bounding box prompt at the seed frame
    x1, y1, x2, y2 = seed_box
    box_xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    print(f"  [SAM 2] seeding with box {seed_box} at frame {ann_frame_idx}", flush=True)

    _, _, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=1,            # Guardado = object id 1
        box=box_xyxy,
    )

    # Propagate forward through video
    print(f"  [SAM 2] propagating forward...", flush=True)
    masks = {}
    t0 = time.time()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # mask_logits shape: (num_obj, H, W) — convert to bool
        m = (out_mask_logits[0] > 0.0).cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        masks[out_frame_idx] = m
        if (out_frame_idx + 1) % 250 == 0:
            elapsed = time.time() - t0
            rate = (out_frame_idx + 1) / max(0.1, elapsed)
            print(f"    SAM 2 frame {out_frame_idx+1}/{len(masks)} ({rate:.1f} fps)", flush=True)

    print(f"  [SAM 2] propagation done in {time.time()-t0:.0f}s, {len(masks)} masks", flush=True)
    return masks


# ─── Stage 2: Pose on masked frames ──────────────────────────────────────────

def build_pose_landmarker():
    model_path = "/home/max/boxingnew1/pose_landmarker_full.task"
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def detect_pose_on_masked(landmarker, frame_bgr, mask, ts_ms):
    """Mask out everything except Guardado, then detect pose."""
    if mask is None or mask.sum() < 100:
        return None
    # Black out background
    masked = frame_bgr.copy()
    masked[~mask] = 0
    rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    r = landmarker.detect_for_video(mp_img, ts_ms)
    return r.pose_landmarks[0] if r.pose_landmarks else None


# ─── Stage 3: Per-fault reel generation (reuse existing logic) ───────────────

# Import the existing per-fault reel rendering
sys.path.insert(0, '/home/max/boxingnew1')
import per_fault_reels as pfr


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--outdir",       default="site/reels")
    p.add_argument("--start",        type=float, default=98.0)
    p.add_argument("--duration",     type=float, default=419.0)
    p.add_argument("--seed-frame",   type=int,   default=0,
                   help="frame INSIDE the [start, end] window to use for seed")
    p.add_argument("--prompt",       default="boxer wearing black shorts.",
                   help="Grounding DINO prompt for Guardado")
    p.add_argument("--seed-box",     type=str, default=None,
                   help="Manual seed bbox 'x1,y1,x2,y2' overriding Grounding DINO")
    p.add_argument("--per-fault",    type=int, default=4)
    p.add_argument("--slowmo",       type=int, default=4)
    p.add_argument("--frames-cache", default="/tmp/sam2_frames",
                   help="dir to cache extracted JPEG frames")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    start_f = int(args.start * fps)
    end_f = min(total, start_f + int(args.duration * fps))
    print(f"[Video] {args.source} {W}x{H} @ {fps:.0f}fps", flush=True)
    print(f"[Window] frames {start_f}..{end_f} ({(end_f-start_f)/fps:.0f}s)", flush=True)

    # ── STAGE 0: Extract frames as JPEGs for SAM 2 ───────────────────────
    print("\n[Stage 0] Extracting frames", flush=True)
    n_extracted = extract_frames(args.source, start_f, end_f, W, H, args.frames_cache)

    # ── STAGE 1: Find Guardado seed box ──────────────────────────────────
    print("\n[Stage 1] Identifying Guardado seed bbox", flush=True)
    seed_path = os.path.join(args.frames_cache, f"{args.seed_frame:05d}.jpg")
    seed_img = cv2.imread(seed_path)
    if args.seed_box:
        seed_box = tuple(int(x) for x in args.seed_box.split(","))
        print(f"  Manual seed box: {seed_box}", flush=True)
    else:
        seed_box = find_guardado_box(seed_img, prompt=args.prompt)
        print(f"  Grounding DINO seed box: {seed_box}", flush=True)

    # Save annotated seed frame for visual verification
    debug_img = seed_img.copy()
    cv2.rectangle(debug_img, (seed_box[0], seed_box[1]), (seed_box[2], seed_box[3]),
                  (0, 255, 0), 3)
    cv2.putText(debug_img, "GUARDADO (seed)", (seed_box[0], seed_box[1] - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    debug_out = os.path.join(args.outdir, "_seed_frame.jpg")
    os.makedirs(args.outdir, exist_ok=True)
    cv2.imwrite(debug_out, debug_img)
    print(f"  Seed frame visualization saved to {debug_out}", flush=True)

    # ── STAGE 2: SAM 2 video propagation ─────────────────────────────────
    print("\n[Stage 2] SAM 2 video mask propagation", flush=True)
    device = torch.device("cuda")
    predictor = build_sam2_video_predictor(
        "configs/sam2/sam2_hiera_l.yaml",
        "/home/max/sam2/checkpoints/sam2_hiera_large.pt",
        device=device,
    )
    masks = propagate_with_sam2(
        predictor, args.frames_cache, seed_box, ann_frame_idx=args.seed_frame)

    # ── STAGE 3: Pose on masked frames ───────────────────────────────────
    print("\n[Stage 3] Pose estimation on Guardado-only masked frames", flush=True)
    landmarker = build_pose_landmarker()
    detector = PatternDetector()

    metas = []
    last_status = time.time()
    t0 = time.time()
    for local_idx in range(n_extracted):
        frame_path = os.path.join(args.frames_cache, f"{local_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None: continue
        mask = masks.get(local_idx)
        global_idx = start_f + local_idx
        ts_ms = int(global_idx * 1000 / fps)
        lms = detect_pose_on_masked(landmarker, frame, mask, ts_ms)
        analysis = detector.analyze(lms, W, H) if lms else FrameAnalysis()
        metas.append(pfr.FrameMeta(idx=global_idx, lms=lms, analysis=analysis))

        if time.time() - last_status > 5:
            done = local_idx + 1
            rate = done / max(0.1, time.time() - t0)
            eta = (n_extracted - done) / max(0.1, rate)
            with_pose = sum(1 for m in metas if m.lms is not None)
            print(f"  pose frame {done}/{n_extracted} | {rate:.1f} fps | ETA {eta:.0f}s | with_pose={with_pose}", flush=True)
            last_status = time.time()

    landmarker.close()
    print(f"  pose done in {time.time()-t0:.0f}s — {len(metas)} frames", flush=True)

    summary = detector.get_session_summary()
    print("\n[Top faults]", flush=True)
    for name, pct in list(summary.items())[:8]:
        print(f"  {name:<40} {pct:5.1f}%", flush=True)

    # Save summary
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "frames_analyzed": len(metas),
            "guardado_pose_frames": sum(1 for m in metas if m.lms is not None),
            "fault_percentages": summary,
            "method": "SAM2 mask + MediaPipe pose on masked region",
            "seed_box": list(seed_box),
            "prompt": args.prompt,
        }, f, indent=2)

    # ── STAGE 4: Render per-fault reels ──────────────────────────────────
    print(f"\n[Stage 4] Rendering per-fault reels", flush=True)
    rendered = {}
    for fault_name in summary.keys():
        slug = pfr.slugify(fault_name)
        out = os.path.join(args.outdir, f"{slug}.mp4")
        idxs = pfr.select_top_per_fault(metas, fault_name, args.per_fault, min_gap_frames=80)
        print(f"  → {fault_name} ({len(idxs)} moments)", flush=True)
        if pfr.render_fault_reel(args.source, metas, fault_name, idxs,
                                 out, fps, W, H, scale=1.0, slowmo=args.slowmo):
            rendered[fault_name] = out

    print(f"\n[DONE] {len(rendered)} per-fault reels written to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
