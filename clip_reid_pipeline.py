#!/usr/bin/env python3
"""
CLIP Re-Identification Pipeline for Guardado-Only Boxing Analysis
==================================================================
Replaces brittle color filters with CLIP visual embeddings.

Approach:
  1. Build a Guardado feature template from seed frame(s) — encode the
     known-Guardado crop with CLIP to get a 512-dim semantic vector.
  2. For each frame: YOLOv8 detects all persons. Crop each person,
     encode with CLIP, compute cosine similarity to the template.
     The person with highest similarity is Guardado.
  3. Threshold: skip frame if best similarity < threshold (no
     confident Guardado match).
  4. Pose runs only on Guardado's bbox crop (same as before).

Why CLIP works better than color/tracking:
  - Learns semantic features (body shape, posture, hairline, complexion,
    clothing pattern) — robust to lighting / motion blur
  - Doesn't depend on brittle color thresholds
  - Doesn't need temporal continuity (works frame-by-frame)
  - Doesn't depend on track IDs that swap during occlusion

Usage:
  python clip_reid_pipeline.py video.mp4 \\
      --start 98 --duration 419 \\
      --seed-frame-time 130 --seed-box 208,163,638,715 \\
      --outdir site/reels
"""

import os, sys, argparse, time, json
import numpy as np
import cv2
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point
import per_fault_reels as pfr


# ─── CLIP ReID ───────────────────────────────────────────────────────────────

class CLIPReID:
    """CLIP-based person re-identification."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: str = "cuda"):
        print(f"  [CLIP] loading {model_name}", flush=True)
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        # We'll use only the vision encoder for ReID
        self.template = None  # will be populated by build_template

    @torch.no_grad()
    def encode(self, frame_bgr, boxes_xyxy):
        """Encode a list of person crops to L2-normalized feature vectors."""
        if len(boxes_xyxy) == 0:
            return torch.zeros(0, self.model.config.projection_dim, device=self.device)
        crops = []
        h, w = frame_bgr.shape[:2]
        for box in boxes_xyxy:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 + 8 or y2 <= y1 + 8:
                # Use a black placeholder for invalid crops
                crops.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb))
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        # Some transformers versions return a structured output, others a tensor
        if hasattr(out, "image_embeds"):
            feats = out.image_embeds
        elif hasattr(out, "pooler_output"):
            feats = out.pooler_output
        else:
            feats = out  # already a tensor
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def build_template(self, seed_crops_bgr):
        """Build Guardado template from one or more seed crops."""
        all_feats = []
        for crop_bgr in seed_crops_bgr:
            h, w = crop_bgr.shape[:2]
            box = [0, 0, w, h]  # full crop
            f = self.encode(crop_bgr, [box])  # shape (1, D)
            all_feats.append(f[0])
        template = torch.stack(all_feats).mean(dim=0)
        template = template / template.norm()
        self.template = template
        print(f"  [CLIP] template built from {len(seed_crops_bgr)} seed crop(s)",
              flush=True)
        return template

    def best_match(self, frame_bgr, boxes_xyxy, threshold: float):
        """
        Return (best_box, best_similarity) or (None, similarity) if below threshold.
        """
        if self.template is None:
            raise RuntimeError("Template not built — call build_template first")
        if len(boxes_xyxy) == 0:
            return None, -1.0
        feats = self.encode(frame_bgr, boxes_xyxy)  # (N, D)
        sims = feats @ self.template  # cosine similarity since both L2-normalized
        sims = sims.cpu().numpy()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < threshold:
            return None, best_sim
        return boxes_xyxy[best_idx], best_sim


# ─── YOLOv8 detection (no tracking needed; CLIP handles identity) ───────────

def build_yolo(model_size: str = "x"):
    weights = f"yolov8{model_size}.pt"
    print(f"  [YOLO] loading {weights}", flush=True)
    return YOLO(weights)


def detect_persons(yolo, frame, device):
    """Return person bboxes (Nx4 xyxy) from YOLO."""
    res = yolo.predict(frame, classes=[0], verbose=False, device=device)[0]
    if len(res.boxes) == 0:
        return np.zeros((0, 4))
    return res.boxes.xyxy.cpu().numpy()


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


def pose_on_bbox(landmarker, frame_bgr, box_xyxy, ts_ms, w, h, padding=0.10):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    bw = x2 - x1; bh = y2 - y1
    pad_x = int(bw * padding); pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
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
    p.add_argument("--seed-frame-time", type=float, action="append", required=True,
                   help="Time(s) of seed frame(s); pass multiple for robustness")
    p.add_argument("--seed-box",        type=str, action="append", required=True,
                   help="Manual seed bbox(es) 'x1,y1,x2,y2' for Guardado at each seed time")
    p.add_argument("--per-fault",       type=int, default=4)
    p.add_argument("--slowmo",          type=int, default=4)
    p.add_argument("--yolo",            default="x")
    p.add_argument("--clip",            default="openai/clip-vit-large-patch14")
    p.add_argument("--threshold",       type=float, default=0.78,
                   help="Min cosine similarity to accept as Guardado")
    p.add_argument("--save-debug",      action="store_true",
                   help="Save annotated debug frames every N frames")
    args = p.parse_args()

    if len(args.seed_frame_time) != len(args.seed_box):
        print("ERROR: number of --seed-frame-time and --seed-box must match",
              flush=True)
        sys.exit(1)

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = int(args.start * fps)
    end_f = min(total, start_f + int(args.duration * fps))

    print(f"[Video] {args.source} {W}x{H} @ {fps:.0f}fps", flush=True)
    print(f"[Window] frames {start_f}..{end_f} ({(end_f-start_f)/fps:.0f}s)", flush=True)

    # ── Setup ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}", flush=True)

    yolo = build_yolo(args.yolo)
    reid = CLIPReID(args.clip, device=device)
    pose_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "pose_landmarker_full.task")
    landmarker = build_pose_landmarker(pose_model_path)
    detector = PatternDetector()

    # ── Build Guardado template from seed frame(s) ───────────────────────
    print("\n[Template] extracting Guardado seed crop(s)", flush=True)
    seed_crops = []
    for stime, sbox_str in zip(args.seed_frame_time, args.seed_box):
        sbox = tuple(int(x) for x in sbox_str.split(","))
        sframe_idx = int(stime * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sframe_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARN: could not read seed frame at t={stime}", flush=True)
            continue
        x1, y1, x2, y2 = sbox
        crop = frame[y1:y2, x1:x2]
        seed_crops.append(crop)
        # Save for visual confirmation
        os.makedirs(args.outdir, exist_ok=True)
        cv2.imwrite(os.path.join(args.outdir, f"_seed_t{int(stime)}.jpg"), crop)
        print(f"  seed t={stime}s bbox={sbox} -> crop {crop.shape}", flush=True)
    if not seed_crops:
        print("ERROR: no valid seed crops", flush=True)
        sys.exit(1)
    reid.build_template(seed_crops)

    # ── Pass 1: per-frame CLIP ReID + pose ───────────────────────────────
    print("\n[Pass 1] CLIP ReID + pose on Guardado", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fi = start_f
    metas = []
    pose_found = 0
    no_match = 0
    no_persons = 0
    sim_log = []
    debug_dir = os.path.join(args.outdir, "_debug")
    if args.save_debug:
        os.makedirs(debug_dir, exist_ok=True)
    t0 = time.time()
    last_status = t0

    while cap.isOpened() and fi < end_f:
        ret, frame = cap.read()
        if not ret: break

        boxes = detect_persons(yolo, frame, device)
        if len(boxes) == 0:
            no_persons += 1
            metas.append(pfr.FrameMeta(idx=fi, lms=None, analysis=FrameAnalysis()))
            fi += 1
            continue

        gbox, sim = reid.best_match(frame, boxes, args.threshold)
        sim_log.append(sim)

        if gbox is None:
            no_match += 1
            metas.append(pfr.FrameMeta(idx=fi, lms=None, analysis=FrameAnalysis()))
        else:
            ts_ms = int(fi * 1000 / fps)
            lms = pose_on_bbox(landmarker, frame, gbox, ts_ms, W, H)
            if lms is not None:
                pose_found += 1
            analysis = detector.analyze(lms, W, H) if lms else FrameAnalysis()
            metas.append(pfr.FrameMeta(idx=fi, lms=lms, analysis=analysis))

            # Debug image every 500 frames showing the chosen bbox
            if args.save_debug and fi % 500 == 0:
                dbg = frame.copy()
                x1, y1, x2, y2 = [int(v) for v in gbox]
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(dbg, f"GUARDADO sim={sim:.2f}", (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(debug_dir, f"f{fi:05d}.jpg"), dbg)

        fi += 1
        if time.time() - last_status > 5:
            done = fi - start_f
            rate = done / max(0.1, time.time() - t0)
            eta = (end_f - fi) / max(0.1, rate)
            mean_sim = float(np.mean(sim_log[-200:])) if sim_log else 0.0
            print(f"  frame {fi}/{end_f} ({done/(end_f-start_f)*100:.0f}%) | {rate:.1f} fps | ETA {eta:.0f}s | pose={pose_found} no_match={no_match} mean_sim_recent={mean_sim:.3f}",
                  flush=True)
            last_status = time.time()

    cap.release()
    landmarker.close()

    print(f"\n[Pass 1] done in {time.time()-t0:.0f}s", flush=True)
    print(f"  total frames: {len(metas)}", flush=True)
    print(f"  Guardado pose: {pose_found}", flush=True)
    print(f"  no_match (sim < {args.threshold}): {no_match}", flush=True)
    print(f"  no_persons detected: {no_persons}", flush=True)
    if sim_log:
        print(f"  similarity stats: mean={np.mean(sim_log):.3f} min={np.min(sim_log):.3f} max={np.max(sim_log):.3f}", flush=True)

    summary = detector.get_session_summary()
    print("\n[Top faults]", flush=True)
    for name, pct in list(summary.items())[:10]:
        print(f"  {name:<40} {pct:5.1f}%", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "method": f"YOLOv8 + CLIP ReID ({args.clip}) + MediaPipe pose",
            "frames_analyzed": len(metas),
            "guardado_pose_frames": pose_found,
            "no_match_frames": no_match,
            "no_persons_frames": no_persons,
            "similarity_threshold": args.threshold,
            "fault_percentages": summary,
            "seed_frame_times": args.seed_frame_time,
            "seed_boxes": args.seed_box,
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

    print(f"\n[DONE] {rendered} per-fault reels in {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
