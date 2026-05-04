#!/usr/bin/env python3
"""
DINOv2 ReID + Manual-Review-Ready Pipeline
==========================================
Path C, Step 1: replace CLIP with DINOv2 (better fine-grained
discrimination) AND generate top-N candidate stills per fault for
manual review instead of auto-rendering reels.

Output:
  outdir/
    summary.json          — fault percentages + filter stats
    candidates.json       — for each fault, list of top-N candidate moments
                            with frame_idx, score, jpg path, bbox
    candidates/
      <fault_slug>/
        c001.jpg          — annotated still (bbox + skeleton + label)
        c002.jpg
        ...
  Then: open review.html (Path B step 2) to approve/reject candidates,
  then run render_approved_reels.py to build the final reels.

Usage:
  python dinov2_reid_pipeline.py video.mp4 \\
      --start 98 --duration 382 \\
      --seed-frame-time 130 --seed-box 208,163,638,715 \\
      --neg-seed-time 130 --neg-seed-box 747,123,986,714 \\
      --top-n 10 --outdir site/review
"""

import os, sys, argparse, time, json, re
import numpy as np
import cv2
import torch
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

from ultralytics import YOLO
from transformers import AutoModel, AutoImageProcessor

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boxing_analyzer.pattern_detector import PatternDetector, FrameAnalysis, Fault
from boxing_analyzer.landmarks import LM, get_point


# ─── DINOv2 ReID ────────────────────────────────────────────────────────────

class DINOv2ReID:
    """DINOv2-based person re-identification."""

    def __init__(self, model_name: str = "facebook/dinov2-large", device: str = "cuda"):
        print(f"  [DINOv2] loading {model_name}", flush=True)
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.template = None

    @torch.no_grad()
    def encode(self, frame_bgr, boxes_xyxy):
        if len(boxes_xyxy) == 0:
            return torch.zeros(0, self.model.config.hidden_size, device=self.device)
        crops = []
        h, w = frame_bgr.shape[:2]
        for box in boxes_xyxy:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 + 8 or y2 <= y1 + 8:
                crops.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(rgb))
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state[:, 0]  # CLS token
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def build_template(self, seed_crops_bgr):
        all_feats = []
        for crop_bgr in seed_crops_bgr:
            h, w = crop_bgr.shape[:2]
            f = self.encode(crop_bgr, [[0, 0, w, h]])
            all_feats.append(f[0])
        template = torch.stack(all_feats).mean(dim=0)
        template = template / template.norm()
        self.template = template
        print(f"  [DINOv2] template built from {len(seed_crops_bgr)} crops", flush=True)
        return template

    def best_match(self, frame_bgr, boxes_xyxy, threshold: float,
                   negative_template=None, margin: float = 0.05):
        if self.template is None:
            raise RuntimeError("Template not built")
        if len(boxes_xyxy) == 0:
            return None, -1.0, -1.0, 0.0, -1
        feats = self.encode(frame_bgr, boxes_xyxy)
        sims_pos = (feats @ self.template).cpu().numpy()
        if negative_template is not None:
            sims_neg = (feats @ negative_template).cpu().numpy()
        else:
            sims_neg = np.zeros_like(sims_pos)
        score = sims_pos - sims_neg
        best_idx = int(np.argmax(score))
        best_pos = float(sims_pos[best_idx])
        best_neg = float(sims_neg[best_idx])
        best_margin = best_pos - best_neg
        if best_pos < threshold:
            return None, best_pos, best_neg, best_margin, best_idx
        if negative_template is not None and best_margin < margin:
            return None, best_pos, best_neg, best_margin, best_idx
        return boxes_xyxy[best_idx], best_pos, best_neg, best_margin, best_idx


# ─── YOLOv8 detection ───────────────────────────────────────────────────────

def build_yolo(size="x"):
    weights = f"yolov8{size}.pt"
    print(f"  [YOLO] loading {weights}", flush=True)
    return YOLO(weights)


def detect_persons(yolo, frame, device):
    res = yolo.predict(frame, classes=[0], verbose=False, device=device)[0]
    if len(res.boxes) == 0:
        return np.zeros((0, 4))
    return res.boxes.xyxy.cpu().numpy()


# ─── Pose ───────────────────────────────────────────────────────────────────

def build_pose_landmarker(model_path):
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


def draw_skeleton_inline(img, lms, w, h, color=(0, 255, 0), color_fault=(0, 0, 255)):
    """Draw skeleton overlay on img in-place."""
    if lms is None: return
    pairs = [
        (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
        (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
        (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
        (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
        (LM.LEFT_HIP, LM.RIGHT_HIP),
        (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
        (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ]
    for a, b in pairs:
        try:
            pa = get_point(lms, a, w, h).astype(int)
            pb = get_point(lms, b, w, h).astype(int)
            cv2.line(img, tuple(pa), tuple(pb), color, 3, cv2.LINE_AA)
        except Exception:
            pass
    for idx in range(33):
        try:
            pt = get_point(lms, idx, w, h).astype(int)
            cv2.circle(img, tuple(pt), 4, color, -1, cv2.LINE_AA)
        except Exception:
            pass


def slugify(s):
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return s


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--outdir",          default="site/review")
    p.add_argument("--start",           type=float, default=98.0)
    p.add_argument("--duration",        type=float, default=382.0)
    p.add_argument("--seed-frame-time", type=float, action="append", required=True)
    p.add_argument("--seed-box",        type=str,   action="append", required=True)
    p.add_argument("--neg-seed-time",   type=float, action="append", default=[])
    p.add_argument("--neg-seed-box",    type=str,   action="append", default=[])
    p.add_argument("--top-n",           type=int, default=10,
                   help="Top N candidate moments per fault for review")
    p.add_argument("--min-gap",         type=int, default=80,
                   help="Min frames between two candidate moments of same fault")
    p.add_argument("--threshold",       type=float, default=0.55,
                   help="DINOv2 sim threshold (typical: 0.5-0.6, lower than CLIP)")
    p.add_argument("--margin",          type=float, default=0.03)
    p.add_argument("--yolo",            default="x")
    p.add_argument("--dinov2",          default="facebook/dinov2-large")
    args = p.parse_args()

    if len(args.seed_frame_time) != len(args.seed_box):
        sys.exit("ERROR: seed-frame-time and seed-box count mismatch")

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(args.start * fps)
    end_f = min(total, start_f + int(args.duration * fps))

    print(f"[Video] {args.source} {W}x{H} @ {fps:.0f}fps", flush=True)
    print(f"[Window] frames {start_f}..{end_f} ({(end_f-start_f)/fps:.0f}s)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}", flush=True)

    yolo = build_yolo(args.yolo)
    reid = DINOv2ReID(args.dinov2, device=device)
    pose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")
    landmarker = build_pose_landmarker(pose_path)
    detector = PatternDetector()

    # ── Build positive template ──────────────────────────────────────────
    print("\n[Template] extracting Guardado seed crop(s)", flush=True)
    seed_crops = []
    os.makedirs(args.outdir, exist_ok=True)
    for stime, sbox_str in zip(args.seed_frame_time, args.seed_box):
        sbox = tuple(int(x) for x in sbox_str.split(","))
        sframe_idx = int(stime * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sframe_idx)
        ret, frame = cap.read()
        if not ret: continue
        x1, y1, x2, y2 = sbox
        crop = frame[y1:y2, x1:x2]
        seed_crops.append(crop)
        cv2.imwrite(os.path.join(args.outdir, f"_seed_t{int(stime)}.jpg"), crop)
    if not seed_crops: sys.exit("No valid seed crops")
    reid.build_template(seed_crops)

    # ── Build negative template ──────────────────────────────────────────
    negative_template = None
    if args.neg_seed_time and args.neg_seed_box:
        print("\n[NegTemplate] extracting Conceição seed crop(s)", flush=True)
        neg_crops = []
        for stime, sbox_str in zip(args.neg_seed_time, args.neg_seed_box):
            sbox = tuple(int(x) for x in sbox_str.split(","))
            sframe_idx = int(stime * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sframe_idx)
            ret, frame = cap.read()
            if not ret: continue
            x1, y1, x2, y2 = sbox
            crop = frame[y1:y2, x1:x2]
            neg_crops.append(crop)
            cv2.imwrite(os.path.join(args.outdir, f"_neg_seed_t{int(stime)}.jpg"), crop)
        all_neg_feats = []
        for crop_bgr in neg_crops:
            h_, w_ = crop_bgr.shape[:2]
            f = reid.encode(crop_bgr, [[0, 0, w_, h_]])
            all_neg_feats.append(f[0])
        negative_template = torch.stack(all_neg_feats).mean(dim=0)
        negative_template = negative_template / negative_template.norm()
        print(f"  [NegTemplate] built from {len(neg_crops)} crops", flush=True)

    # ── Pass 1: ReID + pose, store per-frame meta ────────────────────────
    print("\n[Pass 1] DINOv2 ReID + pose", flush=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fi = start_f
    metas = []   # list of dicts: idx, lms, analysis, score
    pose_found = 0
    no_match = 0
    sim_log = []
    t0 = time.time()
    last_status = t0

    while cap.isOpened() and fi < end_f:
        ret, frame = cap.read()
        if not ret: break
        boxes = detect_persons(yolo, frame, device)
        if len(boxes) == 0:
            metas.append({"idx": fi, "lms": None, "analysis": FrameAnalysis(),
                          "bbox": None, "score": -1, "sim_pos": -1, "sim_neg": -1})
            fi += 1; continue
        gbox, sim_pos, sim_neg, m, _idx = reid.best_match(
            frame, boxes, args.threshold, negative_template, args.margin)
        sim_log.append(sim_pos)
        if gbox is None:
            no_match += 1
            metas.append({"idx": fi, "lms": None, "analysis": FrameAnalysis(),
                          "bbox": None, "score": m, "sim_pos": sim_pos, "sim_neg": sim_neg})
        else:
            ts_ms = int(fi * 1000 / fps)
            lms = pose_on_bbox(landmarker, frame, gbox, ts_ms, W, H)
            if lms is not None: pose_found += 1
            analysis = detector.analyze(lms, W, H) if lms else FrameAnalysis()
            metas.append({"idx": fi, "lms": lms, "analysis": analysis,
                          "bbox": [int(v) for v in gbox],
                          "score": m, "sim_pos": sim_pos, "sim_neg": sim_neg})
        fi += 1
        if time.time() - last_status > 5:
            done = fi - start_f
            rate = done / max(0.1, time.time() - t0)
            eta = (end_f - fi) / max(0.1, rate)
            print(f"  frame {fi}/{end_f} ({done/(end_f-start_f)*100:.0f}%) | {rate:.1f} fps | ETA {eta:.0f}s | pose={pose_found} no_match={no_match}", flush=True)
            last_status = time.time()

    cap.release()
    landmarker.close()
    print(f"\n[Pass 1] done in {time.time()-t0:.0f}s", flush=True)
    print(f"  pose found: {pose_found}, no_match: {no_match}", flush=True)
    print(f"  sim stats: mean={np.mean(sim_log):.3f} max={np.max(sim_log):.3f}", flush=True)

    summary = detector.get_session_summary()
    print("\n[Top faults]", flush=True)
    for n, p in list(summary.items())[:10]:
        print(f"  {n:<40} {p:5.1f}%", flush=True)

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump({
            "method": f"YOLOv8 + DINOv2 ReID ({args.dinov2}) + MediaPipe pose",
            "frames_analyzed": len(metas),
            "guardado_pose_frames": pose_found,
            "no_match_frames": no_match,
            "fault_percentages": summary,
            "threshold": args.threshold,
            "margin": args.margin,
        }, f, indent=2)

    # ── Pass 2: pick top-N candidates per fault, save annotated stills ──
    print(f"\n[Pass 2] saving top-{args.top_n} candidates per fault", flush=True)
    cand_dir = os.path.join(args.outdir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    candidates_manifest = {}

    cap = cv2.VideoCapture(args.source)
    for fault_name in summary.keys():
        slug = slugify(fault_name)
        fault_dir = os.path.join(cand_dir, slug)
        os.makedirs(fault_dir, exist_ok=True)

        # Gather all candidate frames for this fault, sorted by confidence
        scored = []
        for fm in metas:
            if fm["lms"] is None: continue
            for f in fm["analysis"].faults:
                if f.name == fault_name:
                    scored.append((fm["idx"], f.confidence, fm))
        scored.sort(key=lambda x: -x[1])

        # Enforce min gap between picks
        picked = []
        last_idxs = []
        for idx, conf, fm in scored:
            if all(abs(idx - u) >= args.min_gap for u in last_idxs):
                picked.append((idx, conf, fm))
                last_idxs.append(idx)
                if len(picked) >= args.top_n: break

        candidates_for_fault = []
        for ci, (frame_idx, conf, fm) in enumerate(picked):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            # Annotate: draw bbox + skeleton + label
            x1, y1, x2, y2 = fm["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{fault_name} | conf={conf:.0%} | DINOv2 margin={fm['score']:.2f}"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_idx} (t={frame_idx/fps:.1f}s)", (10, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
            draw_skeleton_inline(frame, fm["lms"], W, H)
            jpg_path = os.path.join(fault_dir, f"c{ci+1:02d}.jpg")
            cv2.imwrite(jpg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            candidates_for_fault.append({
                "candidate_id": ci + 1,
                "frame_idx": int(frame_idx),
                "time_sec": round(frame_idx / fps, 2),
                "confidence": round(conf, 3),
                "dinov2_margin": round(fm["score"], 3),
                "bbox": fm["bbox"],
                "jpg": f"candidates/{slug}/c{ci+1:02d}.jpg",
            })

        candidates_manifest[fault_name] = {
            "slug": slug,
            "percentage": summary.get(fault_name, 0),
            "candidates": candidates_for_fault,
        }
        print(f"  → {fault_name}: {len(candidates_for_fault)} candidates", flush=True)
    cap.release()

    with open(os.path.join(args.outdir, "candidates.json"), "w") as f:
        json.dump(candidates_manifest, f, indent=2)

    print(f"\n[DONE] candidates in {args.outdir}/candidates/, manifest at {args.outdir}/candidates.json", flush=True)


if __name__ == "__main__":
    main()
