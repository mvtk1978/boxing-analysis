#!/usr/bin/env python3
"""
Boxing Video Analysis Tool
===========================
Usage:
  python analyze.py <youtube_url_or_local_path> [options]

Options:
  --output PATH      Save annotated video to file (default: boxing_output.mp4)
  --show             Display live preview window (requires display)
  --start SEC        Start time in seconds (default: 0)
  --duration SEC     Duration to analyze in seconds (default: full video)
  --no-save          Don't save output video
  --summary-only     Print only the session fault summary
"""

import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path

from boxing_analyzer.pattern_detector import PatternDetector
from boxing_analyzer.visualizer import BoxingVisualizer
from boxing_analyzer.video_io import download_youtube, is_youtube_url


def build_argparser():
    p = argparse.ArgumentParser(description="Boxing Video Analysis Tool")
    p.add_argument("source", help="YouTube URL or local video file path")
    p.add_argument("--output", default="boxing_output.mp4",
                   help="Output annotated video path")
    p.add_argument("--show", action="store_true",
                   help="Show live window (needs display)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start time in seconds")
    p.add_argument("--duration", type=float, default=None,
                   help="Duration to analyze in seconds")
    p.add_argument("--no-save", action="store_true",
                   help="Don't save output video")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale video for faster processing (0.5 = half size, range 0.1-2.0)")
    return p


def _validate_args(args):
    if not (0.1 <= args.scale <= 2.0):
        raise ValueError(f"--scale must be between 0.1 and 2.0, got {args.scale}")
    if args.start < 0:
        raise ValueError(f"--start must be >= 0, got {args.start}")
    if args.duration is not None and args.duration <= 0:
        raise ValueError(f"--duration must be > 0, got {args.duration}")


def _setup_pose(model_path: str):
    """
    Setup MediaPipe pose estimation.
    Tries new Tasks API first (mediapipe >= 0.10), falls back to legacy solutions API.
    Returns a callable: frame_bgr -> list of 33 landmarks (or None)
    """
    try:
        # New Tasks API (mediapipe >= 0.10)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        print("[Pose] Using MediaPipe Tasks API (PoseLandmarker)")

        def detect_new(frame_bgr, timestamp_ms):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.pose_landmarks:
                return result.pose_landmarks[0]
            return None

        return detect_new, landmarker

    except (AttributeError, ImportError, FileNotFoundError) as e:
        print(f"[Pose] Tasks API unavailable ({e}), trying legacy solutions API...")
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose_legacy = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[Pose] Using MediaPipe legacy solutions.Pose API")

            def detect_legacy(frame_bgr, timestamp_ms):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = pose_legacy.process(rgb)
                if res.pose_landmarks:
                    return res.pose_landmarks.landmark
                return None

            return detect_legacy, pose_legacy
        except Exception as e2:
            raise RuntimeError(f"MediaPipe pose setup failed: {e2}")


def run_analysis(video_path: str, args):
    # ----------------------------------------------------------------
    # Setup MediaPipe Pose (auto-select API version)
    # ----------------------------------------------------------------
    model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
    detect_pose, pose_handle = _setup_pose(model_path)

    detector = PatternDetector()
    visualizer = BoxingVisualizer(panel_width=400)

    # ----------------------------------------------------------------
    # Open video
    # ----------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Video] {os.path.basename(video_path)}")
    print(f"  Resolution: {orig_w}x{orig_h}  FPS: {fps:.1f}  Frames: {total_frames}")

    # Seek to start
    start_frame = int(args.start * fps)
    end_frame = total_frames
    if args.duration is not None:
        end_frame = start_frame + int(args.duration * fps)
    end_frame = min(end_frame, total_frames)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scale = args.scale
    out_w = int(orig_w * scale)
    out_h = int(orig_h * scale)
    combined_w = out_w + 400   # frame + panel

    # ----------------------------------------------------------------
    # Video writer
    # ----------------------------------------------------------------
    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (combined_w, out_h))
        print(f"[Output] Writing to: {args.output}")

    # ----------------------------------------------------------------
    # Main processing loop
    # ----------------------------------------------------------------
    frame_idx = start_frame
    processed = 0
    fault_frames = 0

    print(f"\n[Analysis] Processing frames {start_frame} – {end_frame}...\n")

    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Scale
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))

        h, w = frame.shape[:2]

        # Pose detection (timestamp needed for Tasks API video mode)
        timestamp_ms = int(frame_idx * 1000 / fps)
        lms = detect_pose(frame, timestamp_ms)

        if lms is not None:
            analysis = detector.analyze(lms, w, h)
            session_summary = detector.get_session_summary()

            if analysis.faults:
                fault_frames += 1

            # Draw visualization
            combined = visualizer.draw(frame, lms, analysis,
                                       session_summary, frame_idx)
        else:
            # No person detected — pad with blank panel
            blank = np.zeros((h, 400, 3), dtype=np.uint8)
            cv2.putText(blank, "No pose detected", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            combined = np.hstack([frame, blank])
            analysis = None

        if writer:
            writer.write(combined)

        if args.show:
            cv2.imshow("Boxing Analysis", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[User] Quit")
                break

        processed += 1
        if processed % 30 == 0:
            pct = (frame_idx - start_frame) / max(1, end_frame - start_frame) * 100
            print(f"  Frame {frame_idx}/{end_frame} ({pct:.0f}%) | "
                  f"Faults detected in {fault_frames}/{processed} frames "
                  f"({fault_frames/processed*100:.0f}%)")

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    if hasattr(pose_handle, 'close'):
        pose_handle.close()

    # ----------------------------------------------------------------
    # Session summary report
    # ----------------------------------------------------------------
    summary = detector.get_session_summary()
    print("\n" + "=" * 60)
    print("  BOXING ANALYSIS — SESSION FAULT REPORT")
    print("=" * 60)
    print(f"  Frames analyzed : {processed}")
    print(f"  Frames with faults: {fault_frames} ({fault_frames/max(1,processed)*100:.1f}%)")
    print(f"  Stance detected : {detector.stance or 'unknown'}")
    print()
    print("  FAULT FREQUENCY (% of analyzed frames)")
    print("  " + "-" * 45)
    if summary:
        for fault_name, pct in summary.items():
            bar_len = int(pct / 2)
            bar = "█" * bar_len
            severity_marker = " [!!]" if pct > 60 else ""
            print(f"  {fault_name:<35} {pct:5.1f}% {bar}{severity_marker}")
    else:
        print("  No faults detected — great technique!")
    print("=" * 60)

    if writer:
        print(f"\n  Annotated video saved to: {args.output}")

    return summary


def main():
    parser = build_argparser()
    args = parser.parse_args()
    _validate_args(args)

    source = args.source
    video_path = source

    # Download if YouTube
    if is_youtube_url(source):
        print(f"[YouTube] Detected YouTube URL")
        video_path = download_youtube(source)

    if not os.path.exists(video_path):
        print(f"[Error] File not found: {video_path}")
        sys.exit(1)

    run_analysis(video_path, args)


if __name__ == "__main__":
    main()
