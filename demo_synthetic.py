#!/usr/bin/env python3
"""
Synthetic demo — generates a test video with known faults injected,
runs the analyzer, and saves annotated output.

Use this to verify the tool works without needing a real video.
"""

import numpy as np
import cv2
import sys
import os
import argparse

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from boxing_analyzer.pattern_detector import PatternDetector
from boxing_analyzer.visualizer import BoxingVisualizer


def make_fake_landmarks(frame_idx: int, inject_fault: bool = True):
    """
    Returns a list of 33 fake MediaPipe landmark objects (with x, y, z, visibility).
    Positions encode an orthodox boxer with some injected faults.
    """

    class FakeLM:
        def __init__(self, x, y, z=0.0, visibility=0.99):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility

    t = frame_idx / 30.0  # time in seconds

    # Base body proportions (normalized 0-1, y down)
    # Center of fighter at x~0.45, y varies slightly
    cx = 0.45 + 0.01 * np.sin(t * 0.5)   # slight lateral sway
    cy = 0.3 + 0.005 * np.sin(t * 1.5)   # slight vertical bob

    # Body proportions as fractions of image height
    head_y   = cy - 0.10
    nose_y   = cy - 0.08
    sh_y     = cy + 0.03
    elbow_y  = cy + 0.12
    hip_y    = cy + 0.20
    knee_y   = cy + 0.35
    ankle_y  = cy + 0.50

    # Shoulder width
    sw = 0.09
    # Elbow width (tucked guard baseline)
    ew = 0.07
    # Wrist position — guard height
    wrist_y_normal   = sh_y - 0.04    # wrists at chin level
    wrist_y_dropped  = sh_y + 0.10    # wrists dropped below shoulders

    # === Fault injection ===
    lead_wrist_y = wrist_y_dropped if inject_fault else wrist_y_normal
    chin_y = nose_y + (0.04 if inject_fault else 0.0)   # chin up when fault
    l_elbow_x = cx - ew - (0.05 if inject_fault else 0.0)  # elbow flare

    lms = [None] * 33

    from boxing_analyzer.landmarks import LM

    # Head
    lms[LM.NOSE]           = FakeLM(cx, chin_y)
    lms[LM.LEFT_EYE_INNER] = FakeLM(cx - 0.02, head_y)
    lms[LM.LEFT_EYE]       = FakeLM(cx - 0.03, head_y)
    lms[LM.LEFT_EYE_OUTER] = FakeLM(cx - 0.04, head_y)
    lms[LM.RIGHT_EYE_INNER]= FakeLM(cx + 0.02, head_y)
    lms[LM.RIGHT_EYE]      = FakeLM(cx + 0.03, head_y)
    lms[LM.RIGHT_EYE_OUTER]= FakeLM(cx + 0.04, head_y)
    lms[LM.LEFT_EAR]       = FakeLM(cx - 0.05, head_y + 0.01)
    lms[LM.RIGHT_EAR]      = FakeLM(cx + 0.05, head_y + 0.01)
    lms[LM.MOUTH_LEFT]     = FakeLM(cx - 0.02, chin_y + 0.01)
    lms[LM.MOUTH_RIGHT]    = FakeLM(cx + 0.02, chin_y + 0.01)

    # Shoulders (orthodox — left shoulder slightly forward, smaller x)
    lms[LM.LEFT_SHOULDER]  = FakeLM(cx - sw, sh_y)
    lms[LM.RIGHT_SHOULDER] = FakeLM(cx + sw, sh_y)

    # Elbows
    lms[LM.LEFT_ELBOW]     = FakeLM(l_elbow_x, elbow_y)
    lms[LM.RIGHT_ELBOW]    = FakeLM(cx + ew, elbow_y)

    # Wrists — lead hand (left) dropped when fault
    lms[LM.LEFT_WRIST]     = FakeLM(cx - ew + 0.01, lead_wrist_y)
    lms[LM.RIGHT_WRIST]    = FakeLM(cx + ew - 0.01, wrist_y_normal)

    # Hands
    for idx in [LM.LEFT_PINKY, LM.LEFT_INDEX, LM.LEFT_THUMB]:
        lms[idx] = FakeLM(cx - ew, lead_wrist_y + 0.03)
    for idx in [LM.RIGHT_PINKY, LM.RIGHT_INDEX, LM.RIGHT_THUMB]:
        lms[idx] = FakeLM(cx + ew, wrist_y_normal + 0.03)

    # Hips
    lms[LM.LEFT_HIP]       = FakeLM(cx - 0.06, hip_y)
    lms[LM.RIGHT_HIP]      = FakeLM(cx + 0.06, hip_y)

    # Knees
    lms[LM.LEFT_KNEE]      = FakeLM(cx - 0.07, knee_y)
    lms[LM.RIGHT_KNEE]     = FakeLM(cx + 0.05, knee_y)

    # Ankles — orthodox stance: left foot forward (smaller x)
    lms[LM.LEFT_ANKLE]     = FakeLM(cx - 0.08, ankle_y)
    lms[LM.RIGHT_ANKLE]    = FakeLM(cx + 0.09, ankle_y)

    # Heels and foot index
    lms[LM.LEFT_HEEL]      = FakeLM(cx - 0.09, ankle_y + 0.02)
    lms[LM.RIGHT_HEEL]     = FakeLM(cx + 0.08, ankle_y + 0.02)
    lms[LM.LEFT_FOOT_INDEX]  = FakeLM(cx - 0.06, ankle_y + 0.03)
    lms[LM.RIGHT_FOOT_INDEX] = FakeLM(cx + 0.12, ankle_y + 0.03)

    return lms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_demo.mp4")
    parser.add_argument("--frames", type=int, default=150, help="Number of frames")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    W, H = 854, 480
    fps = 30

    detector = PatternDetector()
    visualizer = BoxingVisualizer(panel_width=400)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W + 400, H))

    print(f"[Demo] Generating {args.frames} frames → {args.output}")

    for i in range(args.frames):
        # Dark canvas with ring background
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (20, 20, 30)
        cv2.rectangle(frame, (50, 50), (W - 50, H - 50), (40, 40, 60), 2)
        cv2.putText(frame, "BOXING ANALYSIS — SYNTHETIC DEMO",
                    (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 80), 1)

        # Inject faults in first half, clean technique in second half
        inject = i < args.frames // 2
        lms = make_fake_landmarks(i, inject_fault=inject)

        analysis = detector.analyze(lms, W, H)
        summary = detector.get_session_summary()

        combined = visualizer.draw(frame, lms, analysis, summary, i)

        # Phase label
        phase = "FAULTY TECHNIQUE" if inject else "CORRECT TECHNIQUE"
        color = (50, 50, 200) if inject else (50, 180, 50)
        cv2.putText(combined, phase, (W // 2 - 100, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        writer.write(combined)

        if args.show:
            cv2.imshow("Demo", combined)
            if cv2.waitKey(33) & 0xFF == ord("q"):
                break

    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    summary = detector.get_session_summary()
    print("\n[Session Summary]")
    for name, pct in summary.items():
        print(f"  {name:<35} {pct:.1f}%")
    print(f"\nOutput saved: {args.output}")


if __name__ == "__main__":
    main()
