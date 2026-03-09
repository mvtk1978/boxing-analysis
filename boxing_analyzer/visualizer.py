"""
Real-time boxing analysis visualizer.
Draws pose skeleton, highlights faults, shows coaching panel.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from .landmarks import LM, get_point
from .pattern_detector import FrameAnalysis, Fault


# MediaPipe skeleton connections (subset of important ones)
POSE_CONNECTIONS = [
    # Head
    (LM.LEFT_EAR, LM.LEFT_EYE), (LM.RIGHT_EAR, LM.RIGHT_EYE),
    (LM.LEFT_EYE, LM.NOSE), (LM.RIGHT_EYE, LM.NOSE),
    # Torso
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    # Arms
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    # Legs
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_ANKLE, LM.LEFT_HEEL),
    (LM.RIGHT_ANKLE, LM.RIGHT_HEEL),
    (LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
    (LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
]

# Color scheme
COLORS = {
    "skeleton_normal": (100, 220, 100),    # green
    "skeleton_fault": (0, 0, 255),          # red
    "joint_normal": (200, 255, 200),
    "joint_fault": (50, 50, 255),
    "joint_critical": (0, 0, 200),
    "panel_bg": (15, 15, 15),
    "panel_header": (0, 200, 255),         # cyan
    "text_normal": (220, 220, 220),
    "text_warning": (0, 165, 255),          # orange
    "text_critical": (50, 50, 255),         # red
    "text_good": (50, 200, 50),
    "action_jab": (255, 255, 0),            # yellow
    "action_cross": (0, 255, 255),
    "stance": (200, 200, 50),
    "highlight_ring": (0, 80, 255),
    "highlight_ring2": (0, 40, 180),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


class BoxingVisualizer:

    def __init__(self, panel_width: int = 380):
        self.panel_width = panel_width
        self._pulse = 0  # animation counter

    def draw(self, frame: np.ndarray, landmarks, analysis: FrameAnalysis,
             session_summary: dict, frame_idx: int = 0) -> np.ndarray:
        """
        Compose the full annotated frame:
          - Pose skeleton with fault highlights
          - Animated fault rings on affected joints
          - Side panel with live coaching tips and metrics
        """
        h, w = frame.shape[:2]
        self._pulse = (self._pulse + 3) % 360

        # Get affected landmark indices from active faults
        fault_lm_set = set()
        for f in analysis.faults:
            fault_lm_set.update(f.affected_landmarks)

        # ----------------------------------------------------------------
        # Draw skeleton on frame
        # ----------------------------------------------------------------
        overlay = frame.copy()

        # Connections
        for a, b in POSE_CONNECTIONS:
            try:
                pa = get_point(landmarks, a, w, h).astype(int)
                pb = get_point(landmarks, b, w, h).astype(int)
                is_fault = (a in fault_lm_set) or (b in fault_lm_set)
                color = COLORS["skeleton_fault"] if is_fault else COLORS["skeleton_normal"]
                thickness = 3 if is_fault else 2
                cv2.line(overlay, tuple(pa), tuple(pb), color, thickness, cv2.LINE_AA)
            except Exception:
                pass

        # Joints
        for idx in range(33):
            try:
                pt = get_point(landmarks, idx, w, h).astype(int)
                is_fault = idx in fault_lm_set
                is_critical = any(
                    idx in f.affected_landmarks and f.severity == "critical"
                    for f in analysis.faults
                )
                if is_critical:
                    color = COLORS["joint_critical"]
                    radius = 8
                elif is_fault:
                    color = COLORS["joint_fault"]
                    radius = 7
                else:
                    color = COLORS["joint_normal"]
                    radius = 5
                cv2.circle(overlay, tuple(pt), radius, color, -1, cv2.LINE_AA)
                cv2.circle(overlay, tuple(pt), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
            except Exception:
                pass

        # Animated pulsing rings on critical fault joints
        pulse_r = int(14 + 6 * abs(np.sin(np.radians(self._pulse))))
        for f in analysis.faults:
            if f.severity == "critical":
                for lm_idx in f.affected_landmarks[:2]:
                    try:
                        pt = get_point(landmarks, lm_idx, w, h).astype(int)
                        alpha = 0.5 + 0.4 * abs(np.sin(np.radians(self._pulse)))
                        cv2.circle(overlay, tuple(pt), pulse_r, COLORS["highlight_ring"], 2, cv2.LINE_AA)
                        cv2.circle(overlay, tuple(pt), pulse_r + 4, COLORS["highlight_ring2"], 1, cv2.LINE_AA)
                    except Exception:
                        pass

        # Blend overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # ----------------------------------------------------------------
        # Fault label bubbles on body
        # ----------------------------------------------------------------
        shown_labels = set()
        for f in analysis.faults[:4]:  # max 4 labels on body
            if f.name in shown_labels:
                continue
            shown_labels.add(f.name)
            try:
                lm_idx = f.affected_landmarks[0]
                pt = get_point(landmarks, lm_idx, w, h).astype(int)
                label = f.name
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
                lx, ly = pt[0] + 10, pt[1] - 10
                lx = min(lx, w - tw - 6)
                ly = max(ly, th + 6)
                bg_color = (30, 30, 200) if f.severity == "critical" else (30, 100, 200)
                cv2.rectangle(frame, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), bg_color, -1)
                cv2.rectangle(frame, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), (255, 255, 255), 1)
                cv2.putText(frame, label, (lx, ly), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                # Arrow from label to joint
                cv2.arrowedLine(frame, (lx, ly - th // 2), tuple(pt), bg_color, 1, cv2.LINE_AA, tipLength=0.3)
            except Exception:
                pass

        # ----------------------------------------------------------------
        # Side coaching panel
        # ----------------------------------------------------------------
        panel = self._build_panel(h, analysis, session_summary)
        combined = np.hstack([frame, panel])

        # Action label top center
        self._draw_action_label(combined, analysis.action, analysis.stance, w, h)

        return combined

    def _build_panel(self, h: int, analysis: FrameAnalysis, session_summary: dict) -> np.ndarray:
        pw = self.panel_width
        panel = np.zeros((h, pw, 3), dtype=np.uint8)
        panel[:] = COLORS["panel_bg"]

        # Vertical divider
        cv2.line(panel, (0, 0), (0, h), (50, 50, 50), 2)

        y = 20

        # === HEADER ===
        cv2.putText(panel, "BOXING ANALYSIS", (12, y), FONT, 0.65, COLORS["panel_header"], 2, cv2.LINE_AA)
        y += 22
        cv2.line(panel, (10, y), (pw - 10, y), (40, 40, 40), 1)
        y += 16

        # === STANCE ===
        stance_text = f"Stance: {analysis.stance.upper()}"
        cv2.putText(panel, stance_text, (12, y), FONT, 0.5, COLORS["stance"], 1, cv2.LINE_AA)
        y += 24
        cv2.line(panel, (10, y), (pw - 10, y), (30, 30, 30), 1)
        y += 14

        # === LIVE FAULTS ===
        cv2.putText(panel, "LIVE FAULTS", (12, y), FONT, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
        y += 22

        if not analysis.faults:
            cv2.putText(panel, "  No faults detected", (12, y), FONT, 0.47, COLORS["text_good"], 1, cv2.LINE_AA)
            y += 20
        else:
            for f in analysis.faults[:6]:
                icon = "!!" if f.severity == "critical" else "! "
                color = COLORS["text_critical"] if f.severity == "critical" else COLORS["text_warning"]
                label_line = f"{icon} {f.name}"
                cv2.putText(panel, label_line, (12, y), FONT, 0.48, color, 1, cv2.LINE_AA)
                y += 18
                # Wrap description
                words = f.description.split()
                line = "   "
                for word in words:
                    test = line + word + " "
                    if len(test) > 42:
                        cv2.putText(panel, line.rstrip(), (12, y), FONT, 0.38,
                                    (160, 160, 160), 1, cv2.LINE_AA)
                        y += 15
                        line = "   " + word + " "
                    else:
                        line = test
                if line.strip():
                    cv2.putText(panel, line.rstrip(), (12, y), FONT, 0.38,
                                (160, 160, 160), 1, cv2.LINE_AA)
                    y += 16
                y += 4

        y += 4
        cv2.line(panel, (10, y), (pw - 10, y), (30, 30, 30), 1)
        y += 14

        # === METRICS BAR CHART ===
        cv2.putText(panel, "METRICS", (12, y), FONT, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
        y += 20

        metrics_to_show = [
            ("Lead Guard", analysis.metrics.get("lead_guard_ratio", 0), -0.2, 0.25, 0.0),
            ("Rear Guard",  analysis.metrics.get("rear_guard_ratio", 0), -0.2, 0.25, 0.0),
            ("Chin Level",  -analysis.metrics.get("chin_elevation", 0), -0.25, 0.1, 0.1),
            ("Stance W",    analysis.metrics.get("stance_width_ratio", 0) - 1.5, -0.6, 0.7, 0.0),
        ]

        bar_w = pw - 100
        for label, val, vmin, vmax, good_min in metrics_to_show:
            norm = (val - vmin) / (vmax - vmin + 1e-9)
            norm = max(0.0, min(1.0, norm))
            good_norm = max(0.0, min(1.0, (good_min - vmin) / (vmax - vmin + 1e-9)))

            cv2.putText(panel, label, (12, y), FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            bx, by = 100, y - 10
            # Bar background
            cv2.rectangle(panel, (bx, by), (bx + bar_w, by + 10), (40, 40, 40), -1)
            # Good zone line
            gx = bx + int(good_norm * bar_w)
            cv2.line(panel, (gx, by - 2), (gx, by + 12), (0, 200, 0), 1)
            # Value bar
            fill_w = int(norm * bar_w)
            bar_color = (50, 200, 50) if abs(norm - good_norm) < 0.25 else (50, 80, 220)
            cv2.rectangle(panel, (bx, by), (bx + fill_w, by + 10), bar_color, -1)
            y += 22

        y += 6
        cv2.line(panel, (10, y), (pw - 10, y), (30, 30, 30), 1)
        y += 14

        # === SESSION SUMMARY ===
        cv2.putText(panel, "SESSION FAULTS %", (12, y), FONT, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
        y += 22

        for fault_name, pct in list(session_summary.items())[:6]:
            bar_fill = int((pct / 100) * (pw - 40))
            bar_color = (50, 50, 180) if pct > 50 else (50, 130, 50)
            cv2.rectangle(panel, (12, y - 9), (12 + bar_fill, y + 2), bar_color, -1)
            short = fault_name[:28]
            cv2.putText(panel, f"{short} {pct:.0f}%", (14, y), FONT, 0.37,
                        (210, 210, 210), 1, cv2.LINE_AA)
            y += 18

        if not session_summary:
            cv2.putText(panel, "  Analyzing...", (12, y), FONT, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

        return panel

    def _draw_action_label(self, frame: np.ndarray, action: str, stance: str, w: int, h: int):
        if action == "neutral":
            return
        label_map = {
            "jab": "JAB",
            "cross": "CROSS",
            "hook": "HOOK",
        }
        label = label_map.get(action, action.upper())
        color_map = {
            "jab": COLORS["action_jab"],
            "cross": COLORS["action_cross"],
        }
        color = color_map.get(action, (200, 200, 200))
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.2, 3)
        tx = (w - tw) // 2
        ty = 60
        cv2.putText(frame, label, (tx + 2, ty + 2), FONT, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), FONT, 1.2, color, 3, cv2.LINE_AA)
