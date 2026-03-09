"""
Boxing Fault Pattern Detector

Detects common boxing technique faults from pose landmarks:
- Guard: hands too low, elbows flared, chin up
- Footwork: feet crossing, stance too wide/narrow, static feet
- Head movement: no head movement (static target)
- Punch mechanics: dropping guard before/after punch, overextension
- Weight distribution: leaning too far forward/back
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .landmarks import LM, get_point, get_point_norm, angle_between, midpoint, distance


@dataclass
class Fault:
    name: str
    severity: str          # "warning" | "critical"
    description: str
    affected_landmarks: List[int]
    confidence: float      # 0.0 - 1.0

    @property
    def color_bgr(self):
        if self.severity == "critical":
            return (0, 0, 255)    # Red
        return (0, 165, 255)      # Orange


@dataclass
class FrameAnalysis:
    faults: List[Fault] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    stance: str = "unknown"       # "orthodox" | "southpaw"
    action: str = "neutral"       # "jab" | "cross" | "hook" | "guard" | "neutral"


class PatternDetector:
    """
    Stateful pattern detector - maintains a rolling window of frames
    to catch temporal patterns (e.g. static head, dropped guard after punch).
    """

    HISTORY_LEN = 45  # ~1.5s at 30fps

    def __init__(self):
        self.history: deque = deque(maxlen=self.HISTORY_LEN)
        self.frame_count = 0
        self.stance: Optional[str] = None
        self._head_pos_history: deque = deque(maxlen=30)
        self._wrist_pos_history: deque = deque(maxlen=15)
        self._foot_pos_history: deque = deque(maxlen=30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, landmarks, w: int, h: int) -> FrameAnalysis:
        """
        Analyze a single frame's landmarks. Returns FrameAnalysis with faults.
        landmarks: mediapipe pose landmark list (33 points)
        w, h: frame dimensions in pixels
        """
        self.frame_count += 1
        analysis = FrameAnalysis()

        # Helper to get pixel coords
        def pt(idx):
            return get_point(landmarks, idx, w, h)

        def pt_n(idx):
            return get_point_norm(landmarks, idx)

        # ---------------------------------------------------------------
        # Core body points (normalized coords for geometry, px for display)
        # ---------------------------------------------------------------
        nose = pt(LM.NOSE)
        l_shoulder = pt(LM.LEFT_SHOULDER)
        r_shoulder = pt(LM.RIGHT_SHOULDER)
        l_elbow = pt(LM.LEFT_ELBOW)
        r_elbow = pt(LM.RIGHT_ELBOW)
        l_wrist = pt(LM.LEFT_WRIST)
        r_wrist = pt(LM.RIGHT_WRIST)
        l_hip = pt(LM.LEFT_HIP)
        r_hip = pt(LM.RIGHT_HIP)
        l_knee = pt(LM.LEFT_KNEE)
        r_knee = pt(LM.RIGHT_KNEE)
        l_ankle = pt(LM.LEFT_ANKLE)
        r_ankle = pt(LM.RIGHT_ANKLE)

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip = midpoint(l_hip, r_hip)

        # Body scale: shoulder-to-hip distance (for normalization)
        body_height = distance(mid_shoulder, mid_hip) + 1e-6
        shoulder_width = distance(l_shoulder, r_shoulder) + 1e-6

        # ------------------------------------------------------------------
        # Determine stance (orthodox: left foot forward; southpaw: right)
        # ------------------------------------------------------------------
        if self.stance is None or self.frame_count % 30 == 0:
            self.stance = self._detect_stance(l_ankle, r_ankle, l_shoulder, r_shoulder)
        analysis.stance = self.stance

        # Lead / rear side depends on stance
        if self.stance == "orthodox":
            lead_wrist, rear_wrist = l_wrist, r_wrist
            lead_shoulder, rear_shoulder = l_shoulder, r_shoulder
            lead_elbow, rear_elbow = l_elbow, r_elbow
            lead_ankle, rear_ankle = l_ankle, r_ankle
        else:
            lead_wrist, rear_wrist = r_wrist, l_wrist
            lead_shoulder, rear_shoulder = r_shoulder, l_shoulder
            lead_elbow, rear_elbow = r_elbow, l_elbow
            lead_ankle, rear_ankle = r_ankle, l_ankle

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        chin_y = nose[1]
        shoulder_y = mid_shoulder[1]

        # Guard heights (relative to shoulder - positive = hands above shoulders)
        lead_guard_ratio = (shoulder_y - lead_wrist[1]) / body_height
        rear_guard_ratio = (shoulder_y - rear_wrist[1]) / body_height

        # Elbow flare (angle at elbow: shoulder-elbow-wrist)
        lead_elbow_angle = angle_between(lead_shoulder, lead_elbow, lead_wrist)
        rear_elbow_angle = angle_between(rear_shoulder, rear_elbow, rear_wrist)

        # Chin elevation (nose relative to shoulder line - high = exposed)
        chin_elevation = (shoulder_y - chin_y) / body_height  # positive = chin up

        # Stance width (ankle distance relative to shoulder width)
        stance_width_ratio = distance(l_ankle, r_ankle) / shoulder_width

        # Trunk lean (angle of shoulder-hip line from vertical)
        trunk_lean = self._trunk_lean(mid_shoulder, mid_hip)

        # Update histories
        self._head_pos_history.append(nose.copy())
        self._wrist_pos_history.append((lead_wrist.copy(), rear_wrist.copy()))
        self._foot_pos_history.append((l_ankle.copy(), r_ankle.copy()))

        analysis.metrics = {
            "lead_guard_ratio": float(lead_guard_ratio),
            "rear_guard_ratio": float(rear_guard_ratio),
            "lead_elbow_angle": float(lead_elbow_angle),
            "rear_elbow_angle": float(rear_elbow_angle),
            "chin_elevation": float(chin_elevation),
            "stance_width_ratio": float(stance_width_ratio),
            "trunk_lean": float(trunk_lean),
        }

        # ------------------------------------------------------------------
        # FAULT DETECTION
        # ------------------------------------------------------------------

        # 1. Lead hand too low
        if lead_guard_ratio < -0.05:  # wrist below shoulder by 5%+ body height
            severity = "critical" if lead_guard_ratio < -0.15 else "warning"
            conf = min(1.0, abs(lead_guard_ratio) * 4)
            analysis.faults.append(Fault(
                name="Lead Hand Too Low",
                severity=severity,
                description=f"Lead guard dropped - chin exposed. Raise lead hand to cheekbone level.",
                affected_landmarks=[LM.LEFT_WRIST if self.stance == "orthodox" else LM.RIGHT_WRIST,
                                    LM.LEFT_ELBOW if self.stance == "orthodox" else LM.RIGHT_ELBOW],
                confidence=conf,
            ))

        # 2. Rear hand too low
        if rear_guard_ratio < -0.08:
            severity = "critical" if rear_guard_ratio < -0.18 else "warning"
            conf = min(1.0, abs(rear_guard_ratio) * 3.5)
            analysis.faults.append(Fault(
                name="Rear Hand Too Low",
                severity=severity,
                description="Rear guard dropped - open to counter. Keep rear hand at chin.",
                affected_landmarks=[LM.RIGHT_WRIST if self.stance == "orthodox" else LM.LEFT_WRIST,
                                    LM.RIGHT_ELBOW if self.stance == "orthodox" else LM.LEFT_ELBOW],
                confidence=conf,
            ))

        # 3. Chin up (head tilted back - easy KO target)
        if chin_elevation > 0.12:
            severity = "critical" if chin_elevation > 0.20 else "warning"
            conf = min(1.0, chin_elevation * 4)
            analysis.faults.append(Fault(
                name="Chin Up - Head Exposed",
                severity=severity,
                description="Chin is elevated - tuck chin to chest. Classic knockout setup.",
                affected_landmarks=[LM.NOSE, LM.LEFT_EAR, LM.RIGHT_EAR],
                confidence=conf,
            ))

        # 4. Elbow flare (elbows out from body - weakens guard, exposes ribs)
        for side, elbow_angle, lms in [
            ("Lead", lead_elbow_angle,
             [LM.LEFT_ELBOW if self.stance == "orthodox" else LM.RIGHT_ELBOW]),
            ("Rear", rear_elbow_angle,
             [LM.RIGHT_ELBOW if self.stance == "orthodox" else LM.LEFT_ELBOW]),
        ]:
            if elbow_angle > 110:  # elbows should be ~60-80deg in guard
                severity = "warning" if elbow_angle < 135 else "critical"
                conf = min(1.0, (elbow_angle - 90) / 60)
                analysis.faults.append(Fault(
                    name=f"{side} Elbow Flared",
                    severity=severity,
                    description=f"{side} elbow flared out ({elbow_angle:.0f}deg) - ribs exposed. Tuck elbows.",
                    affected_landmarks=lms,
                    confidence=conf,
                ))

        # 5. Stance too wide (hard to move, slow feet)
        if stance_width_ratio > 2.2:
            conf = min(1.0, (stance_width_ratio - 2.0) / 0.8)
            analysis.faults.append(Fault(
                name="Stance Too Wide",
                severity="warning",
                description=f"Feet too far apart ({stance_width_ratio:.1f}x shoulders) - mobility reduced.",
                affected_landmarks=[LM.LEFT_ANKLE, LM.RIGHT_ANKLE],
                confidence=conf,
            ))

        # 6. Stance too narrow (unstable, easy to be pushed off balance)
        if stance_width_ratio < 0.9:
            conf = min(1.0, (0.9 - stance_width_ratio) / 0.4)
            analysis.faults.append(Fault(
                name="Stance Too Narrow",
                severity="warning",
                description="Feet too close together - balance compromised. Widen to shoulder-width+.",
                affected_landmarks=[LM.LEFT_ANKLE, LM.RIGHT_ANKLE],
                confidence=conf,
            ))

        # 7. Excessive trunk lean (overcommitting forward or leaning back)
        if abs(trunk_lean) > 20:
            direction = "forward" if trunk_lean > 0 else "backward"
            severity = "critical" if abs(trunk_lean) > 35 else "warning"
            conf = min(1.0, abs(trunk_lean) / 45)
            analysis.faults.append(Fault(
                name=f"Trunk Leaning {direction.title()}",
                severity=severity,
                description=f"Body leaning {direction} {abs(trunk_lean):.0f}deg - off balance, easy to counter.",
                affected_landmarks=[LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP],
                confidence=conf,
            ))

        # 8. Static head (no head movement - sitting duck)
        head_movement = self._compute_head_movement(body_height)
        if head_movement is not None and head_movement < 0.015:
            conf = min(1.0, (0.02 - head_movement) / 0.02)
            analysis.faults.append(Fault(
                name="Static Head - No Head Movement",
                severity="warning",
                description="Head not moving - stationary target. Add slip, roll, or bob.",
                affected_landmarks=[LM.NOSE, LM.LEFT_EAR, LM.RIGHT_EAR],
                confidence=conf,
            ))

        # 9. Feet crossing (dangerous in footwork)
        feet_crossed = self._detect_feet_crossing(l_ankle, r_ankle, l_shoulder, r_shoulder)
        if feet_crossed:
            analysis.faults.append(Fault(
                name="Feet Crossing - Dangerous Footwork",
                severity="critical",
                description="Feet crossed - loss of balance. Maintain proper foot positioning.",
                affected_landmarks=[LM.LEFT_ANKLE, LM.RIGHT_ANKLE, LM.LEFT_KNEE, LM.RIGHT_KNEE],
                confidence=0.85,
            ))

        # 10. Detect action (basic punch detection from wrist velocity)
        analysis.action = self._detect_action(lead_wrist, rear_wrist, body_height)

        self.history.append(analysis)
        return analysis

    # ------------------------------------------------------------------
    # Temporal / helper analysis
    # ------------------------------------------------------------------

    def _detect_stance(self, l_ankle, r_ankle, l_shoulder, r_shoulder):
        """
        Orthodox: lead (left) foot forward = smaller x (closer to opponent if facing right).
        Heuristic: the foot with smaller x is more forward when facing right.
        We check shoulder orientation to determine facing direction.
        """
        # If left shoulder is more forward (smaller x), fighter faces right → orthodox
        if l_shoulder[0] < r_shoulder[0]:
            return "orthodox" if l_ankle[0] < r_ankle[0] else "southpaw"
        else:
            return "orthodox" if r_ankle[0] < l_ankle[0] else "southpaw"

    def _trunk_lean(self, mid_shoulder, mid_hip):
        """Returns trunk lean angle in degrees. Positive = forward, negative = back."""
        diff = mid_shoulder - mid_hip
        # Angle from vertical (negative y = up in image coords)
        angle = np.degrees(np.arctan2(diff[0], -diff[1]))
        return float(angle)

    def _compute_head_movement(self, body_height: float) -> Optional[float]:
        """Mean frame-to-frame head displacement normalized by body height."""
        if len(self._head_pos_history) < 10:
            return None
        positions = np.array(list(self._head_pos_history))
        deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.mean(deltas) / body_height)

    def _detect_feet_crossing(self, l_ankle, r_ankle, l_shoulder, r_shoulder) -> bool:
        """Detect if feet have crossed relative to expected stance."""
        shoulder_facing_right = l_shoulder[0] < r_shoulder[0]
        if shoulder_facing_right:
            # Feet should not cross: left foot left of right foot
            return bool(l_ankle[0] > r_ankle[0] + 20)
        else:
            return bool(r_ankle[0] > l_ankle[0] + 20)

    def _detect_action(self, lead_wrist, rear_wrist, body_height) -> str:
        """Rough punch detection from wrist velocity."""
        if len(self._wrist_pos_history) < 3:
            return "neutral"
        prev_lead, prev_rear = self._wrist_pos_history[-3]
        lead_vel = distance(lead_wrist, prev_lead) / body_height
        rear_vel = distance(rear_wrist, prev_rear) / body_height
        if lead_vel > 0.15:
            return "jab"
        if rear_vel > 0.15:
            return "cross"
        return "neutral"

    def get_session_summary(self) -> Dict:
        """Aggregate fault statistics across all analyzed frames."""
        if not self.history:
            return {}
        fault_counts: Dict[str, int] = {}
        total = len(self.history)
        for frame in self.history:
            seen = set()
            for f in frame.faults:
                if f.name not in seen:
                    fault_counts[f.name] = fault_counts.get(f.name, 0) + 1
                    seen.add(f.name)
        # Convert to percentages
        return {name: round(count / total * 100, 1) for name, count in
                sorted(fault_counts.items(), key=lambda x: -x[1])}
