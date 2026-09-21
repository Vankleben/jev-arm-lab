"""The decision layer.

Two interchangeable implementations behind one interface:

* ``FakeJudge`` — deterministic rules over the same state dict. No API key needed; used
  for development and as the fallback when the API call fails.
* ``JevJudge``  — one TypeSafe call with three typed questions (choice / noul / score).

Everything the judge is allowed to see lives in ``build_state``. Note what is *not* there:
the simulation's ground truth. The state carries the measured gripper width and a single
"a finger feels something" bump bit, so judging whether the grasp is secure is a real
judgment, not a lookup. The ground truth stays in the sim for later calibration.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .skills import TARGET_XY

API_URL = "https://api.typesafe.ai/v1/systemone"
STATE_KEY = "scene"

# --------------------------------------------------------------------------- questions
# One place for every question and threshold — this is the file a human reviews.
QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": (
            "You control a robot arm doing: `instruction`. Look at `scene` and pick the one "
            "skill the arm should execute next. The arm can only move to poses computed by "
            "ordinary code, so choose the intent, not coordinates."
        ),
        "criteria": {
            "approach": "Move to a pose above the object, gripper open, not yet aligned",
            "grasp": "Descend onto the object and close the gripper",
            "lift": "Raise the object straight up off the table",
            "carry": "Move the held object sideways to above the target zone",
            "lower": "Lower the held object down into the target zone",
            "release": "Open the gripper to let the object go",
            "retreat": "Move away from the workspace, task finished",
            "hold": "Do nothing this cycle; the situation is unclear or unsafe",
            "finish": "The task is complete; stop and report",
        },
    },
    "grasp_secure": {
        "type": "noul",
        "instructions": (
            "Is the object currently held securely between the fingers, so that lifting it "
            "would not drop it? Use the gripper readings and recent actions in `scene`."
        ),
        "criteria": {
            "true": "Fingers are closed on the object and it is supported",
            "false": "Gripper is open, or closed on nothing, or the object is slipping",
        },
    },
    "task_progress": {
        "type": "score",
        "instructions": "How far along is the pick-and-place task described by `instruction`?",
        "criteria": [
            "nothing grasped yet",
            "object grasped but still on the table",
            "object lifted and being moved",
            "object placed in the target zone",
        ],
    },
}


@dataclass
class Decision:
    intent: str
    intent_probs: dict = field(default_factory=dict)
    intent_confidence: float = 0.0
    grasp_secure_prob: float = 0.0
    progress_score: float = 0.0
    progress_confidence: float = 0.0
    source: str = "fake"
    error: str = ""

    def as_log(self) -> dict:
        return {
            "intent": self.intent, "intent_probs": self.intent_probs,
            "intent_confidence": round(self.intent_confidence, 3),
            "grasp_secure_prob": round(self.grasp_secure_prob, 3),
            "progress_score": round(self.progress_score, 3),
            "progress_confidence": round(self.progress_confidence, 3),
            "source": self.source, "error": self.error,
        }


# ------------------------------------------------------------------------------- state
def build_state(lab, history: list[dict], instruction: str, cycle: int, sensors=None) -> dict:
    """What the judge is told. With `sensors` the readings and their ages come from the
    sensor layer, so a stale or lying channel is visible in the state itself."""
    if sensors is not None:
        s = sensors.read(lab.d.time)
        obj = s["camera"]["pos_m"]
        clearance = s["camera"]["clearance_m"]
        dist = s["camera"]["distance_to_target_m"]
        cam_age = s["camera"]["age_s"]
        bump = s["tactile"]["bump"]
        width = s["tactile"]["gap_mm"]
        tac_age = s["tactile"]["age_s"]
    else:
        pose = lab.object_pose()
        obj = [round(float(v), 4) for v in pose]
        clearance = round(float(lab.object_clearance()), 4)
        dist = round(float(np.linalg.norm(np.asarray(pose[:2]) - TARGET_XY)), 4)
        touch = lab.pad_contacts()
        bump = bool(touch["left"] or touch["right"])
        width = round(lab.gripper_width() * 1000.0, 1)
        cam_age = tac_age = None

    state = {
        "instruction": instruction,
        "cycle": cycle,
        "tcp": {
            "pos_m": [round(float(v), 4) for v in lab.tcp_pos()],
            "euler_deg": [float(v) for v in lab.tcp_euler()],
        },
        "gripper": {
            "commanded_gap_mm": round(float(lab.commanded_gap()) * 1000.0, 1),
            "measured_gap_mm": width,
            "bump_detected": bump,
            **({"age_s": tac_age} if tac_age is not None else {}),
        },
        "object": {
            "pos_m": obj,
            "height_above_table_m": clearance,
            "xy_distance_to_target_m": dist,
            **({"age_s": cam_age} if cam_age is not None else {}),
        },
        "recent_actions": history[-3:],
    }
    return state


# --------------------------------------------------------------------------- judge API
class FakeJudge:
    """Deterministic policy over the state dict — dev mode and API fallback."""

    name = "fake"

    @staticmethod
    def _features(state: dict) -> dict:
        obj = np.array(state["object"]["pos_m"])
        tcp = np.array(state["tcp"]["pos_m"])
        g = state["gripper"]
        return {
            "d_tcp_obj": float(np.linalg.norm(tcp[:2] - obj[:2])),
            "d_obj_target": float(state["object"]["xy_distance_to_target_m"]),
            "h": float(state["object"]["height_above_table_m"]),
            "tcp_z": float(tcp[2]),
            "bump": bool(g["bump_detected"]),
            "width": float(g["measured_gap_mm"]),
        }

    def decide(self, state: dict) -> Decision:
        f = self._features(state)
        holding = f["bump"] and f["width"] < 80.0      # jaws closed on something
        at_target = f["d_obj_target"] < 0.05
        lifted = f["h"] > 0.02

        if holding:
            if at_target and f["tcp_z"] < 0.20 and not lifted:
                intent = "release"                     # carried to the target and set down
            elif not lifted:
                intent = "lift"                        # held, still resting on the table
            elif not at_target:
                intent = "carry"
            else:
                intent = "lower"
        elif at_target and f["width"] > 70 and f["h"] < 0.03:
            intent = "retreat"                         # placed and released -> back off
        elif f["d_tcp_obj"] > 0.04:
            intent = "approach"
        else:
            intent = "grasp"                           # at the object, jaws not holding it

        grasp_prob = 0.9 if holding else (0.25 if f["bump"] else 0.08)
        progress = 3.0 if (at_target and f["width"] > 70) else (
            2.0 if lifted else (1.0 if holding else 0.0))
        return Decision(
            intent=intent, intent_probs={intent: 1.0}, intent_confidence=1.0,
            grasp_secure_prob=grasp_prob, progress_score=progress, progress_confidence=1.0,
            source="fake",
        )


class JevJudge:
    """One TypeSafe call per cycle: choice intent + noul grasp_secure + score progress."""

    name = "jev"

    def __init__(self, model: str = "jev-latest", timeout: float = 30.0, key_env: str = "TYPESAFE_API_KEY"):
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise RuntimeError(f"{key_env} is not set")
        self.key, self.model, self.timeout = key, model, timeout

    def decide(self, state: dict) -> Decision:
        body = json.dumps({"state": {STATE_KEY: state}, "model": self.model,
                           "questions": QUESTIONS}).encode()
        req = urllib.request.Request(
            API_URL, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            answers = json.loads(r.read().decode())["answers"]

        a_intent = answers["intent"]
        a_grasp = answers["grasp_secure"]
        a_prog = answers["task_progress"]
        return Decision(
            intent=a_intent["choice"],
            intent_probs={k: round(float(v), 4) for k, v in a_intent["probabilities"].items()},
            intent_confidence=float(a_intent["confidence"]),
            grasp_secure_prob=float(a_grasp["noul"]),
            progress_score=float(a_prog["score"]),
            progress_confidence=float(a_prog["confidence"]),
            source="jev",
        )


class FallbackJudge:
    """Live Jev, but never let an API failure stall the arm."""

    def __init__(self, primary, backup=FakeJudge()):
        self.primary, self.backup = primary, backup

    def decide(self, state: dict) -> Decision:
        try:
            return self.primary.decide(state)
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            d = self.backup.decide(state)
            d.source, d.error = "fallback", f"{type(exc).__name__}: {exc}"[:200]
            return d


def make_judge(mode: str):
    """mode: fake | live | auto."""
    if mode == "fake":
        return FakeJudge(), "fake (rules, no API key)"
    if mode == "live":
        return FallbackJudge(JevJudge()), "live Jev via api.typesafe.ai"
    if os.environ.get("TYPESAFE_API_KEY", "").strip():
        return FallbackJudge(JevJudge()), "auto -> live Jev"
    return FakeJudge(), "auto -> fake (TYPESAFE_API_KEY not set)"
