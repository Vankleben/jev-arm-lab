"""Sensor layer: where the numbers the judge sees actually come from.

Before this module, `build_state` read the simulation directly, so a statement like "the
camera reading is 40 seconds old" could not even be expressed. Now every channel publishes
on its own clock and the state carries the **age of each reading** — which is what lets the
code detect a sensor that quietly stopped updating.

That matters because of what the failure map showed:

* a channel that **lies** (stuck tactile saying "no contact") is survivable — the other
  channels contradict it and the judge routes around it (100% vs 0% for a hand-written rule);
* a channel that **stops updating** is not: nothing contradicts it, no error is raised, and
  the system keeps acting on a frozen frame. In the frozen-camera run the task-progress
  estimate silently capped at 2.65 of 3, so the robot never knows it is finished.

Real robots get this from message timestamps (ROS `header.stamp`, heartbeat topics). Here it
is modelled explicitly so the freshness gate in `main.py` has something to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skills import TARGET_XY


@dataclass
class Channel:
    name: str
    period_s: float               # how often the sensor republishes (sim seconds)
    frozen: bool = False          # stop updating after the first publish (cable pulled)
    lying: str | None = None      # "always_true" | "always_false": a stuck bit
    noise_mm: float = 0.0
    bias_mm: float = 0.0
    t_last: float = -1e9
    value: dict = field(default_factory=dict)

    @property
    def age(self) -> float:
        return self.t_last


class SensorSuite:
    """Two channels: a camera publishing the object, and a gripper/tactile channel."""

    def __init__(self, camera_period_s: float = 0.25, tactile_period_s: float = 0.05,
                 rng: np.random.Generator | None = None):
        self.camera = Channel("camera", camera_period_s)
        self.tactile = Channel("tactile", tactile_period_s)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    # ------------------------------------------------------------------ publishing
    def update(self, lab, now: float) -> None:
        """Called after every physics step; each channel publishes on its own clock."""
        for ch in (self.camera, self.tactile):
            if ch.frozen:
                continue
            if now - ch.t_last >= ch.period_s or ch.t_last < 0:
                ch.value = self._sample(ch, lab)
                ch.t_last = now

    def _sample(self, ch: Channel, lab) -> dict:
        if ch.name == "camera":
            obj = lab.object_pose()
            pos = obj.copy()
            if ch.noise_mm:
                pos[:2] += self.rng.normal(0, ch.noise_mm / 1000.0, size=2)
            if ch.bias_mm:
                pos[0] += ch.bias_mm / 1000.0
            return {"pos_m": [round(float(v), 4) for v in pos],
                    "clearance_m": round(float(lab.object_clearance()), 4),
                    "distance_to_target_m": round(float(np.linalg.norm(pos[:2] - TARGET_XY)), 4)}
        touch = lab.pad_contacts()
        bump = bool(touch["left"] or touch["right"])
        if ch.lying == "always_true":
            bump = True
        elif ch.lying == "always_false":
            bump = False
        gap = lab.gripper_width() * 1000.0
        if ch.noise_mm:
            gap += float(self.rng.normal(0, ch.noise_mm))
        if ch.bias_mm:
            gap += ch.bias_mm
        return {"bump": bump, "gap_mm": round(gap, 1)}

    # ---------------------------------------------------------------------- reading
    def read(self, now: float) -> dict:
        """What the judge gets, plus how old each reading is."""
        out = {}
        for ch in (self.camera, self.tactile):
            age = float("inf") if ch.t_last < 0 else max(0.0, now - ch.t_last)
            out[ch.name] = {**ch.value, "age_s": round(age, 2)}
        return out

    def ages(self, now: float) -> dict:
        r = self.read(now)
        return {k: r[k]["age_s"] for k in r}

    def prime(self, lab, now: float) -> None:
        """Publish once immediately (after a reset or a scene change).

        Forced even on a frozen channel: a camera that dies has still sent *something*
        before it died, and the gate's whole job is to notice that the frame never changes
        again. Without this, a frozen channel would hand the judge an empty state instead.
        """
        was_frozen = [(ch, ch.frozen) for ch in (self.camera, self.tactile)]
        for ch, _ in was_frozen:
            ch.frozen = False
            ch.t_last = -1e9
        self.update(lab, now)
        for ch, was in was_frozen:
            ch.frozen = was
