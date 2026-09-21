"""Stressors: deliberate perturbations used to find where the judge breaks.

Two rules keep this honest:

1. **The world stays real.** Except for the grasp offset (a real calibration error) and the
   shove (a real external push), these stressors degrade *only what the judge is told* by
   misconfiguring the sensor layer in `sensors.py`. The simulation's ground truth keeps
   coming from the real world, so scoring stays meaningful.
2. **Each stressor maps to a story you can name** (dead tactile, stuck tactile, slow camera,
   camera unplugged, noisy measurement, miscalibration, someone bumping the table).

Known limit: the grasp *hold* is a documented kinematic carry (see README), so a stressor
like "the object is slippery" cannot be expressed — friction plays no part in transport
here. That is a limitation of this lab, not a property of the judge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Stress:
    name: str = "none"
    # --- real physics -------------------------------------------------------
    grasp_offset_m: float = 0.0      # attempt the grasp a few cm off centre (miscalibration)
    shove_velocity: float = 0.0      # external impulse on the object
    shove_cycle: int = -1
    # --- sensor configuration (the judge's view of the world) ----------------
    camera_period_s: float | None = None   # None keeps the default 0.25 s
    camera_frozen: bool = False            # camera stops publishing entirely
    tactile_frozen: bool = False
    tactile_lying: str | None = None       # "always_true" | "always_false"
    noise_obj_mm: float = 0.0
    noise_gap_mm: float = 0.0

    def configure(self, sensors) -> None:
        """Apply this stressor to a SensorSuite."""
        if self.camera_period_s is not None:
            sensors.camera.period_s = self.camera_period_s
        sensors.camera.frozen = self.camera_frozen
        sensors.tactile.frozen = self.tactile_frozen
        sensors.tactile.lying = self.tactile_lying
        sensors.camera.noise_mm = self.noise_obj_mm
        sensors.tactile.noise_mm = self.noise_gap_mm

    def grasp_target(self, xy, rng):
        """Where the grasp is actually attempted (offsets a real calibration error)."""
        if not self.grasp_offset_m:
            return xy
        angle = float(rng.uniform(0, 2 * np.pi))
        return np.asarray(xy, dtype=float) + self.grasp_offset_m * np.array([np.cos(angle), np.sin(angle)])

    def shove_here(self, cycle: int) -> bool:
        return self.shove_cycle >= 0 and cycle == self.shove_cycle and self.shove_velocity > 0


# Named stressors used by tools/stress_batch.py. Each one is a sentence you could say
# about a real robot cell.
STRESSORS: dict[str, Stress] = {
    "none": Stress(name="none"),
    "grasp_off": Stress(name="grasp_off", grasp_offset_m=0.022),                 # 标定偏 22 mm
    "tactile_dead": Stress(name="tactile_dead", tactile_lying="always_false"),   # 触觉永远说"没碰到"
    "tactile_stuck": Stress(name="tactile_stuck", tactile_lying="always_true"),  # 触觉永远说"碰到了"
    "occlusion": Stress(name="occlusion", camera_period_s=4.0),                  # 相机 4 秒才出一帧
    "camera_frozen": Stress(name="camera_frozen", camera_frozen=True),           # 相机拔线，永远第一帧
    "noise": Stress(name="noise", noise_obj_mm=10.0, noise_gap_mm=4.0),          # 位置抖 10 mm
    "shove": Stress(name="shove", shove_velocity=0.9, shove_cycle=1),            # 第 1 轮被外力推一下
}


def get(name: str) -> Stress:
    """拼错的名字直接报错，而不是静默当成无故障跑——那会让"失败地图"整个作废。"""
    if name not in STRESSORS:
        raise ValueError(f"unknown stressor {name!r}; valid names: {', '.join(STRESSORS)}")
    return STRESSORS[name]
