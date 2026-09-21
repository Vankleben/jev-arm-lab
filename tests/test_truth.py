"""真值一致性：grasp_flags 的 in_target 必须和技能层的 TARGET_XY 同源。

sim.py 曾经把 [0.36, -0.13] 硬编码第二份——将来有人只改 skills.TARGET_XY 调目标，
真值就会静默失配，所有完成率统计都在说谎。本测试把"同源"和 in_target 的几何
定义（半径 5 cm、必须落地）一起锁住。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)          # 模型路径是相对仓库根的

import mujoco           # noqa: E402

from jev_arm.sim import ArmLab          # noqa: E402
from jev_arm.skills import TARGET_XY    # noqa: E402

CUBE_REST_Z = 0.15


class TestTargetTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = ArmLab()

    def set_cube(self, x: float, y: float, z: float = CUBE_REST_Z) -> None:
        self.lab.d.qpos[self.lab.cube_qadr:self.lab.cube_qadr + 3] = [x, y, z]
        mujoco.mj_forward(self.lab.m, self.lab.d)

    def test_cube_on_target_counts(self):
        self.set_cube(float(TARGET_XY[0]), float(TARGET_XY[1]))
        self.assertTrue(self.lab.grasp_flags()["in_target"])

    def test_inside_radius_counts(self):
        self.set_cube(float(TARGET_XY[0]) + 0.04, float(TARGET_XY[1]) - 0.03)  # 5 cm 内
        self.assertTrue(self.lab.grasp_flags()["in_target"])

    def test_outside_radius_does_not(self):
        self.set_cube(float(TARGET_XY[0]) + 0.06, float(TARGET_XY[1]))         # 6 cm 外
        self.assertFalse(self.lab.grasp_flags()["in_target"])

    def test_hovering_above_target_does_not(self):
        self.set_cube(float(TARGET_XY[0]), float(TARGET_XY[1]), z=CUBE_REST_Z + 0.05)
        self.assertFalse(self.lab.grasp_flags()["in_target"])

    def test_truth_and_skills_agree_on_target(self):
        # 真值函数里不允许再有独立的目标坐标——只能来自 skills.TARGET_XY
        import inspect
        import jev_arm.sim as sim
        src = inspect.getsource(sim.ArmLab.grasp_flags)
        self.assertNotIn("0.36", src, "grasp_flags 里又硬编码目标坐标了")
        self.assertNotIn("-0.13", src, "grasp_flags 里又硬编码目标坐标了")


if __name__ == "__main__":
    unittest.main()
