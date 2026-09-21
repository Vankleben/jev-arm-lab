"""物理型难例的单测：难例必须真的改到模型，而不是只改了个名字。

这些断言来自 tools/diagnose_slip.py 的实测标定：μ=0.05 会滑掉、0.08 只滑不脱、
0.5 kg 会滑掉——所以模型侧的值必须精确落地，且不能被下一个场景残留。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import mujoco           # noqa: E402

from jev_arm.sim import ArmLab          # noqa: E402
from jev_arm.stress import STRESSORS    # noqa: E402


class TestStressPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = ArmLab()

    def test_none_changes_nothing(self):
        before = [float(self.lab.m.geom_friction[self.lab.pads[s][0]][0]) for s in ("left", "right")]
        STRESSORS["none"].apply_physics(self.lab)
        after = [float(self.lab.m.geom_friction[self.lab.pads[s][0]][0]) for s in ("left", "right")]
        self.assertEqual(before, after)

    def test_low_friction_sets_pads_and_priority(self):
        STRESSORS["low_friction"].apply_physics(self.lab)
        for side in ("left", "right"):
            gid = self.lab.pads[side][0]
            self.assertAlmostEqual(float(self.lab.m.geom_friction[gid][0]), 0.05)
            # 优先级必须抬起来，否则接触摩擦会取方块那侧的较大值，难例被静默忽略
            self.assertEqual(int(self.lab.m.geom_priority[gid]), 1)

    def test_heavy_object_scales_mass_and_inertia(self):
        STRESSORS["heavy_object"].apply_physics(self.lab)
        body = self.lab.m.geom_bodyid[self.lab.geom_cube]
        self.assertAlmostEqual(float(self.lab.m.body_mass[body]), 1.5)
        # 6 cm 立方体：I = m/12 * (a^2 + b^2) = m * 6e-4
        for i in range(3):
            self.assertAlmostEqual(float(self.lab.m.body_inertia[body][i]), 1.5 * 6e-4, places=8)

    def test_heavy_object_actually_makes_it_drop(self):
        """实测过的物理结论：1.5 kg 时 8 N/指的夹持挡不住，方块留在桌上。"""
        lab = ArmLab()
        STRESSORS["heavy_object"].apply_physics(lab)
        start = lab.object_pose().copy()
        lab.move_tcp([start[0], start[1], 0.26])
        lab.move_tcp([start[0], start[1], 0.15])
        lab.close_gripper(0.06, grip_force_n=8.0)
        lab.move_tcp([start[0], start[1], 0.30])
        lifted_mm = 1000.0 * (float(lab.object_pose()[2]) - start[2])
        self.assertLess(lifted_mm, 20.0, "1.5 kg 应该滑脱（标定值：抬起 < 20 mm）")


if __name__ == "__main__":
    unittest.main()
