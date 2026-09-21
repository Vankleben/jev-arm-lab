"""stress_map.scene_stats 的单测：升级场景的日志必须能重算，不能崩。

修复前的 bug：main() 假定每份日志最后一条是执行记录，直接取 ["result"]——
升级给人类的场景（相机冻结/遮挡）最后一条是升级记录，一读就 KeyError，
README 承诺的"零成本重算"对恰恰最关键的那两行跑不通。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from stress_map import scene_stats     # noqa: E402


def executed(obj=(0.36, -0.13, 0.15), in_target=True) -> dict:
    return {"result": {"object": list(obj)}, "ground_truth_after": {"in_target": in_target}}


def carried_row(tcp_z: float, obj_z: float, attached: bool, in_target: bool = False,
                was_held: bool = False) -> dict:
    return {"result": {"object": [0.36, -0.13, obj_z]},
            "state": {"tcp": {"pos_m": [0.36, -0.13, tcp_z]}},
            "ground_truth": {"attached": was_held},
            "ground_truth_after": {"attached": attached, "in_target": in_target}}


class TestDropDetection(unittest.TestCase):
    def test_approach_alone_is_not_a_drop(self):
        # 回归：曾经把"接近阶段手在方块上方"误判成掉件，旧日志 40 个场景全被判成 100% 掉件
        s = scene_stats([carried_row(tcp_z=0.30, obj_z=0.15, attached=False)])
        self.assertFalse(s["dropped"])

    def test_mid_air_loss_is_a_drop(self):
        rows = [carried_row(tcp_z=0.30, obj_z=0.29, attached=True),
                carried_row(tcp_z=0.30, obj_z=0.15, attached=False, was_held=True)]
        self.assertTrue(scene_stats(rows)["dropped"])

    def test_release_in_target_is_not_a_drop(self):
        rows = [carried_row(tcp_z=0.20, obj_z=0.15, attached=True),
                carried_row(tcp_z=0.30, obj_z=0.15, attached=False, in_target=True)]
        self.assertFalse(scene_stats(rows)["dropped"])

    def test_release_outside_target_while_hand_is_low_is_not_a_drop(self):
        rows = [carried_row(tcp_z=0.16, obj_z=0.15, attached=False, was_held=True)]
        self.assertFalse(scene_stats(rows)["dropped"])   # 手还低着，只是放偏了


class TestSceneStats(unittest.TestCase):
    def test_completed_scene(self):
        s = scene_stats([{"judge": {}}, executed()])
        self.assertTrue(s["finished"])
        self.assertFalse(s["escalated"])
        self.assertAlmostEqual(s["err_mm"], 0.0, places=6)
        self.assertEqual(s["cycles"], 2)

    def test_executed_but_missed(self):
        s = scene_stats([executed(obj=(0.10, 0.10, 0.15), in_target=False)])
        self.assertFalse(s["finished"])
        self.assertGreater(s["err_mm"], 100.0)

    def test_escalated_before_any_motion(self):
        # 相机冻结：第 0 轮就被闸门拦下，整份日志没有一条执行记录
        s = scene_stats([{"escalation": {"reason": "stale"}}])
        self.assertFalse(s["finished"])
        self.assertTrue(s["escalated"])
        self.assertIsNone(s["err_mm"])     # 没有"落点"可言，不许编一个数

    def test_executed_then_escalated(self):
        # 遮挡：能动几轮，最后卡死升级——完成与否看最后一次执行，不是看升级记录
        rows = [executed(obj=(0.31, 0.06, 0.15), in_target=False),
                {"escalation": {"reason": "stale"}}]
        s = scene_stats(rows)
        self.assertFalse(s["finished"])
        self.assertTrue(s["escalated"])
        self.assertGreater(s["err_mm"], 100.0)


if __name__ == "__main__":
    unittest.main()
