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
