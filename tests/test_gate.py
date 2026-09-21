"""新鲜度闸门（stale_channels）与卡死看门狗（watchdog_trip）的单测。

看门狗的契约（README / main.py 注释）：同一动作连续要第三次，**且**判断层自己的
进度估计毫无变化，才升级给人。这里专门锁住两个方向：
  * 看不见世界变化（进度不变）    -> 必须拦；
  * 世界在变（进度在动，合理重试）-> 不许拦。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_arm.main import PROGRESS_EPS, SKILL_NEEDS, stale_channels, watchdog_trip


def h(skill: str, progress: float) -> dict:
    return {"skill": skill, "progress": progress}


class TestStaleChannels(unittest.TestCase):
    def test_only_channels_the_skill_needs_are_checked(self):
        stale = stale_channels("grasp", {"camera": 2.0, "tactile": 0.1}, 1.0)
        self.assertEqual(stale, {"camera": 2.0})     # grasp 依赖两条，只有相机过期

    def test_fresh_channels_pass(self):
        self.assertEqual(stale_channels("lift", {"camera": 9.0, "tactile": 0.2}, 1.0), {})

    def test_hold_needs_nothing(self):
        self.assertEqual(SKILL_NEEDS["hold"], ())
        self.assertEqual(stale_channels("hold", {"camera": 99.0, "tactile": 99.0}, 1.0), {})

    def test_boundary_exactly_at_limit_is_fresh(self):
        self.assertEqual(stale_channels("approach", {"camera": 1.0}, 1.0), {})   # > 才算旧


class TestWatchdog(unittest.TestCase):
    def test_trips_on_repetition_with_frozen_progress(self):
        hist = [h("grasp", 0.0), h("grasp", 0.0)]
        self.assertTrue(watchdog_trip(hist, "grasp", 0.0))

    def test_spares_reasonable_retry_when_progress_moves(self):
        hist = [h("grasp", 0.0), h("grasp", 0.0)]
        self.assertFalse(watchdog_trip(hist, "grasp", 0.8))   # 模型看得见世界在变

    def test_spares_when_prior_estimate_moved(self):
        hist = [h("grasp", 0.0), h("grasp", 1.0)]
        self.assertFalse(watchdog_trip(hist, "grasp", 1.0))   # 三次里有一次在动就不算卡死

    def test_needs_two_prior_repeats(self):
        self.assertFalse(watchdog_trip([h("grasp", 0.0)], "grasp", 0.0))
        self.assertFalse(watchdog_trip([h("grasp", 0.0), h("lift", 0.0)], "grasp", 0.0))

    def test_hold_is_never_blocked(self):
        hist = [h("hold", 0.0), h("hold", 0.0)]
        self.assertFalse(watchdog_trip(hist, "hold", 0.0))

    def test_short_history_is_safe(self):
        self.assertFalse(watchdog_trip([], "grasp", 0.0))

    def test_progress_eps_semantics(self):
        hist = [h("grasp", 0.0), h("grasp", 0.0)]
        self.assertTrue(watchdog_trip(hist, "grasp", PROGRESS_EPS - 1e-9))
        self.assertFalse(watchdog_trip(hist, "grasp", PROGRESS_EPS + 1e-9))


if __name__ == "__main__":
    unittest.main()
