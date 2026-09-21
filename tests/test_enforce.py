"""enforce() 的否决规则单测。

代码否决权是这个实验台的安全边界（judge 只看到孤立问题，跨答案的一致性全靠这里），
三条规则、它们的边界值和优先级顺序必须逐条锁住。跑法见 README：
python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_arm.judge import Decision
from jev_arm.main import GRASP_RETRY_GATE, enforce


def d(intent="grasp", grasp=0.0, conf=1.0, progress=0.0) -> Decision:
    return Decision(intent=intent, intent_confidence=conf,
                    grasp_secure_prob=grasp, progress_score=progress)


class TestGraspGate(unittest.TestCase):
    """规则 1：没抓稳不许搬动（lift/carry/lower -> grasp 或 hold）。"""

    def test_lift_blocked_below_gate(self):
        final, note = enforce(d("lift", grasp=0.44), 0.30, 0.45)
        self.assertEqual(final, "grasp")          # >= GRASP_RETRY_GATE，值得再抓一次
        self.assertIn("不许搬动", note)

    def test_lift_downgrades_to_hold_when_hopeless(self):
        final, _ = enforce(d("lift", grasp=0.10), 0.30, 0.45)
        self.assertEqual(final, "hold")

    def test_retry_boundary_is_inclusive(self):
        final, _ = enforce(d("lift", grasp=GRASP_RETRY_GATE), 0.30, 0.45)
        self.assertEqual(final, "grasp")

    def test_carry_with_secure_grasp_passes_untouched(self):
        final, note = enforce(d("carry", grasp=0.80), 0.30, 0.45)
        self.assertEqual(final, "carry")
        self.assertEqual(note, "")

    def test_lower_blocked_too(self):
        final, _ = enforce(d("lower", grasp=0.20), 0.30, 0.45)
        self.assertEqual(final, "hold")


class TestConfidenceGate(unittest.TestCase):
    """规则 2：intent 置信度不足 -> 原地保持。"""

    def test_low_confidence_holds(self):
        final, note = enforce(d("grasp", grasp=0.9, conf=0.29), 0.30, 0.45)
        self.assertEqual(final, "hold")
        self.assertIn("原地保持", note)

    def test_boundary_confidence_passes(self):
        final, _ = enforce(d("grasp", grasp=0.9, conf=0.30), 0.30, 0.45)
        self.assertEqual(final, "grasp")


class TestProgressOverride(unittest.TestCase):
    """规则 3：判断层自己说任务完成了，却不退开 -> 代码直接改判退开。"""

    def test_progress3_forces_retreat(self):
        final, note = enforce(d("approach", grasp=0.9, progress=3.0), 0.30, 0.45)
        self.assertEqual(final, "retreat")
        self.assertIn("退开", note)

    def test_progress3_respects_explicit_retreat(self):
        final, _ = enforce(d("retreat", grasp=0.9, progress=3.0), 0.30, 0.45)
        self.assertEqual(final, "retreat")

    def test_confidence_veto_wins_over_progress_retreat(self):
        # 置信度闸门在后、把 final 改成 hold；hold 在"不许覆盖"名单里，不能又被改成 retreat
        final, note = enforce(d("approach", grasp=0.9, conf=0.10, progress=3.0), 0.30, 0.45)
        self.assertEqual(final, "hold")
        self.assertIn("原地保持", note)


if __name__ == "__main__":
    unittest.main()
