"""summarize_log 打分口径的 golden 测试。

本项目的核心教训（FINDINGS 第 3 节）：只改打分口径就能把 77% 变成 100%。
所以打分脚本的行为必须被钉死：这里的 13 条记录是手工构造、每个期望值都
能手算的——谁改动口径，测试立刻红，逼他像 FINDINGS 那样写清楚为什么。
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from summarize_log import report   # noqa: E402


def row(source="jev", prob=0.5, evidence="none", prog=0.0, gt_prog=0,
        override="", intent="grasp", slipping=False) -> dict:
    gt: dict = {"grasp_evidence": evidence, "slipping": slipping}
    if gt_prog == 1:
        gt.update(attached=True, lifted=False, in_target=False)
    elif gt_prog == 2:
        gt.update(attached=True, lifted=True, in_target=False)
    elif gt_prog == 3:
        gt.update(attached=False, lifted=False, in_target=True)
    return {"judge": {"source": source, "grasp_secure_prob": prob, "progress_score": prog},
            "ground_truth": gt, "override": override, "final_intent": intent}


# 10 个可打分样本（准确率 8/10，Brier 0.14544，危险FP=1 危险FN=1，代价最优门槛 0.65）
# + 2 个 unproven（跳过打分）+ 1 个 fallback（不进 grasp 打分，但进 progress/veto 统计）
ROWS = [
    row(prob=0.05, evidence="none",    prog=0.0, gt_prog=0, intent="approach"),
    row(prob=0.10, evidence="none",    prog=0.1, gt_prog=0, intent="grasp"),
    row(prob=0.20, evidence="none",    prog=0.0, gt_prog=0, intent="grasp"),
    row(prob=0.30, evidence="proven",  prog=1.0, gt_prog=1, intent="lift"),
    row(prob=0.45, evidence="none",    prog=0.2, gt_prog=0, intent="hold", override="veto"),
    row(prob=0.55, evidence="proven",  prog=2.0, gt_prog=2, intent="carry"),
    row(prob=0.70, evidence="proven",  prog=1.7, gt_prog=2, intent="lower"),
    row(prob=0.85, evidence="proven",  prog=2.0, gt_prog=2, intent="release"),
    row(prob=0.90, evidence="proven",  prog=3.0, gt_prog=3, intent="retreat"),
    row(prob=0.62, evidence="none",    prog=2.4, gt_prog=2, intent="carry"),
    row(prob=0.50, evidence="unproven", prog=1.0, gt_prog=1, intent="grasp"),
    row(prob=0.50, evidence="unproven", prog=0.0, gt_prog=0, intent="hold"),
    row(source="fallback", prob=0.80, evidence="proven", prog=1.0, gt_prog=1,
        intent="approach", override="veto"),
]


def write_fixture(rows: list[dict]) -> str:
    path = Path(tempfile.mkdtemp()) / "fixture.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": True, "scene": {"seed": 0}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


class TestGoldenScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = write_fixture(ROWS)
        cls.stats = report([cls.log], verbose=False)

    def test_grasp_scoring(self):
        self.assertEqual(self.stats["n"], 10)                     # unproven+fallback 被剔除
        self.assertAlmostEqual(self.stats["accuracy"], 0.8)
        self.assertAlmostEqual(self.stats["brier"], 0.14544, places=4)

    def test_danger_directions(self):
        self.assertEqual(self.stats["danger_fp"], 1)   # 0.62 说抓住其实没抓住
        self.assertEqual(self.stats["danger_fn"], 1)   # 0.30 说没抓住其实抓住了

    def test_cost_weighted_gate(self):
        # FP 计 3 倍代价：0.65 恰好 FP=0、FN=2（代价 2），低于它 FP 代价反超
        self.assertAlmostEqual(self.stats["suggested_grasp_gate"], 0.65, places=6)

    def test_progress_mae_counts_all_sources(self):
        # 总误差 0.1+0.2+0.3+0.4 = 1.0，分母是全部 13 轮（含 unproven 和 fallback）
        self.assertAlmostEqual(self.stats["progress_mae"], 1.0 / 13.0, places=6)

    def test_counts(self):
        self.assertEqual(self.stats["cycles"], 13)
        self.assertEqual(self.stats["scenes"], 1)
        self.assertEqual(self.stats["vetoes"], 2)
        self.assertEqual(self.stats["fallback_cycles"], 1)
        self.assertEqual(self.stats["intents"]["grasp"], 3)


class TestDangerSlip(unittest.TestCase):
    """物理危险指标：说"抓稳了"（>=0.6）而真值在滑 —— danger_fp 的物理版本。"""

    def test_only_high_confidence_while_slipping_counts(self):
        rows = [
            row(prob=0.80, evidence="proven", prog=2.0, gt_prog=2, slipping=True),   # 计 1
            row(prob=0.30, evidence="proven", prog=2.0, gt_prog=2, slipping=True),   # 置信度低，不计
            row(prob=0.90, evidence="proven", prog=2.0, gt_prog=2, slipping=False),  # 说对了，不计
        ]
        stats = report([write_fixture(rows)], verbose=False)
        self.assertEqual(stats["danger_slip_fp"], 1)

    def test_old_logs_without_slipping_key_are_safe(self):
        legacy = row(prob=0.9, evidence="proven", prog=2.0, gt_prog=2)
        del legacy["ground_truth"]["slipping"]   # 运动学携带时期的日志没这个键
        stats = report([write_fixture([legacy])], verbose=False)
        self.assertEqual(stats["danger_slip_fp"], 0)


class TestFallbackWarning(unittest.TestCase):
    def test_warning_printed_when_fallback_present(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report([write_fixture(ROWS)], verbose=True)
        self.assertIn("WARNING", out.getvalue())
        self.assertIn("fell back", out.getvalue())

    def test_no_warning_for_clean_live_run(self):
        clean = [r for r in ROWS if r["judge"]["source"] == "jev"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report([write_fixture(clean)], verbose=True)
        self.assertNotIn("WARNING", out.getvalue())


if __name__ == "__main__":
    unittest.main()
