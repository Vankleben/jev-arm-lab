"""端到端冒烟：fake 模式（免 API）从随机位置完整跑一遍抓放，外加 CLI 防线。

* seed 0 必须完成任务（返回码 0）、落点误差 < 10 mm、日志里每轮都带真值；
  阈值定得宽松是有意的：搬运从运动学携带改成真实接触摩擦后，实测中位 2.3 mm /
  最差 3.1 mm（logs/batch_fake_20_physical），这里只拦"物理坏了"级别的回归；
* 拼错的 --stress 名字必须当场报错，而不是静默按无故障跑；
* 已删除的 --no-frames 必须被 argparse 拒绝（防死参数复活）。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)          # 模型路径是相对仓库根的

from jev_arm.main import main           # noqa: E402
from jev_arm.skills import TARGET_XY    # noqa: E402


class TestSmokeRun(unittest.TestCase):
    def test_fake_mode_completes_seeded_scene(self):
        log = Path(tempfile.mkdtemp()) / "smoke.jsonl"
        rc = main(["--jev-mode", "fake", "--randomize", "--seed", "0", "--cycles", "24",
                   "--quiet", "--log", str(log)])
        self.assertEqual(rc, 0, "fake 模式 seed 0 应当完成任务（返回码 0）")

        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertTrue(rows[0].get("meta"), "首行必须是 meta")
        executed = [r for r in rows[1:] if "result" in r]
        self.assertGreater(len(executed), 0)
        for r in executed:
            self.assertIn("ground_truth", r, "每轮日志必须带执行前真值，否则无法打分")
            self.assertIn("ground_truth_after", r)
        self.assertFalse(any("escalation" in r for r in rows[1:]),
                         "基线场景不应触发升级")
        obj = executed[-1]["result"]["object"]
        err_mm = 1000.0 * ((obj[0] - TARGET_XY[0]) ** 2 + (obj[1] - TARGET_XY[1]) ** 2) ** 0.5
        self.assertLess(err_mm, 10.0, f"落点误差 {err_mm:.1f} mm 超出冒烟阈值")


class TestCliGuards(unittest.TestCase):
    def test_unknown_stressor_fails_loudly(self):
        with self.assertRaises(ValueError):
            main(["--jev-mode", "fake", "--stress", "typo_x"])

    def test_removed_no_frames_flag_stays_dead(self):
        with self.assertRaises(SystemExit):
            main(["--no-frames"])


if __name__ == "__main__":
    unittest.main()
