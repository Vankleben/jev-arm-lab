"""Rebuild the failure map from logs already on disk — no simulation, no API calls.

    python tools/stress_map.py

Scoring rules change (we learned that the hard way: the same model went from "useless" to
"excellent" purely by fixing how it was scored), so being able to re-score a finished batch
without paying for it again is the point of this tool.
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_log import load, report                      # noqa: E402

from jev_arm.skills import TARGET_XY                        # noqa: E402


def scene_rows(pattern: str) -> list[dict]:
    rows, _ = load([pattern])
    return rows


def scene_stats(rows: list[dict]) -> dict:
    """一个场景日志的汇总。

    升级给人（新鲜度闸门/看门狗触发）的场景以没有 "result" 字段的记录结尾——
    它们没有"落点"可言：不计入落点误差、不算完成，但计入升级。
    """
    executed = [r for r in rows if "result" in r]
    stats = {"cycles": len(rows),
             "escalated": any("escalation" in r for r in rows),
             "finished": False, "err_mm": None}
    if executed:
        last = executed[-1]
        stats["finished"] = bool((last.get("ground_truth_after") or {}).get("in_target"))
        obj = last["result"]["object"]
        stats["err_mm"] = 1000.0 * ((obj[0] - TARGET_XY[0]) ** 2
                                    + (obj[1] - TARGET_XY[1]) ** 2) ** 0.5
    return stats


def main() -> int:
    dirs = sorted(glob.glob("logs/stress_*"))
    if not dirs:
        print("no logs/stress_* directories found")
        return 1
    out = []
    for d in dirs:
        name = Path(d).name.replace("stress_", "")
        files = sorted(glob.glob(f"{d}/*.jsonl"))
        if not files:
            continue
        finished, escalated, errs, cycles = 0, 0, [], []
        for f in files:
            s = scene_stats(scene_rows(f))
            cycles.append(s["cycles"])
            finished += 1 if s["finished"] else 0
            escalated += 1 if s["escalated"] else 0
            if s["err_mm"] is not None:
                errs.append(s["err_mm"])
        stats = report([f"{d}/*.jsonl"], verbose=False)
        out.append({"stressor": name, "scenes": len(files), "finished": finished / len(files),
                    "escalated": escalated / len(files),
                    "err_median_mm": statistics.median(errs) if errs else None,
                    "cycles_mean": statistics.mean(cycles),
                    "accuracy": stats.get("accuracy"), "brier": stats.get("brier"),
                    "progress_mae": stats.get("progress_mae"), "vetoes": stats.get("vetoes", 0),
                    "danger_fp": stats.get("danger_fp", 0), "danger_fn": stats.get("danger_fn", 0),
                    "n_scored": stats.get("n", 0)})

    hdr = (f"{'stressor':15s} {'场景':>4s} {'完成率':>7s} {'升级率':>7s} {'落点mm':>7s} "
           f"{'轮数':>5s} {'准确率':>7s} {'Brier':>6s} {'进度MAE':>7s} {'危险FP':>7s} {'误拒FN':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(out, key=lambda x: (x["finished"], -x["danger_fp"])):
        acc = f"{r['accuracy']:.0%}" if r["accuracy"] is not None else "-"
        brier = f"{r['brier']:.3f}" if r["brier"] is not None else "-"
        err = f"{r['err_median_mm']:7.1f}" if r["err_median_mm"] is not None else "      -"
        print(f"{r['stressor']:15s} {r['scenes']:4d} {r['finished']:7.0%} {r['escalated']:7.0%} "
              f"{err} {r['cycles_mean']:5.1f} {acc:>7s} {brier:>6s} {r['progress_mae']:7.2f} "
              f"{r['danger_fp']:7d} {r['danger_fn']:7d}")
    Path("logs/stress_map.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to logs/stress_map.json  ({sum(r['scenes'] for r in out)} scenes total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
