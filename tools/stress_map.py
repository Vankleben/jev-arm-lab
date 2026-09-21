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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_log import load, report                      # noqa: E402

TARGET_XY = (0.36, -0.13)   # the placement target of the lab scene


def scene_rows(pattern: str) -> list[dict]:
    rows, _ = load([pattern])
    return rows


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
        finished, errs, cycles = 0, [], []
        for f in files:
            rows = scene_rows(f)
            last = rows[-1]
            obj = last["result"]["object"]
            errs.append(1000.0 * ((obj[0] - TARGET_XY[0]) ** 2 + (obj[1] - TARGET_XY[1]) ** 2) ** 0.5)
            cycles.append(len(rows))
            finished += 1 if (last.get("ground_truth_after") or {}) .get("in_target") else 0
        stats = report([f"{d}/*.jsonl"], verbose=False)
        out.append({"stressor": name, "scenes": len(files), "finished": finished / len(files),
                    "err_median_mm": statistics.median(errs), "cycles_mean": statistics.mean(cycles),
                    "accuracy": stats.get("accuracy"), "brier": stats.get("brier"),
                    "progress_mae": stats.get("progress_mae"), "vetoes": stats.get("vetoes", 0),
                    "danger_fp": stats.get("danger_fp", 0), "danger_fn": stats.get("danger_fn", 0),
                    "n_scored": stats.get("n", 0)})

    hdr = (f"{'stressor':15s} {'场景':>4s} {'完成率':>7s} {'落点mm':>7s} {'轮数':>5s} "
           f"{'准确率':>7s} {'Brier':>6s} {'进度MAE':>7s} {'危险FP':>7s} {'误拒FN':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(out, key=lambda x: (x["finished"], -x["danger_fp"])):
        acc = f"{r['accuracy']:.0%}" if r["accuracy"] is not None else "-"
        brier = f"{r['brier']:.3f}" if r["brier"] is not None else "-"
        print(f"{r['stressor']:15s} {r['scenes']:4d} {r['finished']:7.0%} {r['err_median_mm']:7.1f} "
              f"{r['cycles_mean']:5.1f} {acc:>7s} {brier:>6s} {r['progress_mae']:7.2f} "
              f"{r['danger_fp']:7d} {r['danger_fn']:7d}")
    Path("logs/stress_map.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to logs/stress_map.json  ({sum(r['scenes'] for r in out)} scenes total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
