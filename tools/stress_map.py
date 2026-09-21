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

    `dropped`（掉件）= 手还在高处，方块却已经不在手上、也不在目标区。既覆盖搬运
    中途滑落，也覆盖"在目标区外就松开"——两者都是"东西没送到"，对真机都算事故。
    """
    executed = [r for r in rows if "result" in r]
    stats = {"cycles": len(rows),
             "escalated": any("escalation" in r for r in rows),
             "finished": False, "err_mm": None, "dropped": False}
    if executed:
        last = executed[-1]
        stats["finished"] = bool((last.get("ground_truth_after") or {}).get("in_target"))
        obj = last["result"]["object"]
        stats["err_mm"] = 1000.0 * ((obj[0] - TARGET_XY[0]) ** 2
                                    + (obj[1] - TARGET_XY[1]) ** 2) ** 0.5
    # 掉件：曾经拿住过，之后手在高处而方块已不在手上、也不在目标区。
    # 必须先出现过 attached=True，否则"接近阶段手在方块上方"会被误判成掉件
    # （这是拿旧日志重算时抓到的 bug：40 个成功场景被判成 100% 掉件）。
    was_held = False
    for r in executed:
        state = r.get("state") or {}
        tcp = (state.get("tcp") or {}).get("pos_m")
        gt_after = r.get("ground_truth_after") or {}
        if tcp and was_held and tcp[2] - r["result"]["object"][2] > 0.12 \
                and not gt_after.get("attached") and not gt_after.get("in_target"):
            stats["dropped"] = True
            break
        was_held = was_held or bool(gt_after.get("attached")) \
            or bool((r.get("ground_truth") or {}).get("attached"))
    return stats


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/stress_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"no directories matching {pattern}")
        return 1
    out = []
    for d in dirs:
        name = Path(d).name.replace("stress_", "")
        files = sorted(glob.glob(f"{d}/*.jsonl"))
        if not files:
            continue
        finished, escalated, dropped, errs, cycles = 0, 0, 0, [], []
        for f in files:
            s = scene_stats(scene_rows(f))
            cycles.append(s["cycles"])
            finished += 1 if s["finished"] else 0
            escalated += 1 if s["escalated"] else 0
            dropped += 1 if s["dropped"] else 0
            if s["err_mm"] is not None:
                errs.append(s["err_mm"])
        stats = report([f"{d}/*.jsonl"], verbose=False)
        out.append({"stressor": name, "scenes": len(files), "finished": finished / len(files),
                    "escalated": escalated / len(files), "dropped": dropped / len(files),
                    "err_median_mm": statistics.median(errs) if errs else None,
                    "cycles_mean": statistics.mean(cycles),
                    "accuracy": stats.get("accuracy"), "brier": stats.get("brier"),
                    "progress_mae": stats.get("progress_mae"), "vetoes": stats.get("vetoes", 0),
                    "danger_fp": stats.get("danger_fp", 0), "danger_fn": stats.get("danger_fn", 0),
                    "danger_slip_fp": stats.get("danger_slip_fp", 0),
                    "n_scored": stats.get("n", 0)})

    hdr = (f"{'stressor':15s} {'场景':>4s} {'完成率':>7s} {'升级率':>7s} {'掉件率':>7s} {'落点mm':>7s} "
           f"{'轮数':>5s} {'准确率':>7s} {'Brier':>6s} {'进度MAE':>7s} {'危险FP':>7s} {'滑脱FP':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(out, key=lambda x: (x["finished"], -x["dropped"])):
        acc = f"{r['accuracy']:.0%}" if r["accuracy"] is not None else "-"
        brier = f"{r['brier']:.3f}" if r["brier"] is not None else "-"
        err = f"{r['err_median_mm']:7.1f}" if r["err_median_mm"] is not None else "      -"
        print(f"{r['stressor']:15s} {r['scenes']:4d} {r['finished']:7.0%} {r['escalated']:7.0%} "
              f"{r['dropped']:7.0%} {err} {r['cycles_mean']:5.1f} {acc:>7s} {brier:>6s} "
              f"{r['progress_mae']:7.2f} {r['danger_fp']:7d} {r['danger_slip_fp']:7d}")
    Path("logs/stress_map.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to logs/stress_map.json  ({sum(r['scenes'] for r in out)} scenes total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
