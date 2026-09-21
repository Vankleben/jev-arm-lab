"""Failure map: run the same task under each stressor and see where judgment breaks.

    python tools/stress_batch.py --scenes 10 --mode live

For every stressor (see jev_arm/stress.py) this runs N randomized scenes, then aggregates
one row: did the task finish, how accurate was `grasp_secure` (accuracy + Brier), and — the
row that matters most — how often the judge was confidently wrong:

  danger_fp  said "held" (p >= 0.6) while the object was NOT held   -> code would act on a lie
  danger_fn  said "not held" (p <= 0.4) while it WAS held           -> code would refuse to work

Usage notes: start with --mode fake (free) to check the wiring, then --mode live.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_arm.main import main as run_episode          # noqa: E402
from jev_arm.stress import STRESSORS                  # noqa: E402
from summarize_log import report                      # noqa: E402
from stress_map import scene_stats                    # noqa: E402


def run_stressor(name: str, scenes: int, mode: str, seed0: int, gates: tuple[float, float],
                 verbose: bool, out_root: str = "logs/stress") -> dict:
    out_dir = Path(f"{out_root}_{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jsonl"):
        old.unlink()

    done, escalated, dropped, errs, cycles = 0, 0, 0, [], []
    for i in range(scenes):
        seed = seed0 + i
        log = out_dir / f"scene_{seed:03d}.jsonl"
        rc = run_episode(["--jev-mode", mode, "--stress", name, "--randomize", "--seed", str(seed),
                          "--quiet", "--cycles", "20", "--log", str(log),
                          "--gate-confidence", str(gates[0]), "--gate-grasp", str(gates[1])])
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        s = scene_stats([r for r in rows if not r.get("meta")])
        if s["err_mm"] is not None:
            errs.append(s["err_mm"])
        cycles.append(s["cycles"])
        done += 1 if rc == 0 else 0
        escalated += 1 if rc == 2 else 0
        dropped += 1 if s["dropped"] else 0
        if verbose:
            tag = {0: "OK ", 2: "ESC", 1: "FAIL"}[rc if rc in (0, 1, 2) else 1]
            err = f"{s['err_mm']:6.1f}mm" if s["err_mm"] is not None else "     -  "
            print(f"    scene {i + 1:2d}/{scenes} seed={seed:3d} {tag} err={err} "
                  f"cycles={s['cycles']:2d}{' DROPPED' if s['dropped'] else ''}")

    stats = report([str(out_dir / "*.jsonl")], verbose=False)
    return {"stressor": name, "scenes": scenes, "finished": done / scenes,
            "escalated": escalated / scenes, "dropped": dropped / scenes,
            "err_median_mm": statistics.median(errs) if errs else None,
            "cycles_mean": statistics.mean(cycles),
            "accuracy": stats.get("accuracy"), "brier": stats.get("brier"),
            "progress_mae": stats.get("progress_mae"), "vetoes": stats.get("vetoes"),
            "danger_fp": stats.get("danger_fp", 0), "danger_fn": stats.get("danger_fn", 0),
            "danger_slip_fp": stats.get("danger_slip_fp", 0),
            "n_scored": stats.get("n", 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--mode", default="live", choices=["fake", "live"])
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--stressors", default=",".join(STRESSORS))
    ap.add_argument("--gate-confidence", type=float, default=0.30)
    ap.add_argument("--gate-grasp", type=float, default=0.45)
    ap.add_argument("--out-root", default="logs/stress",
                    help="输出目录前缀，每类难例写成 <前缀>_<名字>/；换前缀可保留上一轮证据")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    names = [n for n in args.stressors.split(",") if n]
    rows = []
    t0 = time.time()
    for name in names:
        print(f"\n[{name}] {args.scenes} scenes ({args.mode}) ...")
        rows.append(run_stressor(name, args.scenes, args.mode, args.seed0,
                                 (args.gate_confidence, args.gate_grasp), args.verbose,
                                 out_root=args.out_root))

    print("\n================ failure map ================")
    hdr = (f"{'stressor':14s} {'完成率':>7s} {'升级给人':>8s} {'掉件率':>7s} {'落点mm':>7s} {'轮数':>5s} "
           f"{'准确率':>7s} {'Brier':>6s} {'危险FP':>7s} {'滑脱FP':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        acc = f"{r['accuracy']:.0%}" if r["accuracy"] is not None else "  -  "
        brier = f"{r['brier']:.3f}" if r["brier"] is not None else "  -   "
        err = f"{r['err_median_mm']:7.1f}" if r["err_median_mm"] is not None else "      -"
        print(f"{r['stressor']:14s} {r['finished']:7.0%} {r['escalated']:8.0%} {r['dropped']:7.0%} "
              f"{err} {r['cycles_mean']:5.1f} {acc:>7s} {brier:>6s} {r['danger_fp']:7d} "
              f"{r['danger_slip_fp']:7d}")
    Path("logs/stress_map.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to logs/stress_map.json | wall time {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
