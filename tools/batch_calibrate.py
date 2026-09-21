"""Batch calibration: run many randomized scenes, then aggregate the judge's answers.

    python tools/batch_calibrate.py --mode fake --n 20          # free sanity check
    python tools/batch_calibrate.py --mode live --n 40          # real Jev (costs pennies)

Each scene randomizes the cube's start position (and a little yaw), runs the full decision
loop with its own JSONL log, and the logs are then scored together so the reliability table
has enough samples to mean something.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_arm.main import main as run_episode          # noqa: E402
from jev_arm.skills import TARGET_XY                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_log import report                      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fake", choices=["fake", "live"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None, help="directory for per-scene logs")
    ap.add_argument("--gate-confidence", type=float, default=0.30)
    ap.add_argument("--gate-grasp", type=float, default=0.45)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out or f"logs/batch_{args.mode}_{args.n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"batch: {args.n} scenes | mode={args.mode} | gates={args.gate_confidence}/{args.gate_grasp}")
    print(f"logs -> {out_dir}\n")

    results, t_start = [], time.time()
    for i in range(args.n):
        seed = args.seed0 + i
        log = out_dir / f"scene_{seed:03d}.jsonl"
        t0 = time.time()
        rc = run_episode(["--jev-mode", args.mode, "--randomize", "--seed", str(seed), "--quiet",
                          "--cycles", "20", "--log", str(log),
                          "--gate-confidence", str(args.gate_confidence),
                          "--gate-grasp", str(args.gate_grasp)])
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        scene = rows[0]["scene"]["cube_start"]
        obj = rows[-1]["result"]["object"]
        err_mm = 1000.0 * ((obj[0] - TARGET_XY[0]) ** 2 + (obj[1] - TARGET_XY[1]) ** 2) ** 0.5
        vetoes = sum(1 for r in rows[1:] if r["override"])
        results.append({"seed": seed, "ok": rc == 0, "cube": scene, "err_mm": err_mm,
                        "cycles": len(rows) - 1, "vetoes": vetoes, "secs": time.time() - t0})
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  [{i + 1:3d}/{args.n}] {mark} seed={seed:3d} cube=({scene[0]:.3f},{scene[1]:.3f}) "
              f"err={err_mm:6.1f}mm cycles={len(rows) - 1:2d} veto={vetoes:2d} {time.time() - t0:5.1f}s")

    ok = [r for r in results if r["ok"]]
    print(f"\n=== batch summary ({args.mode}) ===")
    print(f"completed      : {len(ok)}/{len(results)} = {len(ok) / len(results):.0%}")
    if ok:
        errs = sorted(r["err_mm"] for r in ok)
        print(f"placement error: median {errs[len(errs) // 2]:.1f} mm | worst {errs[-1]:.1f} mm")
    if len(ok) < len(results):
        print("failed scenes  :")
        for r in results:
            if not r["ok"]:
                print(f"   seed={r['seed']:3d} cube=({r['cube'][0]:.3f},{r['cube'][1]:.3f}) "
                      f"err={r['err_mm']:.1f}mm cycles={r['cycles']} vetoes={r['vetoes']}")
    print(f"wall time      : {time.time() - t_start:.0f}s total")

    print("\n=== aggregated judge scoring ===")
    report([str(out_dir / "*.jsonl")])

    (out_dir / "batch_summary.json").write_text(
        json.dumps({"args": vars(args), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nper-scene results written to {out_dir / 'batch_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
