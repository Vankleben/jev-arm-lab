"""Score one or many runs: is the judge right, and is its confidence honest?

    python tools/summarize_log.py logs/jev_run.jsonl                 # one log
    python tools/summarize_log.py "logs/batch_live/*.jsonl"          # a whole batch

The simulation keeps ground truth (`ground_truth` in every record) that the judge never
sees, so every answer can be scored:

* `grasp_secure` (noul) — a probability, so it gets a reliability table: bucket the
  predicted probability, compare with how often it was actually true, plus a Brier score.
  A threshold sweep follows, with the false-positive rate called out, because for a robot
  the costly mistake is "said it was held, it was not".
* `task_progress` (score) — compared against progress derived from the simulation.
* intent + vetoes — how often the code overrode the model, and whether runs finished.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def load(patterns: list[str]) -> tuple[list[dict], list[dict]]:
    files: list[str] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        files.extend(hits or [pattern])
    rows, scenes = [], []
    for f in files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("meta"):
                scenes.append({"file": Path(f).name, **record["scene"]})
            else:
                rows.append(record)
    return rows, scenes


def ground_truth_progress(gt: dict) -> float:
    if gt.get("in_target") and not gt.get("attached"):
        return 3.0
    if gt.get("attached") and gt.get("lifted"):
        return 2.0
    if gt.get("attached"):
        return 1.0
    return 0.0


def report(patterns: list[str], verbose: bool = True) -> dict:
    rows, scenes = load(patterns)
    if not rows:
        if verbose:
            print("no records found")
        return {}
    sources = {r["judge"]["source"] for r in rows}
    say = print if verbose else (lambda *a, **k: None)
    say(f"logs: {len(scenes)} scene(s) | cycles: {len(rows)} | judge source: {', '.join(sorted(sources))}")

    # 只给"无歧义"的样本打分：none = 确实没抓住，proven = 抓住并抬起过，
    # unproven（刚合上、还没动过）= 模型说"不确定"是对的，不计分。
    scored = [(r["judge"]["grasp_secure_prob"], r["ground_truth"].get("grasp_evidence", "unproven"))
              for r in rows if r["judge"]["source"] == "jev"]
    ambiguous = sum(1 for _, ev in scored if ev == "unproven")
    probs = [(p, 1 if ev == "proven" else 0) for p, ev in scored if ev != "unproven"]
    say(f"scored on unambiguous states only: {len(probs)} of {len(scored)} cycles "
          f"({ambiguous} 'just closed, not yet proven' skipped)")
    prog_err = [r["judge"]["progress_score"] - ground_truth_progress(r["ground_truth"]) for r in rows]

    stats: dict = {"cycles": len(rows), "scenes": len(scenes)}
    if probs:
        ps = [p for p, _ in probs]
        ys = [y for _, y in probs]
        correct = sum((p >= 0.5) == bool(y) for p, y in probs)
        brier = sum((p - y) ** 2 for p, y in probs) / len(probs)
        stats.update({"accuracy": correct / len(probs), "brier": brier, "n": len(probs)})
        say(f"\ngrasp_secure (noul), n={len(probs)}")
        say(f"  accuracy @0.5 : {correct}/{len(probs)} = {correct / len(probs):.0%}")
        say(f"  Brier score   : {brier:.3f}   (0 = perfect, 0.25 = always saying 0.5)")
        say("  reliability   : bucket -> observed frequency (n)")
        for k in range(5):
            ys_b = [y for p, y in probs if min(int(p * 5), 4) == k]
            if ys_b:
                say(f"    {k * 0.2:.1f}-{k * 0.2 + 0.2:.1f} : {sum(ys_b) / len(ys_b):.2f}  (n={len(ys_b)})")

        say("  threshold sweep (false positive = said held, was not)")
        say("    thr   kept   FP   FN")
        rows_sweep = []
        for i in range(1, 19):
            t = i * 0.05
            tp = sum(1 for p, y in probs if p >= t and y)
            fp = sum(1 for p, y in probs if p >= t and not y)
            fn = sum(1 for p, y in probs if p < t and y)
            rows_sweep.append((t, tp, fp, fn))
            say(f"    {t:.2f}  {tp:5d} {fp:5d} {fn:5d}")
        # 机器人上"说抓住了其实没抓住"(FP)比"能抓却没抓"(FN)危险得多，按 3:1 计代价
        t, tp, fp, fn = min(rows_sweep, key=lambda r: 3 * r[2] + r[3])
        say(f"  -> cost-weighted pick (FP weighs 3x FN): thr={t:.2f} "
              f"(keeps {tp} real holds, FP={fp}, FN={fn})")
        stats["suggested_grasp_gate"] = t
    else:
        say("\ngrasp_secure: judge is rule-based (fake), no probabilities to score")

    mae = sum(abs(e) for e in prog_err) / len(prog_err)
    stats["progress_mae"] = mae
    vetoes = sum(1 for r in rows if r["override"])
    say(f"\ntask_progress: mean absolute error {mae:.2f} levels (of 0-3)")
    say(f"code vetoes  : {vetoes}/{len(rows)} cycles")
    intents: dict[str, int] = {}
    for r in rows:
        intents[r["final_intent"]] = intents.get(r["final_intent"], 0) + 1
    say("final intents: " + ", ".join(f"{k} x{v}" for k, v in sorted(intents.items(), key=lambda x: -x[1])))
    stats["vetoes"] = vetoes
    if probs:
        stats["danger_fp"] = sum(1 for p, y in probs if p >= 0.6 and not y)   # 说抓住其实没抓住
        stats["danger_fn"] = sum(1 for p, y in probs if p <= 0.4 and y)       # 说没抓住其实抓住了
    stats["intents"] = intents
    return stats


if __name__ == "__main__":
    args = sys.argv[1:] or ["logs/jev_run.jsonl"]
    report(args)
