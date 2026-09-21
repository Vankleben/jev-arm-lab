"""P0/P1 measurement tool: does the grasp actually hold, and what makes it let go?

Runs the real grasp-then-lift path and records per-step contact forces, friction-cone
utilisation (|tangential| / (mu * normal)) and how far the object drifts relative to the
hand. Three modes:

    python tools/diagnose_slip.py                 # parameter sweep (condim / friction / grip force)
    python tools/diagnose_slip.py --mode ramp     # baseline in-hand slip vs ramp segments
    python tools/diagnose_slip.py --mode stress   # do the physics stressors really make it slip?

The lesson this tool exists to enforce: measure before you change. The original "unknown
cause" of slip was not friction at all (30x margin) but an `ik()` side effect that teleported
the arm, so the pads never physically travelled — see sim.ArmLab.ik.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_arm.sim import ArmLab                     # noqa: E402
from jev_arm.skills import GRASP_Z, PREGRASP_Z, CARRY_Z, OBJECT_WIDTH   # noqa: E402
from jev_arm.stress import STRESSORS               # noqa: E402


class Probe:
    """Per-physics-step sampler: cube contacts, forces, cone utilisation, drift."""

    def __init__(self, lab: ArmLab):
        self.lab = lab
        self.rows: list[dict] = []

    def __call__(self) -> None:
        lab, d = self.lab, self.lab.d
        n_con, normal_sum, worst, dims, fric = 0, 0.0, 0.0, set(), None
        for i in range(d.ncon):
            c = d.contact[i]
            if lab.geom_cube not in (int(c.geom1), int(c.geom2)):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(lab.m, d, i, force)
            n_con += 1
            normal = abs(float(force[0]))
            normal_sum += normal
            mu = float(c.friction[0])
            if mu > 0.0:
                worst = max(worst, float(np.hypot(force[1], force[2])) / (mu * max(normal, 1e-9)))
            dims.add(int(c.dim))
            fric = [round(float(x), 4) for x in c.friction[:3]]
        self.rows.append({
            "t": round(float(d.time), 3),
            "cube_z": round(float(lab.object_pose()[2]), 5),
            "tcp_z": round(float(lab.tcp_pos()[2]), 5),
            "gap_mm": round(lab.gripper_width() * 1000.0, 2),
            "ncon": n_con, "normal_n": round(normal_sum, 2),
            "cone_util": round(worst, 3), "dims": sorted(dims), "fric": fric,
        })


def capture_baseline(lab: ArmLab) -> dict:
    geoms = [lab.pads["left"][0], lab.pads["right"][0], lab.geom_cube]
    body = lab.m.geom_bodyid[lab.geom_cube]
    return {"geoms": geoms,
            "condim": {g: int(lab.m.geom_condim[g]) for g in geoms},
            "friction": {g: lab.m.geom_friction[g].copy() for g in geoms},
            "priority": {g: int(lab.m.geom_priority[g]) for g in geoms},
            "body": int(body),
            "body_mass": float(lab.m.body_mass[body]),
            "body_inertia": lab.m.body_inertia[body].copy()}


def reset_model(lab: ArmLab, base: dict) -> None:
    """Undo anything a previous variant (or a stressor) did to the model."""
    for gid in base["geoms"]:
        lab.m.geom_condim[gid] = base["condim"][gid]
        lab.m.geom_friction[gid][:] = base["friction"][gid]
        lab.m.geom_priority[gid] = base["priority"][gid]
    lab.m.body_mass[base["body"]] = base["body_mass"]
    lab.m.body_inertia[base["body"]] = base["body_inertia"]
    mujoco.mj_setConst(lab.m, lab.d)


def run_variant(lab: ArmLab, base: dict, label: str, grip_force_n: float = 8.0,
                condim: int | None = None, pad_friction: float | None = None,
                ramp: int = 8, stress_name: str | None = None,
                want_trace: bool = False) -> dict:
    lab.tick_hook = None
    lab.reset()
    reset_model(lab, base)
    if stress_name is not None:
        STRESSORS[stress_name].apply_physics(lab)
    if condim is not None:
        for gid in base["geoms"]:
            lab.m.geom_condim[gid] = condim
    if pad_friction is not None:
        for side in ("left", "right"):
            gid = lab.pads[side][0]
            lab.m.geom_priority[gid] = 1
            lab.m.geom_friction[gid][0] = pad_friction

    t0 = time.time()
    start = lab.object_pose().copy()
    lab.move_tcp([start[0], start[1], PREGRASP_Z], ramp=ramp)   # approach
    probe = Probe(lab)
    lab.tick_hook = probe

    xy = start[:2].copy()
    if stress_name is not None:      # 标定偏这类难例靠 stress.grasp_target 生效
        xy = np.asarray(STRESSORS[stress_name].grasp_target(xy, np.random.default_rng(7)),
                        dtype=float)
    lab.move_tcp([xy[0], xy[1], GRASP_Z], ramp=ramp)            # descend
    gap = lab.close_gripper(OBJECT_WIDTH, grip_force_n=grip_force_n)
    pinch = lab.pad_contacts()
    rel0 = float(lab.object_pose()[2] - lab.tcp_pos()[2])
    normal_after_close = probe.rows[-1]["normal_n"]

    lab.move_tcp([start[0], start[1], CARRY_Z], ramp=ramp)      # lift!
    end = lab.object_pose().copy()
    rel1 = float(end[2] - lab.tcp_pos()[2])

    drift_mm = 1000.0 * (rel1 - rel0)
    lifted_mm = 1000.0 * (end[2] - start[2])
    # HELD = 方块真的跟着手起来了（以前这里是"运动学携带"，永远为真）；
    # drift 是它在指间滑了多少 —— 质量指标，不是成败判据。
    held = bool(lifted_mm > 50.0 and lab.pad_contacts()["left"] + lab.pad_contacts()["right"] > 0)
    first_violation = next((r["t"] for r in probe.rows if r["cone_util"] > 0.9), None)
    peak_normal = max((r["normal_n"] for r in probe.rows), default=0.0)

    verdict = {
        "variant": label, "grip_force_n": grip_force_n, "ramp": ramp,
        "dims": probe.rows[-1]["dims"], "fric": probe.rows[-1]["fric"],
        "gap_after_close_mm": round(gap * 1000, 2),
        "pad_contacts_LR": [pinch["left"], pinch["right"]],
        "normal_after_close_n": normal_after_close, "peak_normal_n": peak_normal,
        "drift_mm": round(drift_mm, 2), "lifted_mm": round(lifted_mm, 2),
        "cone_first_over_0.9_at_t": first_violation,
        "seconds": round(time.time() - t0, 1),
        "HELD": held,
    }
    if want_trace:      # per-step trail is for debugging, not for the repo
        trace = Path(f"logs/diagnose_slip_trace_{label.split(' ')[0]}.json")
        trace.write_text(json.dumps(probe.rows, ensure_ascii=False), encoding="utf-8")
    return verdict


def build_variants(mode: str) -> list[dict]:
    if mode == "params":
        return [
            dict(label="baseline (condim=3, mu=1.2, 8N)"),
            dict(label="squeeze 20N", grip_force_n=20.0),
            dict(label="condim=4 (torsional on)", condim=4),
            dict(label="condim=6 (torsion+roll)", condim=6),
            dict(label="pad mu=1.8 (rubber)", pad_friction=1.8),
            dict(label="rubber + condim=6 + 20N", condim=6, pad_friction=1.8, grip_force_n=20.0),
        ]
    if mode == "ramp":
        return [dict(label=f"ramp={k}", ramp=k) for k in (2, 4, 8, 16, 32)]
    if mode == "stress":
        return [dict(label=f"stress:{name}", stress_name=name)
                for name in ("none", "low_friction", "heavy_object", "grasp_off")]
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["params", "ramp", "stress"], default="params")
    ap.add_argument("--trace", action="store_true", help="also dump per-step trails to logs/")
    args = ap.parse_args()

    lab = ArmLab()
    base = capture_baseline(lab)
    variants = build_variants(args.mode)

    results = []
    print(f"{'variant':26s} {'condim':>7s} {'mu':>5s} {'L/R':>5s} {'正常N':>7s} {'峰值N':>7s} "
          f"{'滑移mm':>8s} {'提起mm':>7s} {'耗时s':>5s}  held")
    for v in variants:
        r = run_variant(lab, base, want_trace=args.trace, **v)
        results.append(r)
        lr = f"{r['pad_contacts_LR'][0]}/{r['pad_contacts_LR'][1]}"
        print(f"{r['variant']:26s} {str(r['dims']):>7s} {r['fric'][0]:5.2f} {lr:>5s} "
              f"{r['normal_after_close_n']:7.2f} {r['peak_normal_n']:7.2f} "
              f"{r['drift_mm']:8.2f} {r['lifted_mm']:7.2f} {r['seconds']:5.1f}  "
              f"{'YES' if r['HELD'] else 'no'}")

    out = Path(f"logs/diagnose_slip_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
