"""P0-A: why does the grasped cube slip when lifted? Measure before changing anything.

Runs the real grasp-then-lift path on the current model, recording per-step contact
forces, friction-cone utilisation (|tangential| / (mu * normal)) and how far the cube
drifts relative to the hand. Then re-runs the same path under a few single-parameter
variants (contact dimension, pad friction, squeeze force) so the fix is chosen from
numbers instead of guesses.

    python tools/diagnose_slip.py            # sweep, print table, write logs/diagnose_slip.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_arm.sim import ArmLab                     # noqa: E402
from jev_arm.skills import GRASP_Z, PREGRASP_Z, CARRY_Z, OBJECT_WIDTH   # noqa: E402


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


def apply_variant(lab: ArmLab, base: dict, condim: int | None, pad_friction: float | None) -> None:
    """Set (or restore) model parameters under test. None -> model default."""
    geoms = [lab.pads["left"][0], lab.pads["right"][0], lab.geom_cube]
    for gid in geoms:
        lab.m.geom_condim[gid] = base["condim"][gid] if condim is None else condim
        if pad_friction is not None and gid != lab.geom_cube:
            lab.m.geom_friction[gid][:] = [pad_friction, 0.05, 0.001]
        else:
            lab.m.geom_friction[gid][:] = base["friction"][gid]


def capture_baseline(lab: ArmLab) -> dict:
    geoms = [lab.pads["left"][0], lab.pads["right"][0], lab.geom_cube]
    return {"condim": {g: int(lab.m.geom_condim[g]) for g in geoms},
            "friction": {g: lab.m.geom_friction[g].copy() for g in geoms}}


def run_variant(lab: ArmLab, base: dict, label: str, grip_force_n: float, condim: int | None,
                pad_friction: float | None, want_trace: bool = False) -> dict:
    lab.tick_hook = None
    lab.reset()
    apply_variant(lab, base, condim, pad_friction)

    start = lab.object_pose().copy()
    lab.move_tcp([start[0], start[1], PREGRASP_Z])      # approach, not instrumented
    probe = Probe(lab)
    lab.tick_hook = probe

    lab.move_tcp([start[0], start[1], GRASP_Z])         # descend
    gap = lab.close_gripper(OBJECT_WIDTH, grip_force_n=grip_force_n)
    pinch = lab.pad_contacts()
    rel0 = float(lab.object_pose()[2] - lab.tcp_pos()[2])
    normal_after_close = probe.rows[-1]["normal_n"]

    lab.move_tcp([start[0], start[1], CARRY_Z])         # lift!
    end = lab.object_pose().copy()
    rel1 = float(end[2] - lab.tcp_pos()[2])

    drift_mm = 1000.0 * (rel1 - rel0)
    lifted_mm = 1000.0 * (end[2] - start[2])
    held = abs(drift_mm) < 5.0 and lifted_mm > 50.0 and lab.pad_contacts()["left"] + lab.pad_contacts()["right"] > 0
    first_violation = next((r["t"] for r in probe.rows if r["cone_util"] > 0.9), None)
    lost_contact = next((r["t"] for r in probe.rows if r["cone_util"] > 0 and r["ncon"] == 0), None)
    peak_normal = max((r["normal_n"] for r in probe.rows), default=0.0)

    verdict = {
        "variant": label, "grip_force_n": grip_force_n,
        "dims": probe.rows[-1]["dims"], "fric": probe.rows[-1]["fric"],
        "gap_after_close_mm": round(gap * 1000, 2),
        "pad_contacts_LR": [pinch["left"], pinch["right"]],
        "normal_after_close_n": normal_after_close, "peak_normal_n": peak_normal,
        "drift_mm": round(drift_mm, 2), "lifted_mm": round(lifted_mm, 2),
        "cone_first_over_0.9_at_t": first_violation, "contact_lost_at_t": lost_contact,
        "HELD": held,
    }
    if want_trace:      # per-step trail is for debugging, not for the repo
        trace = Path(f"logs/diagnose_slip_trace_{label.split(' ')[0]}.json")
        trace.write_text(json.dumps(probe.rows, ensure_ascii=False), encoding="utf-8")
    return verdict


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="store_true", help="also dump per-step trails to logs/")
    args = ap.parse_args()

    lab = ArmLab()
    base = capture_baseline(lab)

    variants = [
        ("baseline (condim=3, mu=1.2, 8N)", 8.0, None, None),
        ("squeeze 20N", 20.0, None, None),
        ("condim=4 (torsional on)", 8.0, 4, None),
        ("condim=6 (torsion+roll)", 8.0, 6, None),
        ("pad mu=1.8 (rubber)", 8.0, None, 1.8),
        ("rubber + condim=6 + 20N", 20.0, 6, 1.8),
    ]

    results = []
    print(f"{'variant':26s} {'condim':>7s} {'mu':>5s} {'L/R':>5s} {'正常N':>7s} {'峰值N':>7s} "
          f"{'漂移mm':>8s} {'提起mm':>7s} {'锥破t':>7s}  held")
    for label, force, condim, mu in variants:
        v = run_variant(lab, base, label, force, condim, mu, want_trace=args.trace)
        results.append(v)
        lr = f"{v['pad_contacts_LR'][0]}/{v['pad_contacts_LR'][1]}"
        viol = f"{v['cone_first_over_0.9_at_t']:.2f}" if v["cone_first_over_0.9_at_t"] else "-"
        print(f"{label:26s} {str(v['dims']):>7s} {v['fric'][0]:5.1f} {lr:>5s} "
              f"{v['normal_after_close_n']:7.2f} {v['peak_normal_n']:7.2f} "
              f"{v['drift_mm']:8.2f} {v['lifted_mm']:7.2f} {viol:>7s}  {'YES' if v['HELD'] else 'no'}")

    out = Path("logs/diagnose_slip.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
