"""Skill library: each intent from the judge becomes a short Cartesian motion.

The judge never returns coordinates. It picks an intent; the geometry (where the object is,
where the target is) comes from this file, which is deterministic code you can read and
test. Same contract as the reference PoCs.
"""

from __future__ import annotations

import numpy as np

OBJECT_WIDTH = 0.06   # the cube is 6 cm
GRASP_Z = 0.15        # TCP (the grasp point between the pads) lands at the cube centre height
PREGRASP_Z = 0.26     # 11 cm above the grasp height
CARRY_Z = 0.30        # clears table + cube
TARGET_XY = np.array([0.36, -0.13])
RETREAT_XY = np.array([0.28, 0.0])

INTENTS = ["approach", "grasp", "lift", "carry", "lower", "release", "retreat", "hold", "finish"]


def _result(name: str, lab, note: str = "", tcp_err: float | None = None) -> dict:
    return {
        "skill": name,
        "note": note,
        "tcp_err_m": None if tcp_err is None else round(float(tcp_err), 4),
        "object": [round(float(v), 4) for v in lab.object_pose()],
        "gripper_gap_mm": round(lab.gripper_width() * 1000, 1),
    }


def approach(lab, ctx) -> dict:
    lab.open_gripper()
    obj = lab.object_pose()
    err = lab.move_tcp([obj[0], obj[1], PREGRASP_Z])
    return _result("approach", lab, "moved above the object", err)


def grasp(lab, ctx) -> dict:
    obj = lab.object_pose()
    xy = obj[:2]
    if getattr(lab, "stress", None) is not None:
        xy = lab.stress.grasp_target(xy, lab.rng)     # e.g. a 2 cm calibration error
    err = lab.move_tcp([xy[0], xy[1], GRASP_Z])
    gap = lab.close_gripper(OBJECT_WIDTH)
    t = lab.pad_contacts()
    held = t["left"] > 0 and t["right"] > 0
    note = (f"closed to {gap * 1000:.1f} mm, pad contacts L{t['left']}/R{t['right']}"
            + (" -> holding (by contact friction)" if held else " -> NOT holding"))
    return _result("grasp", lab, note, err)


def lift(lab, ctx) -> dict:
    tcp = lab.tcp_pos()
    err = lab.move_tcp([tcp[0], tcp[1], CARRY_Z])
    return _result("lift", lab, "raised the object", err)


def carry(lab, ctx) -> dict:
    err = lab.move_tcp([TARGET_XY[0], TARGET_XY[1], CARRY_Z])
    return _result("carry", lab, "moved above the target zone", err)


def lower(lab, ctx) -> dict:
    err = lab.move_tcp([TARGET_XY[0], TARGET_XY[1], GRASP_Z])
    return _result("lower", lab, "lowered to the target zone", err)


def release(lab, ctx) -> dict:
    lab.open_gripper()
    return _result("release", lab, "opened the gripper")


def retreat(lab, ctx) -> dict:
    err = lab.move_tcp([RETREAT_XY[0], RETREAT_XY[1], CARRY_Z])
    return _result("retreat", lab, "backed away from the workspace", err)


def hold(lab, ctx) -> dict:
    lab.settle(400)
    return _result("hold", lab, "held position (code vetoed the proposal)")


def finish(lab, ctx) -> dict:
    lab.settle(200)
    return _result("finish", lab, "task declared complete")


SKILLS = {
    "approach": approach, "grasp": grasp, "lift": lift, "carry": carry, "lower": lower,
    "release": release, "retreat": retreat, "hold": hold, "finish": finish,
}
