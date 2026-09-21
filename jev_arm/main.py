"""Run the pick-and-place decision loop end to end.

    python -m jev_arm.main --jev-mode fake --cycles 24            # no API key needed
    python -m jev_arm.main --jev-mode live --cycles 12            # real Jev, needs TYPESAFE_API_KEY
    python -m jev_arm.main --jev-mode fake --viewer               # watch it in a MuJoCo window

Every cycle is written to a JSONL log together with the simulation's ground truth, so the
judge's answers can be scored afterwards (tools/summarize_log.py).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import mujoco
import numpy as np

from .judge import build_state, make_judge
from .sensors import SensorSuite
from .sim import ArmLab
from .skills import SKILLS, TARGET_XY
from .stress import get as get_stress

DEFAULT_INSTRUCTION = "把桌上的红色方块抓起来放进蓝色目标区，然后退开"

# 每个技能真正依赖哪条传感通道。新鲜度闸门只检查该技能需要的那些数据——
# 拿的是过期数据就不许动，交给人。见 README 里"视觉冻结"那一条。
SKILL_NEEDS = {
    "approach": ("camera",), "grasp": ("camera", "tactile"),
    "lift": ("tactile",), "carry": ("camera", "tactile"),
    "lower": ("camera",), "release": ("tactile",),
    "retreat": ("camera",), "finish": ("camera",), "hold": (),
}


def enforce(d, gate_confidence: float, gate_grasp: float) -> tuple[str, str]:
    """Code owns the loop: it may veto, downgrade or override the judge's proposal.

    The judge sees each question in isolation, so consistent behaviour across answers has
    to be enforced here — never by hoping the model agrees with itself.
    """
    final, notes = d.intent, []
    if final in ("lift", "carry", "lower") and d.grasp_secure_prob < gate_grasp:
        final = "grasp" if d.grasp_secure_prob >= 0.30 else "hold"
        notes.append(f"grasp_secure={d.grasp_secure_prob:.2f} < {gate_grasp} -> 不许搬动")
    if d.intent_confidence < gate_confidence:
        notes.append(f"intent_confidence={d.intent_confidence:.2f} < {gate_confidence} -> 原地保持")
        final = "hold"
    if d.progress_score >= 3.0 and final not in ("retreat", "finish", "hold"):
        final = "retreat"
        notes.append("progress=3 -> 任务已完成，退开")
    return final, "; ".join(notes)


PROGRESS_EPS = 0.05   # 进度估计变化小于这个数视为"毫无变化"


def stale_channels(intent: str, ages: dict, stale_limit: float) -> dict:
    """该技能依赖的通道里，哪些读数已经旧过 stale_limit。"""
    return {ch: ages[ch] for ch in SKILL_NEEDS.get(intent, ()) if ages[ch] > stale_limit}


def watchdog_trip(history: list[dict], intent: str, progress_now: float) -> bool:
    """卡死看门狗：同一动作连续要第三次，且判断层自己的进度估计毫无变化。

    "说谎"的传感器能靠通道间矛盾发现，"冻结"的只能靠时钟和重复：模型看不见
    世界变化时，它的进度估计会停在原地。"毫无变化"这半边必须对照 history 里
    记录的每轮进度来查（修复前只查了动作重复，与注释承诺不符）。
    """
    if intent == "hold" or len(history) < 2:
        return False
    if [h["skill"] for h in history[-2:]] != [intent, intent]:
        return False
    return all(abs(h["progress"] - progress_now) < PROGRESS_EPS for h in history[-2:])


def write_png(path: Path, rgb) -> None:
    """Minimal PNG writer, so rendering needs no extra dependency."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    blob = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


class LapTimer:
    """Keep simulated time roughly equal to wall time while a viewer is open."""

    def __init__(self, lab: ArmLab):
        self.lab = lab
        self.wall = time.time()
        self.sim = lab.d.time

    def __call__(self) -> None:
        ahead = (self.lab.d.time - self.sim) - (time.time() - self.wall)
        if ahead > 0:
            time.sleep(min(ahead, 0.05))
        self.wall, self.sim = time.time(), self.lab.d.time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="xArm7 + typed-judgment pick-and-place lab")
    ap.add_argument("--jev-mode", default="auto", choices=["auto", "fake", "live"])
    ap.add_argument("--cycles", type=int, default=24, help="max decision cycles")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--log", default="logs/run.jsonl")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer window")
    ap.add_argument("--no-realtime", action="store_true", help="do not throttle to real time")
    ap.add_argument("--frames", default=None, help="directory for PNG snapshots")
    ap.add_argument("--frame-every", type=int, default=3)
    # 默认门槛按实测标定：真实 Jev 在 9 选项下 intent 置信度只有 0.35-0.64，
    # grasp_secure 概率 0.47-0.64。用 0.55/0.70 会把几乎所有提议都否决掉（实测 24 轮否决 19-20 轮，
    # 机械臂原地不动）。见 README「实测结果」。
    ap.add_argument("--randomize", action="store_true",
                    help="随机化方块起始位置（批量标定用）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子，便于复现某个场景")
    ap.add_argument("--quiet", action="store_true", help="批量跑时不要逐轮打印")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--stale-limit", type=float, default=1.0,
                    help="传感数据超过这个年龄（仿真秒）就不许动，交给人处理")
    ap.add_argument("--gate-confidence", type=float, default=0.30)
    ap.add_argument("--gate-grasp", type=float, default=0.45)
    ap.add_argument("--stress", default="none",
                    help="注入哪种难例，见 jev_arm/stress.py（none/grasp_off/tactile_dead/...）")
    args = ap.parse_args(argv)

    stress = get_stress(args.stress)   # 先校验名字，别等仿真都建好了才失败
    rng = np.random.default_rng(args.seed)
    lab = ArmLab()
    lab.stress, lab.rng = stress, rng
    scene_info = {"cube_start": [round(float(v), 4) for v in lab.object_pose()],
                  "target_xy": [float(TARGET_XY[0]), float(TARGET_XY[1])],
                  "seed": args.seed, "stress": stress.name}
    if args.randomize:
        lab.randomize_object(rng)
        scene_info["cube_start"] = [round(float(v), 4) for v in lab.object_pose()]

    # Sensor layer: the judge reads published (and timestamped) channel values, not the
    # simulation directly. This is what makes "this reading is 8 seconds old" expressible,
    # and it is where the stressors reconfigure the world the judge is told about.
    sensors = SensorSuite(rng=rng)
    stress.configure(sensors)
    sensors.prime(lab, lab.d.time)

    judge, label = make_judge(args.jev_mode)
    if not args.quiet:
        print(f"judge: {label}")
        print(f"instruction: {args.instruction}")
        print(f"object starts at {[round(float(v), 3) for v in lab.object_pose()]}, "
              f"target at ({TARGET_XY[0]}, {TARGET_XY[1]})")

    viewer, viewer_syncs = None, 0
    pace = None if args.no_realtime else LapTimer(lab)
    if args.viewer:
        import mujoco.viewer as mj_viewer  # imported here so headless runs need no window
        viewer = mj_viewer.launch_passive(lab.m, lab.d)
    else:
        pace = None          # headless: run as fast as the CPU allows

    def hook() -> None:
        """Per-step callback: publish sensors, pace to real time, refresh the window.

        The viewer half matters too: without viewer.sync() the window shows the initial
        pose forever while the physics runs on, which looks exactly like a frozen robot.
        """
        nonlocal viewer_syncs
        sensors.update(lab, lab.d.time)
        if pace is not None:
            pace()
        if viewer is not None and viewer.is_running():
            viewer.sync()
            viewer_syncs += 1

    lab.tick_hook = hook

    renderer, camera = None, None
    if args.frames:
        renderer = mujoco.Renderer(lab.m, height=720, width=1280)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.32, 0.0, 0.22]
        camera.distance, camera.azimuth, camera.elevation = 1.05, 145.0, -18.0

    def snapshot(path: Path) -> None:
        renderer.update_scene(lab.d, camera=camera)
        write_png(path, renderer.render())

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    overrides = 0
    done = False
    escalated = False

    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"meta": True, "scene": scene_info, "jevmode": args.jev_mode,
                              "stress": stress.name,
                              "gates": [args.gate_confidence, args.gate_grasp]},
                             ensure_ascii=False) + "\n")
        if renderer:
            snapshot(Path(args.frames) / "cycle_000.png")
        for cycle in range(args.cycles):
            state = build_state(lab, history, args.instruction, cycle, sensors=sensors)
            # 真值必须在"执行技能之前"抓：模型的判断针对的是这一刻的状态。
            # 记在执行之后会把 release 这类翻转的动作算错（实测踩过：p=0.8 被判为错）。
            gt_before = lab.grasp_flags()
            decision = judge.decide(state)
            final, note = enforce(decision, args.gate_confidence, args.gate_grasp)
            if note:
                overrides += 1

            # 新鲜度闸门：技能依赖哪条通道，就要求那条通道的数据足够新。
            # 这是"视觉冻结"那个发现的解法——不更新且不报错的传感器，只能靠时钟发现。
            ages = sensors.ages(lab.d.time)
            stale = stale_channels(final, ages, args.stale_limit)
            # 卡死看门狗：同一个动作连着要第三次，而且判断层自己的进度估计毫无变化
            # —— 说明它看不见世界的变化（典型原因：某个通道的数据烂住了）。交给人。
            repeated = watchdog_trip(history, final, decision.progress_score)
            if repeated and not stale:
                escalated = True
                record = {"cycle": cycle, "state": state, "judge": decision.as_log(),
                          "final_intent": final, "override": note,
                          "escalation": {"blocked_intent": final, "reason": "same skill requested 3x with no change",
                                         "progress_estimates": [h["progress"] for h in history[-2:]]
                                                               + [round(decision.progress_score, 3)],
                                         "limit_s": args.stale_limit,
                                         "action": "blocked; ask a human (not making progress)"},
                          "ground_truth": gt_before}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                if not args.quiet:
                    print(f"cycle {cycle:2d} | BLOCKED: {final} 连续第 3 次且无进展 -> 交给人处理")
                break
            if stale:
                escalated = True
                record = {
                    "cycle": cycle, "state": state, "judge": decision.as_log(),
                    "final_intent": final, "override": note,
                    "escalation": {"blocked_intent": final, "stale_channels": stale,
                                   "limit_s": args.stale_limit,
                                   "action": "blocked; ask a human (sensor data too old to act on)"},
                    "ground_truth": gt_before,
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                if not args.quiet:
                    detail = ", ".join(f"{k} {v:.1f}s" for k, v in stale.items())
                    print(f"cycle {cycle:2d} | BLOCKED: {detail} > {args.stale_limit}s "
                          f"-> 不执行 {final}，交给人处理")
                break

            if stress.shove_here(cycle):          # external interference, real physics
                lab.shove(stress.shove_velocity)
            ctx = {"object": lab.object_pose()}
            result = SKILLS[final](lab, ctx)
            history.append({"skill": final, "note": result["note"], "tcp_err_m": result["tcp_err_m"],
                            "progress": round(decision.progress_score, 3)})

            gt = lab.grasp_flags()
            record = {
                "cycle": cycle,
                "state": state,
                "judge": decision.as_log(),
                "final_intent": final,
                "override": note,
                "result": result,
                "ground_truth": gt_before,
                "ground_truth_after": gt,
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()

            if not args.quiet:
                print(f"cycle {cycle:2d} | judge={decision.source:8s} {decision.intent:8s}"
                  f" p={decision.intent_confidence:.2f} grasp={decision.grasp_secure_prob:.2f}"
                  f" prog={decision.progress_score:.1f} -> {final:8s}"
                  f" | obj={[round(float(v), 3) for v in lab.object_pose()]}"
                  f" gt_secure={gt['grasp_secure']} in_target={gt['in_target']}"
                  f" | age cam={ages['camera']:.2f}s tac={ages['tactile']:.2f}s"
                  + (f" | VETO: {note}" if note else ""))

            if renderer and (cycle + 1) % args.frame_every == 0:
                snapshot(Path(args.frames) / f"cycle_{cycle + 1:03d}.png")

            if gt["in_target"] and lab.gripper_width() > 0.07:
                done = True
                if not args.quiet:
                    print("object is in the target zone and released -> done")
                break

    if viewer:
        print(f"viewer: refreshed the window {viewer_syncs} times "
              f"({'window was closed early' if not viewer.is_running() else 'ok'})")
        viewer.close()
    obj = lab.object_pose()
    err = float(((obj[:2] - TARGET_XY) ** 2).sum() ** 0.5)
    outcome = ("DONE" if done else
               "BLOCKED (sensor data stale or not changing -> asked a human)" if escalated else
               "STOPPED after cycle limit")
    if not args.quiet:
        print(f"\n{outcome} | "
              f"object at {[round(float(v), 4) for v in obj]} | horizontal error {err * 1000:.1f} mm | "
              f"code vetoed {overrides} proposal(s) | log: {log_path}")
    # 0 = 完成, 2 = 拦住并升级给人, 1 = 跑到轮数上限
    return 0 if done else (2 if escalated else 1)


if __name__ == "__main__":
    sys.exit(main())
