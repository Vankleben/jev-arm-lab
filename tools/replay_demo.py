"""Replay a recorded batch and render a demo — no API calls, no cost.

The batch logs already contain the judge's decision for every cycle, so a scene can be
replayed exactly: rebuild the same randomized scene from its seed, re-execute the recorded
skills in order, and record frames along the way.

    python tools/replay_demo.py --logs "logs/batch_live_20c/*.jsonl" --out out_demo

Outputs (all under --out):
  scene_XXX.gif    one animation per scene
  reel.gif         all scenes, six key frames each, in order
  sheet_1.png / sheet_2.png   contact sheets: one row per scene, six key moments
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_arm.sim import ArmLab                     # noqa: E402
from jev_arm.skills import SKILLS, TARGET_XY       # noqa: E402

W, H = 640, 360          # capture size
TILE = (320, 180)        # size used in gifs and sheets
STEP_EVERY = 40          # capture a frame every 40 physics steps (0.08 s of sim time)


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.32, 0.0, 0.22]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 145.0, -18.0
    return cam


def replay_scene(path: Path, renderer: mujoco.Renderer, cam: mujoco.MjvCamera) -> dict:
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = records[0]["scene"]
    intents = [r["final_intent"] for r in records[1:]]

    lab = ArmLab()
    lab.randomize_object(np.random.default_rng(meta["seed"]))
    start = [round(float(v), 4) for v in lab.object_pose()]

    frames: list[Image.Image] = []
    keys: list[Image.Image] = []

    def grab() -> None:
        renderer.update_scene(lab.d, camera=cam)
        img = Image.fromarray(renderer.render())
        frames.append(img)

    step = 0

    def advance(n: int) -> None:
        nonlocal step
        for _ in range(n):
            lab._carry()
            mujoco.mj_step(lab.m, lab.d)
            step += 1
            if step % STEP_EVERY == 0:
                grab()

    grab()
    keys.append(frames[-1])
    for intent in intents:
        skill = SKILLS.get(intent, SKILLS["hold"])
        # run the skill body, but advance the physics through our capture loop
        real_step, real_settle = lab.step, lab.settle
        lab.step = lambda n=1: advance(n)                                    # type: ignore[method-assign]
        lab.settle = lambda max_steps=1500, tol=0.02: (advance(max_steps) or max_steps)  # type: ignore[method-assign]
        try:
            skill(lab, {"object": lab.object_pose()})
        finally:
            lab.step, lab.settle = real_step, real_settle                    # type: ignore[method-assign]
        lab.step(1)
        grab()
        keys.append(frames[-1])

    final = lab.object_pose()
    err_mm = 1000.0 * float(((final[:2] - TARGET_XY) ** 2).sum() ** 0.5)
    return {"name": path.stem, "seed": meta["seed"], "start": start, "intents": intents,
            "frames": frames, "keys": keys, "err_mm": err_mm,
            "done": bool((records[-1].get("ground_truth_after") or records[-1]["ground_truth"])["in_target"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/batch_live_20c/*.jsonl")
    ap.add_argument("--out", default="out_demo")
    ap.add_argument("--gif-fps", type=float, default=8.0, help="gif frames per second (of sim time)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs))
    if not files:
        print(f"no logs matched {args.logs}")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab0 = ArmLab()
    renderer = mujoco.Renderer(lab0.m, height=H, width=W)
    cam = make_camera()

    sheets: list[list[Image.Image]] = []
    reel: list[Image.Image] = []
    print(f"replaying {len(files)} scenes -> {out}")
    for i, f in enumerate(files):
        info = replay_scene(Path(f), renderer, cam)
        frames = [im.resize(TILE) for im in info["frames"]]
        keys = [im.resize(TILE) for im in info["keys"]]
        # per-scene animation (decimate if long, to keep the files reasonable)
        if len(frames) > 90:
            frames = frames[::2]
        frames[0].save(out / f"{info['name']}.gif", save_all=True,
                       append_images=frames[1:], duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        sheets.append(keys[:6])
        reel.extend(keys[:6])
        print(f"  [{i + 1:2d}/{len(files)}] {info['name']} seed={info['seed']:3d} "
              f"start=({info['start'][0]:.3f},{info['start'][1]:.3f}) "
              f"{len(info['intents'])} decisions | err={info['err_mm']:5.1f} mm | "
              f"{len(info['frames'])} frames | {'OK' if info['done'] else 'FAIL'}")

    # contact sheets: 10 scenes per image, one row per scene
    for part in range(0, len(sheets), 10):
        chunk = sheets[part:part + 10]
        sheet = Image.new("RGB", (TILE[0] * 6, TILE[1] * len(chunk)), (20, 24, 30))
        for row, keys in enumerate(chunk):
            padded = list(keys) + [keys[-1]] * (6 - len(keys))   # short scenes: repeat the last frame
            for col, im in enumerate(padded[:6]):
                sheet.paste(im, (col * TILE[0], row * TILE[1]))
        sheet_path = out / f"sheet_{part // 10 + 1}.png"
        sheet.save(sheet_path)
        print(f"  wrote {sheet_path} ({sheet.size[0]}x{sheet.size[1]})")

    reel[0].save(out / "reel.gif", save_all=True, append_images=reel[1:], duration=420, loop=0, optimize=True)
    print(f"  wrote {out / 'reel.gif'} ({len(reel)} frames, all scenes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
