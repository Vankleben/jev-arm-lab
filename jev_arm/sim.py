"""MuJoCo wrapper for the xArm7 pick-and-place lab.

Arm: Menagerie `ufactory_xarm7`, driven by its position-servo actuators (ctrl 0..6 = joint
angles in radians).

Gripper: the lab uses a **parallel-jaw hand** (`xarm7_lab.xml`), not Menagerie's linkage
hand. Measured reason: the linkage model is back-drivable — squeezing a 6 cm cube developed
85-265 N of contact force from an actuator producing only ~14 N, and under a lift the cube
wedged the fingers open, so nothing could ever be carried. A prismatic jaw with a force
limit behaves like a real gripper: `set_gap()` commands the clearance, `forcerange` caps
the squeeze, and the fingers cannot be pushed open by the payload.

Ground truth for the decision layer comes from contacts and object motion, not from the
model's opinion: `grasp_flags()` is what we compare the judge's answers against.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .skills import TARGET_XY   # 真值与技能层必须用同一个目标区，不许各写一份

CUBE_GEOM = "cube_geom"
PAD_GEOMS = {"left": ("finger_left_pad",), "right": ("finger_right_pad",)}
CUBE_REST_Z = 0.15       # cube centre height when resting on the table
PAD_FACE_AT_ZERO = 0.005  # pad inner face offset when the slide joint is at 0 (metres)
MAX_GAP = 0.11            # 2 * (PAD_FACE_AT_ZERO + slide range 0.05)
GRIP_KP = 900.0           # position-actuator gain of the jaw (N/m per finger)
SLIP_TOLERANCE_MPS = 0.05  # object-vs-hand relative speed above which a grasp is sliding


class ArmLab:
    def __init__(self, scene: str = "models/menagerie_xarm7/lab_pick_place.xml"):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self._ids()
        self._calibrate_tcp()
        self.home_arm = self.m.qpos0[:7].copy()
        self.tick_hook = None  # optional callback after each physics step (viewer sync)
        self.ramp = 24         # move_tcp sub-targets; tuned end-to-end (0.4 mm median placement,
                               # vs 11 mm at a single step) — see tools/diagnose_slip.py
        self.reset()

    # ---------------------------------------------------------------- setup
    def _ids(self) -> None:
        m = self.m
        self.site_tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self.geom_cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        self.pads = {
            side: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
            for side, names in PAD_GEOMS.items()
        }
        self.pad_half_y = float(m.geom_size[self.pads["left"][0]][1])
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_joint = jid
        self.cube_qadr = m.jnt_qposadr[jid]
        self.stress = None          # jev_arm.stress.Stress, set by main; None = no stress
        self.rng = np.random.default_rng(0)
        self.tcp_body = m.site_bodyid[self.site_tcp]
        self.grip_acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                          for n in ("grip_left", "grip_right")]
        self.grip_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_left")
        self.grip_qadr = m.jnt_qposadr[self.grip_joint]

    def _calibrate_tcp(self) -> None:
        """Put the TCP site at the centre of the two finger pads (the grasp point)."""
        d, m = self.d, self.m
        d.qpos[:] = m.qpos0
        self.set_gap(MAX_GAP * 0.8)
        mujoco.mj_forward(m, d)
        pads = [self.pads["left"][0], self.pads["right"][0]]
        centre_world = np.mean([d.geom_xpos[i] for i in pads], axis=0)
        rot = d.xmat[self.tcp_body].reshape(3, 3)
        m.site_pos[self.site_tcp] = rot.T @ (centre_world - d.xpos[self.tcp_body])
        mujoco.mj_forward(m, d)
        self.home_mat = d.site_xmat[self.site_tcp].reshape(3, 3).copy()

    # ------------------------------------------------------------- primitives
    READY_TCP = (0.30, 0.0, 0.32)   # clear of the table: the folded home pose puts the
                                    # pads *inside* the table (they extend 45 mm below the TCP)

    def ready(self) -> None:
        """Servo to a clear pose above the table."""
        self.open_gripper()
        q, _ = self.ik(self.READY_TCP)
        self.set_arm_target(q)
        self.settle(1500)

    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.ctrl[:7] = self.home_arm
        self.set_gap(MAX_GAP * 0.8)
        self.settle(600)
        self.ready()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            if self.tick_hook is not None:
                self.tick_hook()

    def settle(self, max_steps: int = 1500, tol: float = 0.02) -> int:
        """Step until everything stops moving (arm joints *and* gripper fingers)."""
        for i in range(max_steps):
            mujoco.mj_step(self.m, self.d)
            if self.tick_hook is not None:
                self.tick_hook()
            if i > 50 and np.max(np.abs(self.d.qvel)) < tol:
                return i
        return max_steps

    def set_arm_target(self, q) -> None:
        self.d.ctrl[:7] = np.clip(q, self.m.jnt_range[:7, 0], self.m.jnt_range[:7, 1])

    # ------------------------------------------------------------- gripper
    @staticmethod
    def _gap_to_ctrl(gap_m: float) -> float:
        return float(np.clip(gap_m / 2.0 - PAD_FACE_AT_ZERO, 0.0, 0.05))

    def set_gap(self, gap_m: float) -> None:
        """Command the clearance between the pads (metres). Force is capped by forcerange."""
        for act in self.grip_acts:
            self.d.ctrl[act] = self._gap_to_ctrl(gap_m)

    def commanded_gap(self) -> float:
        return 2.0 * (float(self.d.ctrl[self.grip_acts[0]]) + PAD_FACE_AT_ZERO)

    def open_gripper(self, gap_m: float = MAX_GAP * 0.8) -> None:
        self.set_gap(gap_m)
        self.settle(500)

    def close_gripper(self, object_width_m: float = 0.06, grip_force_n: float = 8.0) -> float:
        """Close onto an object with a target grip force.

        The jaw is a position servo (kp = 900 N/m per finger), so the force comes from
        commanding a gap *narrower* than the object: interference = force / kp. Commanding
        only a few millimetres of interference produced a measured 4 N pinch, which slipped
        under load; 15 N gives ~18 N of friction per side. `forcerange` caps the squeeze at
        30 N per finger, so an over-command cannot crush the payload.
        """
        interference = grip_force_n / GRIP_KP
        self.set_gap(max(0.01, object_width_m - 2.0 * interference))
        self.settle(700)
        self.settle(400)   # let the contact settle before anything moves
        return self.gripper_width()

    def gripper_width(self) -> float:
        """Current clearance between the pad faces, in metres."""
        left = np.mean([self.d.geom_xpos[i] for i in self.pads["left"]], axis=0)
        right = np.mean([self.d.geom_xpos[i] for i in self.pads["right"]], axis=0)
        return float(np.linalg.norm(left - right) - 2.0 * self.pad_half_y)

    # ------------------------------------------------------------ grasp hold
    # Carrying is *physical*: the pads squeeze the cube and friction does the rest. There is
    # no attach; the object is never teleported. Getting here needed two fixes, both measured
    # with tools/diagnose_slip.py: (1) ik() used to write its solution into qpos, so every
    # Cartesian move teleported the arm and the pads never actually travelled past the cube;
    # (2) that artifact made contact-friction carrying look impossible ("gripped but never
    # lifted"). With the solver kept pure, an 8 N pinch carries the cube 148 mm with 3 mm of
    # transient slip at baseline parameters — no friction or contact tweaks required.

    def slip_speed(self) -> float:
        """Relative speed between the object and the TCP — nonzero means it is sliding."""
        m, d = self.m, self.d
        dof0 = m.jnt_dofadr[self.cube_joint]
        v_obj = d.qvel[dof0:dof0 + 3]
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.site_tcp)
        return float(np.linalg.norm(v_obj - jacp @ d.qvel))

    def randomize_object(self, rng, x_range=(0.31, 0.43), y_range=(0.06, 0.20),
                         yaw_range=(-0.15, 0.15)) -> None:
        """Place the object at a random spot on the table (batch scenes + faithful replay)."""
        x = float(rng.uniform(*x_range))
        y = float(rng.uniform(*y_range))
        yaw = float(rng.uniform(*yaw_range))
        self.d.qpos[self.cube_qadr:self.cube_qadr + 3] = [x, y, CUBE_REST_Z]
        self.d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        self.settle(600)

    def shove(self, velocity: float, direction=(0.0, 1.0, 0.0)) -> None:
        """Push the object: external interference, e.g. someone bumping the table."""
        d0 = self.m.jnt_dofadr[self.cube_joint]
        self.d.qvel[d0:d0 + 3] += np.asarray(direction, dtype=float) * velocity

    def object_clearance(self) -> float:
        """Height of the object's underside above the table (0 when it is resting on it)."""
        obj = self.object_pose()
        return float(obj[2] - self.m.geom_size[self.geom_cube][2] - 0.12)

    def object_width(self) -> float:
        """Longest horizontal extent of the object (the cube is a box)."""
        size = self.m.geom_size[self.geom_cube]
        return float(2.0 * max(size[0], size[1]))

    # -------------------------------------------------------------------- IK
    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.site_tcp].copy()

    def tcp_mat(self) -> np.ndarray:
        return self.d.site_xmat[self.site_tcp].reshape(3, 3).copy()

    def ik(self, target_pos, target_mat=None, iters: int = 800, damp: float = 1e-2,
           step: float = 0.5, tol: float = 5e-4) -> tuple[np.ndarray, float]:
        """Damped least-squares IK on the TCP site. Returns (joint targets, residual).

        Pure solver: it restores the arm's qpos before returning. It used to leave the
        solution in d.qpos, i.e. every move_tcp *teleported* the arm to the target pose and
        only then let the servo hold it. That went unnoticed while carrying was kinematic;
        with real contacts it means the fingers jump past the object instead of lifting it
        (measured: 16 N pinch, cube left on the table, 0 mm rise — see tools/diagnose_slip.py).

        The orientation error must be expressed in the same (world) frame as the
        Jacobian's rotational rows: R_err = R_target @ R_current^T.
        """
        m, d = self.m, self.d
        q0 = d.qpos[:7].copy()
        q = q0.copy()
        tgt_mat = self.home_mat if target_mat is None else np.asarray(target_mat)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        err = np.zeros(6)
        q_err = np.zeros(4)
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_forward(m, d)
            err[:3] = np.asarray(target_pos) - d.site_xpos[self.site_tcp]
            cur = d.site_xmat[self.site_tcp].reshape(3, 3)
            mujoco.mju_mat2Quat(q_err, (tgt_mat @ cur.T).flatten())
            mujoco.mju_quat2Vel(err[3:], q_err, 1.0)
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_tcp)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q = np.clip(q + step * dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
        d.qpos[:7] = q
        mujoco.mj_forward(m, d)
        residual = float(np.linalg.norm(np.asarray(target_pos) - d.site_xpos[self.site_tcp]))
        d.qpos[:7] = q0                      # restore: the servo, not the solver, moves the arm
        mujoco.mj_forward(m, d)
        return q, residual

    def move_tcp(self, target_pos, target_mat=None, tol: float = 0.002, iters: int = 4,
                 ramp: int | None = None) -> float:
        """Servo the TCP to a Cartesian pose, correcting for the steady-state sag.

        The move is ramped — a linear interpolation over `ramp` sub-targets (default
        `self.ramp`) — instead of one step command, because a step makes the position servos
        accelerate hard and momentarily exceeds the friction cone of the grasp, letting the
        payload slide inside the fingers. The position servos settle a few millimetres away
        from the commanded pose under gravity (measured up to 8 mm), which is enough to spoil
        a 60 mm cube grasp; the sag is repeatable, so commanding `target + (target - measured)`
        twice or thrice closes it.
        """
        ramp = self.ramp if ramp is None else ramp
        target = np.asarray(target_pos, dtype=float)
        cmd = target.copy()
        for _ in range(iters):
            start = self.tcp_pos()
            for k in range(1, ramp + 1):
                waypoint = start + (cmd - start) * (k / ramp)
                q, _ = self.ik(waypoint, target_mat)
                self.set_arm_target(q)
                self.settle()
            err = target - self.tcp_pos()
            if float(np.linalg.norm(err)) < tol:
                break
            cmd = cmd + 0.9 * err
        return float(np.linalg.norm(target - self.tcp_pos()))

    # ------------------------------------------------------------ perception
    def object_pose(self) -> np.ndarray:
        return self.d.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def pad_contacts(self) -> dict:
        """Contact count between each finger pad and the object."""
        cube = self.geom_cube
        touch = {"left": 0, "right": 0}
        for i in range(self.d.ncon):
            pair = {int(self.d.contact[i].geom1), int(self.d.contact[i].geom2)}
            if cube not in pair:
                continue
            for side in ("left", "right"):
                if pair & set(self.pads[side]):
                    touch[side] += 1
        return touch

    def grasp_flags(self) -> dict:
        """Ground truth in simulation (what the judge has to infer from state).

        Held = both pads in contact with the object. Carrying is physical now, so a held
        object that slides relative to the hand is *not* secure. The `attached` key is kept
        because the logs and summarize_log read it — it now means "held by contact".
        """
        t = self.pad_contacts()
        obj = self.object_pose()
        held = t["left"] > 0 and t["right"] > 0
        slipping = held and self.slip_speed() > SLIP_TOLERANCE_MPS
        return {
            "left_touch": t["left"] > 0,
            "right_touch": t["right"] > 0,
            "grasp_secure": bool(held and not slipping),
            "slipping": bool(slipping),
            "lifted": bool(obj[2] > CUBE_REST_Z + 0.02),
            "in_target": bool(np.linalg.norm(obj[:2] - TARGET_XY) < 0.05
                              and obj[2] < CUBE_REST_Z + 0.02),
            "attached": held,
            # evidence 用于标定打分：刚合上还没验证过的那一段单独标出来，
            # 否则"问我抓稳没有"会把模型正确的"不确定"算成错误。
            "grasp_evidence": ("proven" if (held and self.object_clearance() > 0.02)
                               else "unproven" if held else "none"),
        }

    def tcp_euler(self) -> np.ndarray:
        """TCP orientation as roll/pitch/yaw in degrees (for the text state)."""
        r = self.tcp_mat()
        pitch = np.degrees(np.arcsin(np.clip(-r[2, 0], -1, 1)))
        roll = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
        yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
        return np.round([roll, pitch, yaw], 1)

    def table_contacts(self) -> int:
        """Contacts involving the table top — a sign the arm path is not clear."""
        tid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        return sum(1 for i in range(self.d.ncon)
                   if tid in (int(self.d.contact[i].geom1), int(self.d.contact[i].geom2)))
