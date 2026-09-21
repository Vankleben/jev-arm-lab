"""Dump the ids/names we need: joints, actuators, finger pads, gripper body."""
import mujoco, sys

path = sys.argv[1] if len(sys.argv) > 1 else "models/menagerie_xarm7/scene.xml"
m = mujoco.MjModel.from_xml_path(path)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
print("\n-- joints --")
for i in range(m.njnt):
    print(f"  {i:2d} {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i):>22s} type={m.jnt_type[i]} qposadr={m.jnt_qposadr[i]}")
print("\n-- actuators --")
for i in range(m.nu):
    print(f"  {i:2d} {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i):>20s} ctrlrange={m.actuator_ctrlrange[i]}")
print("\n-- bodies of interest --")
for n in ("link7", "xarm_gripper_base_link", "left_finger", "right_finger"):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    print(f"  {n:>24s} id={i} xpos={d.xpos[i] if i>=0 else None}")
print("\n-- pad geoms --")
for n in ("left_finger_pad_1", "left_finger_pad_2", "right_finger_pad_1", "right_finger_pad_2"):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    print(f"  {n:>20s} id={i} xpos={d.geom_xpos[i].round(4) if i>=0 else None}")
print("\n-- sites --")
print("  ", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)])
