import numpy as np
from jev_arm.sim import ArmLab
from jev_arm import skills
lab = ArmLab()
for name in ["approach", "grasp", "lift", "carry", "lower", "release", "retreat", "finish"]:
    r = skills.SKILLS[name](lab, {"object": lab.object_pose()})
    gt = lab.grasp_flags()
    print(f"{name:9s} obj={np.round(lab.object_pose(),3)} gap={lab.gripper_width()*1000:5.1f}mm "
          f"tcp_err={r['tcp_err_m']} gt: secure={gt['grasp_secure']} lifted={gt['lifted']} "
          f"in_target={gt['in_target']} attached={gt['attached']}")
    print(f"          note: {r['note']}")
