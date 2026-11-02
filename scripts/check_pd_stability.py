"""Reproduce the airborne PD stability diagnostic with and without rotor inertia."""

import json
import numpy as np
import mujoco
from biped_ppo.config import ROOT, read
from biped_ppo.robot import generate
from biped_ppo.diagnostics import initialize
from biped_ppo.tracking import write_json

cfg = read("configs/robot/futuristic.yaml")
xml, _ = generate("configs/robot/futuristic.yaml", motors=True)
rows = []
for armature in (0, 0.0001):
    model = mujoco.MjModel.from_xml_string(xml)
    model.dof_armature[6:] = armature
    data = initialize(model, cfg, height=0.8)
    qids = [model.joint(n).qposadr[0] for n in cfg["joint_names"]]
    vids = [model.joint(n).dofadr[0] for n in cfg["joint_names"]]
    target = data.qpos[qids].copy() + [0.02, 0.03, -0.03, -0.02, -0.03, 0.03]
    peak = 0.0
    saturated = 0
    for _ in range(250):
        torque = (
            np.array(cfg["kp"]) * (target - data.qpos[qids]) - np.array(cfg["kd"]) * data.qvel[vids]
        )
        saturated += int((np.abs(torque) > model.actuator_ctrlrange[:, 1]).sum())
        data.ctrl[:] = np.clip(
            torque, -model.actuator_ctrlrange[:, 1], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)
        peak = max(peak, float(np.abs(data.qvel[vids]).max()))
    rows.append(
        dict(
            armature_kg_m2=armature,
            max_joint_speed_rad_s=peak,
            saturated_joint_steps=saturated,
            final_max_error_rad=float(np.abs(data.qpos[qids] - target).max()),
        )
    )
result = dict(
    test="Airborne 0.5 s, 0.02/0.03 rad target perturbations, fixed explicit PD at 500 Hz",
    results=rows,
    assumption="Reflected rotor inertia is not hardware-calibrated",
)
write_json(ROOT / "docs/futuristic_pd_diagnostic.json", result)
print(json.dumps(result, indent=2))
