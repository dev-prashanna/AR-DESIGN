import json
import os
import platform
import shutil
import subprocess
import time
from importlib.metadata import version
import numpy as np


def doctor():
    import torch
    import psutil

    result = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        platform=platform.platform(),
        cpu=platform.processor(),
        cpu_details=json.loads(
            subprocess.run(["lscpu", "-J"], capture_output=True, text=True, check=True).stdout
        ),
        ram_bytes=psutil.virtual_memory().total,
        available_ram_bytes=psutil.virtual_memory().available,
        cuda_available=torch.cuda.is_available(),
        torch_cuda=torch.version.cuda,
        packages={
            p: version(p)
            for p in ["mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch"]
        },
    )
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    result["nvidia_smi"] = (
        subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        if shutil.which("nvidia-smi")
        else None
    )
    return result


def initialize(model, cfg, height=None):
    import mujoco

    d = mujoco.MjData(model)
    d.qpos[:3] = [0, 0, cfg["base_height"] if height is None else height]
    d.qpos[3:7] = [1, 0, 0, 0]
    for n, q in zip(cfg["joint_names"], cfg["nominal"]):
        d.qpos[model.joint(n).qposadr[0]] = q
    mujoco.mj_forward(model, d)
    return d


def physics_checks(model, cfg):
    import mujoco

    d = initialize(model, cfg)
    qids = [model.joint(n).qposadr[0] for n in cfg["joint_names"]]
    vids = [model.joint(n).dofadr[0] for n in cfg["joint_names"]]
    aid = [model.actuator(n + "_motor").id for n in cfg["joint_names"]]
    limits = model.actuator_ctrlrange[aid, 1]
    report = dict(
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        com_world_m=d.subtree_com[1].tolist(),
        initial_contacts=[
            dict(
                geom1=model.geom(c.geom1).name,
                geom2=model.geom(c.geom2).name,
                distance_m=float(c.dist),
            )
            for c in d.contact
        ],
        root_free=bool(model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE),
    )
    collision_ids = [i for i in range(model.ngeom) if "collision" in model.geom(i).name]
    bounds = []
    for index in collision_ids:
        extent = np.abs(d.geom_xmat[index].reshape(3, 3)) @ model.geom_size[index]
        bounds.append((d.geom_xpos[index] - extent, d.geom_xpos[index] + extent))
    report["height_m"] = float(max(b[1][2] for b in bounds) - min(b[0][2] for b in bounds))
    report["foot_contact_sites"] = ["l_foot_contact", "r_foot_contact"]
    report["nominal_support_xy_m"] = [
        [-0.019, -0.051],
        [0.019, -0.051],
        [0.019, 0.051],
        [-0.019, 0.051],
    ]
    report["joint_command_checks"] = []
    # Floating robot in the air avoids floor friction masking sign checks.
    for n, qi, vi, ai in zip(cfg["joint_names"], qids, vids, aid):
        directions = []
        for sign in [-1, 1]:
            tmp = initialize(model, cfg, 0.4)
            # At upper joint limit (knees), positive torque is limit-constrained.
            tmp.qpos[qi] = np.mean(model.joint(n).range)
            tmp.ctrl[ai] = sign * 0.02
            mujoco.mj_forward(model, tmp)
            directions.append(float(tmp.qacc[vi]))
        report["joint_command_checks"].append(
            dict(
                joint=n,
                negative_acceleration=directions[0],
                positive_acceleration=directions[1],
                passed=directions[0] < 0 < directions[1],
            )
        )
    report["standing_trace"] = []
    steps = round(cfg["standing_gate_seconds"] / model.opt.timestep)
    min_height = 1.0
    saturation = 0
    finite = True
    for step in range(steps):
        tau = (
            np.array(cfg["kp"]) * (np.array(cfg["nominal"]) - d.qpos[qids])
            - np.array(cfg["kd"]) * d.qvel[vids]
        )
        saturation += np.count_nonzero(np.abs(tau) >= limits)
        d.ctrl[aid] = np.clip(tau, -limits, limits)
        mujoco.mj_step(model, d)
        min_height = min(min_height, float(d.qpos[2]))
        finite = finite and bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if step % 100 == 0:
            report["standing_trace"].append(
                [step * model.opt.timestep, float(d.qpos[2]), *d.qpos[3:7].tolist()]
            )
    report["standing"] = dict(
        seconds=steps * model.opt.timestep,
        min_height_m=min_height,
        final_height_m=float(d.qpos[2]),
        finite=finite,
        saturation_fraction=saturation / (steps * 6),
        passed=finite and min_height >= cfg["standing_gate_min_height"],
    )
    report["standing_contacts"] = [
        dict(
            geom1=model.geom(c.geom1).name, geom2=model.geom(c.geom2).name, distance_m=float(c.dist)
        )
        for c in d.contact
    ]
    drop = initialize(model, cfg, 0.3)
    min_z = 0.3
    for _ in range(500):
        mujoco.mj_step(model, drop)
        min_z = min(min_z, float(drop.qpos[2]))
    report["gravity_drop"] = dict(
        start_height_m=0.3,
        min_height_m=min_z,
        final_height_m=float(drop.qpos[2]),
        finite=bool(np.isfinite(drop.qpos).all()),
    )
    report["gates_passed"] = bool(
        report["root_free"]
        and report["standing"]["passed"]
        and all(c["passed"] for c in report["joint_command_checks"])
        and report["gravity_drop"]["finite"]
    )
    return report


def render_model(model, cfg, path):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import matplotlib.pyplot as plt

    d = initialize(model, cfg)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.095]
    camera.distance = 0.4
    camera.azimuth = 135
    camera.elevation = -15
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 800
    with mujoco.Renderer(model, height=800, width=800) as renderer:
        renderer.update_scene(d, camera=camera)
        plt.imsave(path, renderer.render())
