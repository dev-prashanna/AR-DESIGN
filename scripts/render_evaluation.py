"""Reconstruct recorded states only; never simulate invented success snapshots."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import matplotlib.pyplot as plt
import imageio.v2 as imageio

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation", required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
record = json.loads((args.evaluation / "trajectories.json").read_text())
model = mujoco.MjModel.from_xml_path(str(root / "assets/robot/models/microduck.xml"))
# Plane collision is infinite; extend its visual patch to cover vector-world offsets.
model.geom_size[model.geom("floor").id, :2] = 100.0
model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
model.vis.headlight.diffuse[:] = [0.5, 0.5, 0.5]
model.vis.global_.offwidth = 600
model.vis.global_.offheight = 600
data = mujoco.MjData(model)
spec = json.loads((args.evaluation.parent / "manifest.json").read_text())
robot = json.loads((root / "docs/model_audit.json").read_text())["robot"]
qids = [model.joint(n).qposadr[0] for n in robot["joint_names"]]
selected = [record["cases"][0], next(c for c in record["cases"] if c["case"]["command"][0] > 0.001)]
outputs = []
with mujoco.Renderer(model, height=600, width=600) as renderer:
    for case in selected:
        trajectory = case["trajectory"]
        command = case["case"]["command"]
        camera = mujoco.MjvCamera()
        camera.distance = 0.42
        camera.azimuth = 135
        camera.elevation = -15
        label = "standing" if command[0] == 0 else "forward"
        frames = []
        for state in trajectory:
            data.qpos[:7] = state["root_pose"]
            data.qpos[qids] = state["joint_pos"]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.09]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
        if not frames:
            continue
        path = args.evaluation / f"{label}_recorded.mp4"
        imageio.mimsave(path, frames, fps=10, macro_block_size=1)
        plt.imsave(args.evaluation / f"{label}_final.png", frames[-1])
        outputs.append(
            dict(
                video=str(path),
                frames=len(frames),
                fps=10,
                command=command,
                final_t_s=trajectory[-1]["t_s"],
                terminal=trajectory[-1]["terminal"],
                selection="first seed/friction/replica, first zero or forward command; all outcomes retained",
            )
        )
(args.evaluation / "render_manifest.json").write_text(json.dumps(outputs, indent=2) + "\n")
print(json.dumps(outputs, indent=2))
