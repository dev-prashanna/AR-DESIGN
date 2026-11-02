"""Render every predefined command from saved evaluation trajectories."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
from biped_ppo.config import read
from biped_ppo.robot import generate
from biped_ppo.tracking import sha, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation", type=Path, required=True)
args = parser.parse_args()
record = json.loads((args.evaluation / "trajectories.json").read_text())
config = json.loads((args.evaluation / "summary.json").read_text())["config"]
robot = read(config["robot_path"])
xml, _ = generate(config["robot_path"], ground=True, motors=True)
model = mujoco.MjModel.from_xml_string(xml)
model.geom_size[model.geom("floor").id, :2] = 100
model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 640
data = mujoco.MjData(model)
qids = [model.joint(n).qposadr[0] for n in robot["joint_names"]]
manifest = []
with mujoco.Renderer(model, height=640, width=640) as renderer:
    for index, case in enumerate(record["cases"]):
        camera = mujoco.MjvCamera()
        camera.distance = 0.48
        camera.azimuth = 135
        camera.elevation = -12
        path = args.evaluation / f"command_{index:02d}.mp4"
        with imageio.get_writer(path, fps=10, macro_block_size=1) as writer:
            for state in case["trajectory"]:
                data.qpos[:7] = state["root_pose"]
                data.qpos[qids] = state["joint_pos"]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.12]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                writer.append_data(frame)
        if case["trajectory"]:
            plt.imsave(args.evaluation / f"command_{index:02d}_final.png", frame)
        manifest.append(
            dict(
                video=str(path),
                sha256=sha(path),
                command=case["case"]["command"],
                final_time_s=case["trajectory"][-1]["t_s"],
                terminal=case["trajectory"][-1]["terminal"],
                selection=record["selection_rule"],
            )
        )
write_json(args.evaluation / "video_manifest.json", manifest)
print(args.evaluation / "video_manifest.json")
