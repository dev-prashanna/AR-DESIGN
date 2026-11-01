"""Small direct MuJoCo Warp physics probe; no PPO training or performance claim."""

import json
import time
import mujoco
import mujoco_warp as mjw
import warp as wp
import numpy as np
from biped_ppo.robot import generate
from biped_ppo.config import read
from biped_ppo.diagnostics import initialize

wp.init()
print("CUDA devices:", wp.get_cuda_devices(), flush=True)
xml, _ = generate(ground=True, motors=True)
model = mujoco.MjModel.from_xml_string(xml)
data = initialize(model, read("configs/robot/custom.yaml"))
started = time.monotonic()
with wp.ScopedDevice("cuda:0"):
    wm = mjw.put_model(model)
    wd = mjw.put_data(model, data, nworld=32, nconmax=64, njmax=128)
    for _ in range(5):
        mjw.step(wm, wd)
    wp.synchronize()
    qpos = wd.qpos.numpy()
result = dict(
    worlds=len(qpos),
    steps=5,
    finite=bool(np.isfinite(qpos).all()),
    min_root_z_m=float(qpos[:, 2].min()),
    wall_s=time.monotonic() - started,
)
print(json.dumps(result), flush=True)
