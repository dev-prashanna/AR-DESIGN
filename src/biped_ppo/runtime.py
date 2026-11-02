"""Sensor-to-position policy runtime shared with the simulator's controller.

No hardware I/O is performed. Joint/IMU calibration and the physical transport
must be supplied for the actual robot before using these target angles.
"""

from pathlib import Path
import numpy as np
import torch
from .control import MotorTargetController


class PositionPolicyRuntime:
    def __init__(self, checkpoint):
        from .config import ROOT, RunConfig, read
        from .evaluation import inference_runner

        torch.set_num_threads(1)
        self.session = None
        self.checkpoint = Path(checkpoint).resolve()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.cfg = RunConfig(**state["project"]["config"])
        self.spec = read(self.cfg.robot_path)
        self.task = read(self.cfg.task_path)
        if not self.task.get("motor_target_filter"):
            raise ValueError("This runtime requires the filtered transfer-policy profile")
        audit = read_audit(ROOT / self.spec["audit_path"])
        limits = {j["name"]: j["limits"] for j in audit["joints"] if j["type"] == "revolute"}
        self.lower = [float(limits[n]["lower"]) for n in self.spec["joint_names"]]
        self.upper = [float(limits[n]["upper"]) for n in self.spec["joint_names"]]
        self.runner = inference_runner(self.checkpoint)
        self.reset()

    @classmethod
    def from_export(cls, directory):
        """Load just ONNX, controller metadata and CPU runtime sources."""
        import json
        import onnxruntime as ort

        directory = Path(directory)
        metadata = json.loads((directory / "deployment.json").read_text())
        if not metadata.get("motor_target_controller"):
            raise ValueError("Export does not include the filtered controller contract")
        self = cls.__new__(cls)
        self.spec = metadata["robot"]
        self.task = metadata["motor_target_controller"]
        limits = metadata["actuator_limits"]
        self.lower = [float(limits[n]["lower"]) for n in self.spec["joint_names"]]
        self.upper = [float(limits[n]["upper"]) for n in self.spec["joint_names"]]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "policy.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.runner = None
        self.reset()
        return self

    def reset(self):
        self.motor = MotorTargetController(
            torch.tensor([self.spec["nominal"]]),
            self.spec["action_scale"],
            self.lower,
            self.upper,
            self.task,
            self.spec["policy_dt"],
        )
        self.previous = np.zeros(6, dtype=np.float32)
        self.time = 0.0

    def step(self, joint_position, joint_velocity, body_gyro, projected_gravity, command):
        arrays = [
            np.asarray(x, dtype=float)
            for x in (joint_position, joint_velocity, body_gyro, projected_gravity, command)
        ]
        if [x.shape for x in arrays] != [(6,), (6,), (3,), (3,), (3,)]:
            raise ValueError(
                "Expected six joint positions/velocities and three gyro/gravity/command values"
            )
        if not all(np.isfinite(x).all() for x in arrays):
            raise ValueError("Sensor and command values must be finite")
        q, qd, gyro, gravity, requested = arrays
        if not 0.8 <= np.linalg.norm(gravity) <= 1.2:
            raise ValueError(
                "Projected gravity must be a body-frame unit vector, not acceleration in m/s^2"
            )
        observation = policy_observation(
            self.spec, self.task, self.motor, self.previous, q, qd, gyro, gravity, self.time
        )
        obs = torch.from_numpy(observation).unsqueeze(0)
        if self.session is not None:
            latent = torch.from_numpy(
                self.session.run(None, {self.session.get_inputs()[0].name: observation[None]})[0]
            )
        else:
            from tensordict import TensorDict

            with torch.inference_mode():
                latent = self.runner.alg.actor(
                    TensorDict({"actor": obs, "critic": obs}, batch_size=[1])
                )
        target = self.motor.advance(
            latent,
            torch.as_tensor(requested, dtype=torch.float32).unsqueeze(0),
            torch.tensor([self.time]),
        )
        self.previous = np.tanh(latent.numpy()[0])
        self.time += self.spec["policy_dt"]
        return target[0].numpy().copy()


def read_audit(path):
    import json

    return json.loads(Path(path).read_text())


def policy_observation(spec, task, motor, previous, q, qd, gyro, gravity, time):
    """One sensor encoding for the viewer and the physical runtime interface."""
    phase = 2 * np.pi * task["gait_frequency_hz"] * time
    return np.concatenate(
        (
            np.asarray(gyro) * 0.25,
            gravity,
            np.asarray(q) - spec["nominal"],
            np.asarray(qd) * 0.05,
            previous,
            motor.command[0].numpy(),
            [np.sin(phase), np.cos(phase)],
            motor.observation()[0].numpy(),
        )
    ).astype("float32")
