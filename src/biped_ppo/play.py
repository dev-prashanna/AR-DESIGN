"""Native MuJoCo WASD viewer and repeatable headless command playback."""

from pathlib import Path
import time
import numpy as np
import torch
import mujoco
from tensordict import TensorDict
from .config import RunConfig, read
from .robot import generate
from .diagnostics import initialize
from .evaluation import inference_runner
from .tracking import write_json, sha


def keyboard_command(pressed, speed=0.05, turn=0.25):
    """W/S translate; A/D turn; space stops; releasing keys stops."""
    if "SPACE" in pressed:
        return np.zeros(3)
    return np.array(
        [
            speed * (("W" in pressed) - ("S" in pressed)),
            0.0,
            turn * (("A" in pressed) - ("D" in pressed)),
        ]
    )


class PolicySimulation:
    def __init__(self, checkpoint):
        torch.set_num_threads(1)
        self.checkpoint = Path(checkpoint).resolve()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.cfg = RunConfig(**state["project"]["config"])
        self.spec = read(self.cfg.robot_path)
        self.task = read(self.cfg.task_path)
        xml, _ = generate(self.cfg.robot_path, ground=True, motors=True)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.geom_size[self.model.geom("floor").id, :2] = 100
        self.model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
        self.model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        self.runner = inference_runner(self.checkpoint)
        names = self.spec["joint_names"]
        self.qids = [self.model.joint(n).qposadr[0] for n in names]
        self.vids = [self.model.joint(n).dofadr[0] for n in names]
        self.aids = [self.model.actuator(n + "_motor").id for n in names]
        self.limits = np.array([self.model.joint(n).range for n in names])
        self.efforts = self.model.actuator_ctrlrange[self.aids, 1]
        self.root = self.model.body("torso").id
        self.reset()

    def reset(self):
        self.data = initialize(self.model, self.spec)
        self.motor = None
        if self.task.get("motor_target_filter"):
            from .control import MotorTargetController

            self.motor = MotorTargetController(
                torch.tensor([self.spec["nominal"]]),
                self.spec["action_scale"],
                self.limits[:, 0],
                self.limits[:, 1],
                self.task,
                self.spec["policy_dt"],
            )
        self.previous = np.zeros(6)
        self.target = np.array(self.spec["nominal"])

    def observation(self, command):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.root, velocity, 1
        )
        rotation = self.data.xmat[self.root].reshape(3, 3)
        if self.motor is not None:
            from .runtime import policy_observation

            return policy_observation(
                self.spec,
                self.task,
                self.motor,
                self.previous,
                self.data.qpos[self.qids],
                self.data.qvel[self.vids],
                velocity[:3],
                rotation.T @ np.array([0, 0, -1]),
                self.data.time,
            )
        fields = [
            velocity[:3] * 0.25,
            rotation.T @ np.array([0, 0, -1]),
            self.data.qpos[self.qids] - self.spec["nominal"],
            self.data.qvel[self.vids] * 0.05,
            self.previous,
            self.motor.command[0].numpy() if self.motor is not None else command,
        ]
        if "gait_frequency_hz" in self.task:
            phase = 2 * np.pi * self.task["gait_frequency_hz"] * self.data.time
            fields.append([np.sin(phase), np.cos(phase)])
        if self.motor is not None:
            fields.append(self.motor.observation()[0].numpy())
        return np.concatenate(fields).astype("float32")

    def step(self, command):
        observation = torch.from_numpy(self.observation(command)).unsqueeze(0)
        with torch.inference_mode():
            latent = self.runner.alg.actor(
                TensorDict({"actor": observation, "critic": observation}, batch_size=[1])
            )
        self.previous = np.tanh(latent.numpy()[0])
        baseline = np.array(self.spec["nominal"])
        if self.motor is None and self.task.get("reference_gait", False):
            from .gait import reference_targets

            baseline = reference_targets(
                torch.tensor([self.data.time]),
                torch.as_tensor(np.asarray(command), dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(baseline, dtype=torch.float32).unsqueeze(0),
                frequency=self.task["gait_frequency_hz"],
            )[0].numpy()
        self.target = np.clip(
            baseline + np.array(self.spec["action_scale"]) * self.previous,
            self.limits[:, 0],
            self.limits[:, 1],
        )
        if self.motor is not None:
            self.target = self.motor.advance(
                latent,
                torch.as_tensor(np.asarray(command), dtype=torch.float32).unsqueeze(0),
                torch.tensor([self.data.time]),
            )[0].numpy()
        for _ in range(round(self.spec["policy_dt"] / self.spec["physics_dt"])):
            torque = (
                np.array(self.spec["kp"]) * (self.target - self.data.qpos[self.qids])
                - np.array(self.spec["kd"]) * self.data.qvel[self.vids]
            )
            self.data.ctrl[self.aids] = np.clip(torque, -self.efforts, self.efforts)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.fallen

    @property
    def fallen(self):
        return bool(
            self.data.qpos[2] < self.spec["fall_height"]
            or self.data.xmat[self.root].reshape(3, 3)[2, 2] < np.cos(self.spec["fall_angle"])
        )


def play(checkpoint, seconds=None, headless=False, output=None):
    sim = PolicySimulation(checkpoint)
    if headless:
        return scripted_playback(sim, seconds or 30, output)
    import glfw

    if not glfw.init():
        raise RuntimeError("GLFW could not open a display. Run from a graphical desktop.")
    window = glfw.create_window(
        1100, 800, "MicroDuck — W/S walk · A/D turn · Space stop · R reset · Esc quit", None, None
    )
    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create MuJoCo window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    scene = mujoco.MjvScene(sim.model, maxgeom=10000)
    context = mujoco.MjrContext(sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    camera = mujoco.MjvCamera()
    camera.distance = 0.55
    camera.azimuth = 135
    camera.elevation = -15
    option = mujoco.MjvOption()
    telemetry = []
    start = time.monotonic()
    next_step = start
    reset_down = False
    try:
        while not glfw.window_should_close(window) and (
            seconds is None or time.monotonic() - start < seconds
        ):
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            pressed = {
                name
                for name in ["W", "A", "S", "D", "SPACE"]
                if glfw.get_key(window, getattr(glfw, "KEY_" + name)) == glfw.PRESS
            }
            reset = glfw.get_key(window, glfw.KEY_R) == glfw.PRESS
            if reset and not reset_down:
                sim.reset()
            reset_down = reset
            command = keyboard_command(pressed)
            now = time.monotonic()
            if now >= next_step:
                if not sim.fallen:
                    sim.step(command)
                if output:
                    telemetry.append(
                        dict(
                            t_s=sim.data.time,
                            command=command.tolist(),
                            position=sim.data.qpos[:3].tolist(),
                            fall=sim.fallen,
                        )
                    )
                next_step = max(next_step + sim.spec["policy_dt"], now)
            camera.lookat[:] = [sim.data.qpos[0], sim.data.qpos[1], 0.12]
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                sim.model, sim.data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            mujoco.mjr_render(viewport, scene, context)
            message = (
                "FALL — press R to reset"
                if sim.fallen
                else f"vx {command[0]:+.2f} m/s   turn {command[2]:+.2f} rad/s"
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                "W/S forward/back · A/D turn · release/Space stop\nR reset · Esc exit",
                message,
                context,
            )
            glfw.swap_buffers(window)
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
    if output:
        write_json(
            output,
            dict(
                checkpoint=str(sim.checkpoint),
                backend="interactive CPU MuJoCo",
                trajectory=telemetry,
            ),
        )
        return Path(output)


def scripted_playback(sim, seconds, output=None):
    """CPU transfer check including command transitions; independent of GPU metrics."""
    schedule = [
        ("stop", set()),
        ("W", {"W"}),
        ("A", {"A"}),
        ("S", {"S"}),
        ("D", {"D"}),
        ("stop", set()),
    ]
    rows = []
    for step in range(round(seconds / sim.spec["policy_dt"])):
        t = step * sim.spec["policy_dt"]
        segment = min(int(t / (seconds / len(schedule))), len(schedule) - 1)
        label, keys = schedule[segment]
        command = keyboard_command(keys)
        fallen = sim.step(command)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, sim.root, velocity, 1
        )
        rows.append(
            dict(
                t_s=sim.data.time,
                key=label,
                segment=segment,
                command=command.tolist(),
                body_velocity=velocity[3:].tolist(),
                body_angular_velocity=velocity[:3].tolist(),
                position=sim.data.qpos[:3].tolist(),
                root_pose=sim.data.qpos[:7].tolist(),
                joint_pos=sim.data.qpos[sim.qids].tolist(),
                fall=fallen,
            )
        )
        if fallen:
            break
    segments = []
    for index, (label, keys) in enumerate(schedule):
        samples = [r for r in rows if r["segment"] == index]
        steady = [r for r in samples if r["t_s"] - index * seconds / len(schedule) > 0.5]
        command = keyboard_command(keys)

        def errors(samples):
            if not samples:
                return None, None
            speed = np.array([r["body_velocity"][:2] for r in samples])
            yaw = np.array([r["body_angular_velocity"][2] for r in samples])
            return float(np.mean(np.sum((speed - command[:2]) ** 2, axis=1)) ** 0.5), float(
                np.mean((yaw - command[2]) ** 2) ** 0.5
            )

        raw_speed, raw_yaw = errors(samples)
        speed, yaw = errors(steady)
        segments.append(
            dict(
                key=label,
                command=command.tolist(),
                samples=len(samples),
                steady_samples=len(steady),
                velocity_rmse=raw_speed,
                yaw_rmse=raw_yaw,
                steady_velocity_rmse=speed,
                steady_yaw_rmse=yaw,
                passed=speed is not None and speed <= 0.025 and yaw <= 0.1,
            )
        )
    result = dict(
        segments=segments,
        tracking_gate=dict(settling_seconds=0.5, velocity_rmse=0.025, yaw_rmse=0.1),
        command_tracking_passed=all(s["passed"] for s in segments) and not sim.fallen,
        checkpoint=str(sim.checkpoint),
        checkpoint_sha=sha(sim.checkpoint),
        backend="CPU MuJoCo; actual policy and PD, no scripted joint trajectories",
        requested_seconds=seconds,
        survived_seconds=sim.data.time,
        fall=sim.fallen,
        trajectory=rows,
    )
    target = Path(output) if output else sim.checkpoint.parent / "keyboard_playback.json"
    write_json(target, result)
    return target
