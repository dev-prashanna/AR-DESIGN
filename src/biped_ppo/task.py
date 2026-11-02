"""mjlab task with bounded position targets and named proprioception."""

from dataclasses import dataclass, field
import math
import torch
import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg
from .config import read
from .robot import generate


class BoundedPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.motor = None
        if env.cfg.task_spec.get("motor_target_filter"):
            from .control import MotorTargetController

            self.motor = MotorTargetController(
                self._offset,
                self._scale,
                self._clip[:, :, 0],
                self._clip[:, :, 1],
                env.cfg.task_spec,
                env.cfg.robot_spec["policy_dt"],
            )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.motor is not None:
            self.motor.reset(env_ids)

    def process_actions(self, actions):
        bounded = torch.tanh(actions)
        task = self._env.cfg.task_spec
        if self.motor is not None:
            self._raw_actions[:] = bounded
            self._processed_actions = self.motor.advance(
                actions,
                self._env.command_manager.get_command("twist"),
                self._env.episode_length_buf * self._env.step_dt,
            )
        elif task.get("reference_gait", False):
            from .gait import reference_targets

            cmd = self._env.command_manager.get_command("twist")
            time = self._env.episode_length_buf * self._env.step_dt
            reference = reference_targets(
                time, cmd, self._offset, frequency=task["gait_frequency_hz"]
            )
            self._raw_actions[:] = bounded
            self._processed_actions = torch.clamp(
                reference + bounded * self._scale, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        else:
            super().process_actions(bounded)


@dataclass(kw_only=True)
class BoundedPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return BoundedPositionAction(self, env)


@dataclass(kw_only=True)
class MicroduckEnvCfg(ManagerBasedRlEnvCfg):
    robot_spec: dict = field(default_factory=dict)
    task_spec: dict = field(default_factory=dict)


def robot(env):
    return env.scene["robot"].data


def angular_velocity(env):
    return robot(env).root_link_ang_vel_b + getattr(env, "gyro_bias", 0)


def joint_position(env):
    return mdp.joint_pos_rel(env) + getattr(env, "encoder_bias", 0)


def sensor_bias(env, env_ids):
    if not hasattr(env, "gyro_bias"):
        env.gyro_bias = torch.zeros(env.num_envs, 3, device=env.device)
        env.encoder_bias = torch.zeros(env.num_envs, 6, device=env.device)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    task = env.cfg.task_spec
    env.gyro_bias[ids] = torch.empty(len(ids), 3, device=env.device).uniform_(
        -task["gyro_bias_range"], task["gyro_bias_range"]
    )
    env.encoder_bias[ids] = torch.empty(len(ids), 6, device=env.device).uniform_(
        -task["encoder_bias_range"], task["encoder_bias_range"]
    )


def previous_action(env):
    return torch.tanh(env.action_manager.action)


def control_command(env):
    term = env.action_manager.get_term("joint_pos")
    return (
        term.motor.command if term.motor is not None else env.command_manager.get_command("twist")
    )


def motor_state(env):
    return env.action_manager.get_term("joint_pos").motor.observation()


def gait_phase(env):
    frequency = env.cfg.task_spec["gait_frequency_hz"]
    phase = 2 * math.pi * frequency * env.episode_length_buf * env.step_dt
    return torch.stack((phase.sin(), phase.cos()), dim=-1)


def reward_terms(env, term):
    task = getattr(env.cfg, "task_spec", TASK_SPEC) if hasattr(env, "cfg") else TASK_SPEC
    d = robot(env)
    cmd = (
        control_command(env)
        if task.get("motor_target_filter")
        else env.command_manager.get_command("twist")
    )
    if term == "yaw_error":
        return ((d.root_link_ang_vel_b[:, 2] - cmd[:, 2]) / 0.25).square()
    if term == "target_rate":
        motor = env.action_manager.get_term("joint_pos").motor
        return (motor.velocity / motor.speed_limit).square().mean(-1)
    if term == "target_acceleration":
        motor = env.action_manager.get_term("joint_pos").motor
        return (motor.acceleration_target / (motor.speed_limit / 0.1)).square().mean(-1)
    if term == "overspeed":
        limits = torch.tensor(env.cfg.robot_spec["speed_limits"], device=env.device)
        return ((d.joint_vel.abs() / limits - 1).clamp_min(0)).square().mean(-1)
    if term == "linear_progress":
        return ((d.root_link_lin_vel_b[:, :2] * cmd[:, :2]).sum(-1) / 0.05**2).clamp(-2, 2)
    if term == "yaw_progress":
        return (d.root_link_ang_vel_b[:, 2] * cmd[:, 2] / 0.25**2).clamp(-2, 2)
    if term in ("clearance", "contact_phase"):
        phase = gait_phase(env)[:, 0]
        moving = (cmd[:, 0].abs() > 0.005) | (cmd[:, 2].abs() > 0.025)
        swing = task.get("left_swing_phase_sign", 1) * torch.stack((phase, -phase), dim=-1)
        activation = (
            (cmd[:, 0].abs() / 0.02 + cmd[:, 2].abs() / 0.15).clamp(0, 1)
            if task.get("motor_target_filter")
            else moving
        )
        lift = ((swing - 0.3) / 0.7).clamp_min(0) * activation[:, None]
        if term == "contact_phase":
            contact = env.scene["feet"].data.found > 0
            expected = lift < 0.05
            return (contact == expected).float().mean(-1)
        ids, _ = env.scene["robot"].find_sites(("l_foot_contact", "r_foot_contact"))
        height = d.site_pos_w[:, ids, 2]
        target = task["foot_clearance_m"] * lift
        return torch.exp(-((height - target) / 0.004).square()).mean(-1)
    if term == "termination":
        return env.termination_manager.terminated.float()
    if term == "tracking":
        return torch.exp(
            -((d.root_link_lin_vel_b[:, :2] - cmd[:, :2]) ** 2).sum(-1)
            / task["tracking_bandwidth_m_s"] ** 2
        )
    if term == "yaw":
        return torch.exp(
            -((d.root_link_ang_vel_b[:, 2] - cmd[:, 2]) ** 2) / task["yaw_bandwidth_rad_s"] ** 2
        )
    if term == "upright":
        return torch.exp(
            -d.projected_gravity_b[:, :2].square().sum(-1) / task["upright_bandwidth"] ** 2
        )
    if term == "height":
        return torch.exp(
            -(
                (
                    (d.root_link_pos_w[:, 2] - env.cfg.robot_spec["base_height"])
                    / task["height_bandwidth_m"]
                )
                ** 2
            )
        )
    if term == "vertical":
        return d.root_link_lin_vel_b[:, 2].square()
    if term == "roll_pitch":
        return d.root_link_ang_vel_b[:, :2].square().sum(-1)
    if term == "torque":
        return d.qfrc_actuator.square().sum(-1)
    if term == "speed":
        return d.joint_vel.square().sum(-1)
    if term == "action_rate":
        return (
            (torch.tanh(env.action_manager.action) - torch.tanh(env.action_manager.prev_action))
            .square()
            .sum(-1)
        )
    if term == "limits":
        lo, hi = d.soft_joint_pos_limits.unbind(-1)
        return ((lo - d.joint_pos).clamp_min(0) + (d.joint_pos - hi).clamp_min(0)).sum(-1)
    if term == "saturation":
        limits = torch.tensor(env.cfg.robot_spec["effort_limits"], device=env.device)
        return (d.qfrc_actuator.abs() > 0.95 * limits).float().mean(-1)
    if term == "body_contact":
        return (env.scene["body_ground"].data.found > 0).float().sum(-1)
    if term == "slip":
        ids, _ = env.scene["robot"].find_sites(("l_foot_contact", "r_foot_contact"))
        # Foot sites near floor plus contact match; prevents swing-velocity penalty.
        found = env.scene["feet"].data.found > 0
        velocity = d.site_lin_vel_w[:, ids, :2].square().sum(-1)
        return (velocity * found).sum(-1)
    raise ValueError(term)


TASK_SPEC = read("configs/task/flat.yaml")


def fall(env):
    d = robot(env)
    return (d.root_link_pos_w[:, 2] < env.cfg.robot_spec["fall_height"]) | (
        d.projected_gravity_b[:, 2] > -math.cos(env.cfg.robot_spec["fall_angle"])
    )


def invalid(env):
    d = robot(env)
    finite = (
        torch.isfinite(d.joint_pos).all()
        & torch.isfinite(d.joint_vel).all()
        & torch.isfinite(d.root_link_pos_w).all()
    )
    if not finite:
        raise FloatingPointError(
            "Non-finite simulation state; aborting rather than resetting silently"
        )
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def make_cfg(run, robot_path=None):
    robot_path = robot_path or run.robot_path
    task = read(run.task_path)
    spec = read(robot_path)
    xml, audit = generate(robot_path)
    limits = {j["name"]: j["limits"] for j in audit["joints"] if j["type"] == "revolute"}
    spec["effort_limits"] = [float(limits[n]["effort"]) for n in spec["joint_names"]]
    spec["speed_limits"] = [float(limits[n]["velocity"]) for n in spec["joint_names"]]
    actuators = tuple(
        IdealPdActuatorCfg(
            target_names_expr=(name,),
            stiffness=kp,
            damping=kd,
            effort_limit=float(limits[name]["effort"]),
            delay_max_lag=task.get("delay_max_lag", 2) if run.randomization else 0,
            delay_update_period=task.get("delay_update_period", 0),
            delay_hold_prob=task.get("delay_hold_prob", 0.0),
        )
        for name, kp, kd in zip(spec["joint_names"], spec["kp"], spec["kd"])
    )
    entity = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(xml),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(spec["base_height"])),
            joint_pos={n: float(q) for n, q in zip(spec["joint_names"], spec["nominal"])},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators, soft_joint_pos_limit_factor=1.0
        ),
    )
    terms = {
        "angular_velocity": ObservationTermCfg(
            func=angular_velocity, scale=0.25, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02)
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
        ),
        "joint_position": ObservationTermCfg(
            func=joint_position, noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005)
        ),
        "joint_velocity": ObservationTermCfg(
            func=mdp.joint_vel_rel, scale=0.05, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        ),
        "previous_action": ObservationTermCfg(func=previous_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
        ),
    }
    if "gait_frequency_hz" in task:
        terms["gait_phase"] = ObservationTermCfg(func=gait_phase)
    if task.get("motor_target_filter"):
        terms["command"] = ObservationTermCfg(func=control_command)
        terms["motor_state"] = ObservationTermCfg(func=motor_state)
    from .commands import KeyboardVelocityCommandCfg

    command_class = (
        KeyboardVelocityCommandCfg
        if task.get("keyboard_command_mixture")
        else UniformVelocityCommandCfg
    )
    cmd = command_class(
        entity_name="robot",
        resampling_time_range=tuple(
            task.get("command_resampling_range", [task["command_resampling_seconds"]] * 2)
        ),
        rel_standing_envs=task["standing_fraction"],
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=tuple(task["final_forward_range"]),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=tuple(task["final_yaw_range"]),
        ),
    )
    if run.curriculum:
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
    events = {
        "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "roll": (-task["reset_base_tilt_rad"], task["reset_base_tilt_rad"]),
                    "pitch": (-task["reset_base_tilt_rad"], task["reset_base_tilt_rad"]),
                    "z": (0.0, task["reset_spawn_lift_m"]),
                },
                "velocity_range": {},
            },
        ),
        "reset_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (
                    -task["reset_joint_offset_rad"],
                    task["reset_joint_offset_rad"],
                ),
                "velocity_range": (0.0, 0.0),
            },
        ),
        "friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "ranges": (0.8, 0.8),
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*collision.*",)),
            },
        ),
    }
    if run.randomization:
        asset = SceneEntityCfg("robot", body_names=(".*",))
        events["density"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "alpha_range": tuple(task.get("density_alpha_range", [-0.025, 0.025])),
                "asset_cfg": asset,
            },
        )
        events["friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "ranges": tuple(task.get("friction_range", [0.7, 0.9])),
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*collision.*",)),
            },
        )
        events["gains"] = EventTermCfg(
            func=dr.pd_gains,
            mode="reset",
            params={
                "kp_range": tuple(task.get("gain_range", [0.9, 1.1])),
                "kd_range": tuple(task.get("gain_range", [0.9, 1.1])),
            },
        )
        if task.get("motor_target_filter"):
            events["sensor_bias"] = EventTermCfg(func=sensor_bias, mode="reset")
            events["effort"] = EventTermCfg(
                func=dr.effort_limits,
                mode="startup",
                params={"effort_limit_range": tuple(task.get("effort_scale_range", [0.8, 1.1]))},
            )
    return MicroduckEnvCfg(
        robot_spec=spec,
        task_spec=task,
        seed=run.seed,
        scene=SceneCfg(
            num_envs=run.num_envs,
            extent=0.5,
            entities={"robot": entity},
            terrain=TerrainEntityCfg(terrain_type="plane"),
            sensors=(
                ContactSensorCfg(
                    name="feet",
                    primary=ContactMatch(mode="body", pattern="[lr]_foot", entity="robot"),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found", "force"),
                    reduce="netforce",
                    num_slots=1,
                ),
                ContactSensorCfg(
                    name="body_ground",
                    primary=ContactMatch(
                        mode="body",
                        pattern="(torso|head|.*(hip|thigh|knee|shin).*)",
                        entity="robot",
                    ),
                    secondary=ContactMatch(mode="body", pattern="terrain"),
                    fields=("found",),
                    reduce="netforce",
                    num_slots=1,
                ),
            ),
        ),
        observations={
            name: ObservationGroupCfg(
                terms=terms, concatenate_terms=True, enable_corruption=run.randomization
            )
            for name in ["actor", "critic"]
        },
        actions={
            "joint_pos": BoundedPositionActionCfg(
                entity_name="robot",
                actuator_names=tuple(spec["joint_names"]),
                scale=dict(zip(spec["joint_names"], spec["action_scale"])),
                clip={
                    n: (float(limits[n]["lower"]), float(limits[n]["upper"]))
                    for n in spec["joint_names"]
                },
            )
        },
        commands={"twist": cmd},
        events=events,
        rewards={
            n: RewardTermCfg(func=reward_terms, weight=w, params={"term": n})
            for n, w in {n: r["weight"] for n, r in task["rewards"].items()}.items()
        },
        terminations={
            "timeout": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fall": TerminationTermCfg(func=fall),
            "invalid": TerminationTermCfg(func=invalid),
        },
        sim=SimulationCfg(
            nconmax=64,
            njmax=256,
            mujoco=MujocoCfg(
                timestep=spec["physics_dt"],
                iterations=30,
                ls_iterations=20,
                integrator="implicitfast",
            ),
        ),
        decimation=round(spec["policy_dt"] / spec["physics_dt"]),
        episode_length_s=run.episode_seconds,
        auto_reset=False,
    )


def make_env(run, device="cuda:0"):
    return ManagerBasedRlEnv(cfg=make_cfg(run), device=device)
