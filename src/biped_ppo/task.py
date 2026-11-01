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
    def process_actions(self, actions):
        super().process_actions(torch.tanh(actions))


@dataclass(kw_only=True)
class BoundedPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return BoundedPositionAction(self, env)


@dataclass(kw_only=True)
class MicroduckEnvCfg(ManagerBasedRlEnvCfg):
    robot_spec: dict = field(default_factory=dict)


def robot(env):
    return env.scene["robot"].data


def angular_velocity(env):
    return robot(env).root_link_ang_vel_b


def previous_action(env):
    return torch.tanh(env.action_manager.action)


def reward_terms(env, term):
    d = robot(env)
    cmd = env.command_manager.get_command("twist")
    if term == "tracking":
        return torch.exp(
            -((d.root_link_lin_vel_b[:, :2] - cmd[:, :2]) ** 2).sum(-1)
            / TASK_SPEC["tracking_bandwidth_m_s"] ** 2
        )
    if term == "yaw":
        return torch.exp(
            -((d.root_link_ang_vel_b[:, 2] - cmd[:, 2]) ** 2)
            / TASK_SPEC["yaw_bandwidth_rad_s"] ** 2
        )
    if term == "upright":
        return torch.exp(
            -d.projected_gravity_b[:, :2].square().sum(-1) / TASK_SPEC["upright_bandwidth"] ** 2
        )
    if term == "height":
        return torch.exp(
            -(
                (
                    (d.root_link_pos_w[:, 2] - env.cfg.robot_spec["base_height"])
                    / TASK_SPEC["height_bandwidth_m"]
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
REWARDS = {name: term["weight"] for name, term in TASK_SPEC["rewards"].items()}


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


def make_cfg(run, robot_path="configs/robot/custom.yaml"):
    spec = read(robot_path)
    xml, audit = generate(robot_path)
    limits = {j["name"]: j["limits"] for j in audit["joints"] if j["type"] == "revolute"}
    spec["effort_limits"] = [float(limits[n]["effort"]) for n in spec["joint_names"]]
    actuators = tuple(
        IdealPdActuatorCfg(
            target_names_expr=(name,),
            stiffness=kp,
            damping=kd,
            effort_limit=float(limits[name]["effort"]),
            delay_max_lag=2 if run.randomization else 0,
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
            func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005)
        ),
        "joint_velocity": ObservationTermCfg(
            func=mdp.joint_vel_rel, scale=0.05, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        ),
        "previous_action": ObservationTermCfg(func=previous_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
        ),
    }
    cmd = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 3.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.06), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.15, 0.15)
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
                    "roll": (-TASK_SPEC["reset_base_tilt_rad"], TASK_SPEC["reset_base_tilt_rad"]),
                    "pitch": (-TASK_SPEC["reset_base_tilt_rad"], TASK_SPEC["reset_base_tilt_rad"]),
                    "z": (0.0, TASK_SPEC["reset_spawn_lift_m"]),
                },
                "velocity_range": {},
            },
        ),
        "reset_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (
                    -TASK_SPEC["reset_joint_offset_rad"],
                    TASK_SPEC["reset_joint_offset_rad"],
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
            params={"alpha_range": (-0.025, 0.025), "asset_cfg": asset},
        )
        events["friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "ranges": (0.7, 0.9),
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*collision.*",)),
            },
        )
        events["gains"] = EventTermCfg(
            func=dr.pd_gains, mode="reset", params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)}
        )
    return MicroduckEnvCfg(
        robot_spec=spec,
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
            for n, w in REWARDS.items()
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
