"""Bounded orchestration around mjlab's maintained RSL-RL PPO integration."""

from dataclasses import asdict
import json
import signal
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from tensordict import TensorDict
from mjlab.rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from torch.utils.tensorboard import SummaryWriter
from .config import ROOT, read
from .tracking import new_run, append_json, write_json, seed_all, rng_state, restore_rng, sha
from .task import make_env
from .resources import ResourceMonitor


def timeout_reward(reward, terminal_values, truncated, terminated, gamma):
    """Bootstrap time limits from post-action terminal states; falls take precedence."""
    return reward + gamma * terminal_values.reshape(-1) * (truncated & ~terminated).float()


class TerminalAwareWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
        self.critic = None
        self.gamma = 0.99
        self.completed = []
        super().__init__(env)
        self.acc = torch.zeros(self.num_envs, 9, device=self.device)
        self.last_x = self.env.scene["robot"].data.root_link_pos_w[:, :2].clone()

    def step(self, actions):
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        terminated = terminated.clone()
        truncated = truncated.clone()
        rewards = rew.clone()
        self.last_task_reward = rewards.clone()
        self.last_reward_rates = self.env.reward_manager._step_reward.mean(0).clone()
        obs = TensorDict(obs, batch_size=[self.num_envs]).clone()
        if not all(torch.isfinite(v).all() for v in obs.values()):
            raise FloatingPointError("Non-finite observations")
        d = self.env.scene["robot"].data
        cmd = self.env.command_manager.get_command("twist")
        self.acc[:, 0] += self.env.step_dt
        self.acc[:, 1] += (d.root_link_lin_vel_b[:, :2] - cmd[:, :2]).square().sum(
            -1
        ) * self.env.step_dt
        self.acc[:, 2] += (d.root_link_ang_vel_b[:, 2] - cmd[:, 2]).square() * self.env.step_dt
        self.acc[:, 3] += (d.qfrc_actuator * d.joint_vel).abs().sum(-1) * self.env.step_dt
        self.acc[:, 4] += (d.root_link_pos_w[:, :2] - self.last_x).norm(dim=-1)
        self.acc[:, 5] += rewards
        limit = torch.tensor(self.env.cfg.robot_spec["effort_limits"], device=self.device)
        self.acc[:, 6] += (d.qfrc_actuator.abs() > 0.95 * limit).float().mean(-1) * self.env.step_dt
        lo, hi = d.soft_joint_pos_limits.unbind(-1)
        self.acc[:, 7] += ((d.joint_pos < lo - 0.001) | (d.joint_pos > hi + 0.001)).any(
            -1
        ).float() * self.env.step_dt
        from .task import reward_terms

        self.acc[:, 8] += reward_terms(self.env, "slip") * self.env.step_dt
        self.last_x[:] = d.root_link_pos_w[:, :2]
        done = terminated | truncated
        ids = done.nonzero().flatten()
        # The upstream PPO uses V(s_t) for extras['time_outs']. Supply the correct
        # terminal bootstrap in reward, then omit that key to avoid adding it twice.
        if self.critic is not None and truncated.any():
            with torch.no_grad():
                rewards = timeout_reward(
                    rewards, self.critic(obs), truncated, terminated, self.gamma
                )
        for i in ids.tolist():
            a = self.acc[i].tolist()
            self.completed.append(
                dict(
                    duration_s=a[0],
                    velocity_rmse=(a[1] / max(a[0], 1e-9)) ** 0.5,
                    yaw_rmse=(a[2] / max(a[0], 1e-9)) ** 0.5,
                    mechanical_work_J=a[3],
                    path_length_m=a[4],
                    reward=a[5],
                    saturation_fraction=a[6] / max(a[0], 1e-9),
                    joint_limit_fraction=a[7] / max(a[0], 1e-9),
                    foot_slip_rms_m_s=(a[8] / max(2 * a[0], 1e-9)) ** 0.5,
                    fall=bool(terminated[i]),
                    truncated=bool(truncated[i]),
                )
            )
        extras = dict(extras)
        extras.pop("time_outs", None)
        extras["terminal_observation"] = obs[ids].clone()
        extras["terminal_env_ids"] = ids.clone()
        extras["terminal_physics"] = {
            "root_pose": d.root_link_pose_w[ids].clone(),
            "joint_position": d.joint_pos[ids].clone(),
            "linear_velocity": d.root_link_lin_vel_b[ids].clone(),
        }
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        if len(ids):
            reset_obs, _ = self.env.reset(env_ids=ids)
            for key in obs.keys():
                obs[key][ids] = reset_obs[key][ids]
            self.acc[ids] = 0
            self.last_x[ids] = d.root_link_pos_w[ids, :2]
        return obs, rewards, done.long(), extras


def runner_config(cfg):
    return asdict(
        RslRlOnPolicyRunnerCfg(
            seed=cfg.seed,
            num_steps_per_env=cfg.horizon,
            max_iterations=cfg.iterations,
            save_interval=cfg.checkpoint_interval,
            logger="tensorboard",
            upload_model=False,
            actor=RslRlModelCfg(
                hidden_dims=(256, 256, 128),
                activation="elu",
                obs_normalization=True,
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 0.5,
                    "std_type": "scalar",
                },
            ),
            critic=RslRlModelCfg(
                hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True
            ),
            algorithm=RslRlPpoAlgorithmCfg(
                learning_rate=3e-4,
                schedule="fixed",
                entropy_coef=0.01,
                gamma=0.99,
                lam=0.95,
                clip_param=0.2,
                num_learning_epochs=5,
                num_mini_batches=4,
                value_loss_coef=1.0,
                max_grad_norm=1.0,
            ),
        )
    )


def construct(cfg):
    env = TerminalAwareWrapper(make_env(cfg))
    try:
        runner = MjlabOnPolicyRunner(env, runner_config(cfg), log_dir=None, device="cuda:0")
    except BaseException:
        env.close()
        raise
    env.critic = lambda obs: runner.alg.critic(obs)
    return env, runner


def save_checkpoint(path, runner, cfg, updates, transitions, level):
    state = runner.alg.save()
    state.update(
        iter=updates,
        infos={"env_state": {"common_step_counter": runner.env.env.common_step_counter}},
        project=dict(
            config=asdict(cfg),
            updates=updates,
            transitions=transitions,
            curriculum_level=level,
            rng=rng_state(),
            robot_hash=sha(ROOT / "configs/robot/custom.yaml"),
            urdf_hash=sha(ROOT / read("configs/robot/custom.yaml")["urdf"]),
            task_hash=sha(ROOT / "configs/task/flat.yaml"),
            lock_hash=sha(ROOT / "uv.lock"),
            simulator_state_restored=False,
        ),
    )
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, runner, training=False):
    state = torch.load(path, map_location="cuda:0", weights_only=False)
    if state["project"]["robot_hash"] != sha(ROOT / "configs/robot/custom.yaml"):
        raise ValueError("Robot specification changed since checkpoint")
    if state["project"]["urdf_hash"] != sha(ROOT / read("configs/robot/custom.yaml")["urdf"]):
        raise ValueError("Robot URDF changed since checkpoint")
    if training and state["project"]["lock_hash"] != sha(ROOT / "uv.lock"):
        raise ValueError("Dependency lock changed since checkpoint")
    if state["project"]["task_hash"] != sha(ROOT / "configs/task/flat.yaml"):
        raise ValueError("Task definition changed since checkpoint")
    runner.alg.load(state, None, True)
    if training:
        restore_rng(state["project"]["rng"])
    runner.env.env.common_step_counter = state["infos"]["env_state"]["common_step_counter"]
    return state["project"]


def set_level(env, level):
    term = env.env.command_manager.get_term("twist")
    term.cfg.ranges.lin_vel_x = [(0.0, 0.0), (0.0, 0.03), (0.0, 0.06)][level]
    term.cfg.ranges.ang_vel_z = [(0.0, 0.0), (0.0, 0.0), (-0.15, 0.15)][level]


def train(cfg, resume=None):
    audit = ROOT / "docs/model_audit.json"
    if not audit.exists() or not json.loads(audit.read_text())["gates_passed"]:
        raise RuntimeError("Run validate-model and pass physics gates before training")
    model_report = json.loads(audit.read_text())
    if model_report["urdf_sha"] != sha(
        ROOT / read("configs/robot/custom.yaml")["urdf"]
    ) or model_report["robot_spec_sha"] != sha(ROOT / "configs/robot/custom.yaml"):
        raise RuntimeError("Model changed after validation; rerun validate-model")
    seed_all(cfg.seed)
    torch.set_num_threads(4)
    path = new_run(cfg, parent=str(resume) if resume else None)
    monitor = ResourceMonitor().start()
    start = time.monotonic()
    interrupted = False

    def stop(*_):
        nonlocal interrupted
        interrupted = True

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    env = runner = None
    updates = transitions = level = 0
    status = "initializing"
    writer = SummaryWriter(str(path / "tensorboard"))
    try:
        env, runner = construct(cfg)
        if resume:
            state = load_checkpoint(resume, runner, training=True)
            old = state["config"]
            for key in (
                "num_envs",
                "horizon",
                "curriculum",
                "randomization",
                "episode_seconds",
                "seed",
            ):
                if old[key] != asdict(cfg)[key]:
                    raise ValueError(f"Resume cannot change {key}; start a distinct experiment")
            updates, transitions, level = (
                state["updates"],
                state["transitions"],
                state["curriculum_level"],
            )
        if cfg.curriculum:
            set_level(env, level)
        obs = env.get_observations()
        runner.alg.train_mode()
        torch.cuda.reset_peak_memory_stats()
        initialization = time.monotonic() - start
        write_json(
            path / "architecture.json",
            {
                "actor": str(runner.alg.actor),
                "critic": str(runner.alg.critic),
                "parameters": sum(p.numel() for p in runner.alg.actor.parameters())
                + sum(p.numel() for p in runner.alg.critic.parameters()),
            },
        )
        status = "budget_completed"
        while (
            updates < cfg.iterations
            and transitions + cfg.num_envs * cfg.horizon <= cfg.max_transitions
        ):
            if interrupted or time.monotonic() - start >= cfg.max_wall_time:
                status = "interrupted" if interrupted else "wall_time_limit"
                break
            torch.cuda.synchronize()
            t = time.monotonic()
            rewards_sum = 0.0
            component_sum = torch.zeros(len(env.env.reward_manager.active_terms), device=env.device)
            with torch.inference_mode():
                for _ in range(cfg.horizon):
                    action = runner.alg.act(obs)
                    obs, reward, done, extras = env.step(action)
                    runner.alg.process_env_step(obs, reward, done, extras)
                    rewards_sum += env.last_task_reward.mean().item()
                    component_sum += env.last_reward_rates
                runner.alg.compute_returns(obs)
            torch.cuda.synchronize()
            rollout_s = time.monotonic() - t
            t = time.monotonic()
            losses = runner.alg.update()
            with torch.no_grad():
                st = runner.alg.storage
                rollout_obs = st.observations.flatten(0, 1)
                mean = runner.alg.actor(rollout_obs)
                runner.alg.actor.distribution.update(mean)
                log_new = runner.alg.actor.get_output_log_prob(st.actions.flatten(0, 1))
                log_ratio = log_new - st.actions_log_prob.flatten()
                ratios = log_ratio.exp()
                losses["approx_kl"] = float(((ratios - 1) - log_ratio).mean())
                losses["clip_fraction"] = float(((ratios - 1).abs() > 0.2).float().mean())
                target = st.returns.flatten()
                prediction = runner.alg.critic(rollout_obs).flatten()
                variance = target.var()
                losses["explained_variance"] = (
                    float(1 - (target - prediction).var() / variance) if variance > 1e-10 else None
                )
                losses["last_actor_clipped_gradient_norm"] = float(
                    torch.stack(
                        [
                            p.grad.square().sum()
                            for p in runner.alg.actor.parameters()
                            if p.grad is not None
                        ]
                    )
                    .sum()
                    .sqrt()
                )
            torch.cuda.synchronize()
            optimizer_s = time.monotonic() - t
            updates += 1
            transitions += cfg.num_envs * cfg.horizon
            episodes = env.completed
            env.completed = []
            if episodes:
                for e in episodes:
                    append_json(path / "episodes.jsonl", dict(update=updates, **e))
            if cfg.curriculum and len(episodes) >= 8:
                survival = np.mean([e["duration_s"] for e in episodes]) / cfg.episode_seconds
                error = np.mean([e["velocity_rmse"] for e in episodes])
                next_level = (
                    min(2, level + 1)
                    if survival > 0.9 and error < 0.025
                    else max(0, level - 1)
                    if survival < 0.5
                    else level
                )
                if next_level != level:
                    append_json(
                        path / "curriculum.jsonl",
                        dict(
                            update=updates,
                            old=level,
                            new=next_level,
                            survival=survival,
                            velocity_rmse=error,
                        ),
                    )
                    level = next_level
                    set_level(env, level)
            components = {
                f"reward_rate/{n}": float(v / cfg.horizon)
                for n, v in zip(env.env.reward_manager.active_terms, component_sum)
            }
            record = dict(
                update=updates,
                transitions=transitions,
                policy_steps_per_world=updates * cfg.horizon,
                aggregate_physics_substeps=transitions * 10,
                gradient_steps=updates * 5 * 4,
                ppo_epochs_per_update=5,
                learning_rate=runner.alg.learning_rate if runner is not None else None,
                wall_s=time.monotonic() - start,
                initialization_s=initialization,
                rollout_s=rollout_s,
                optimizer_s=optimizer_s,
                transitions_per_second=cfg.num_envs * cfg.horizon / (rollout_s + optimizer_s),
                peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),
                ram_rss_bytes=psutil.Process().memory_info().rss,
                peak_device_vram_bytes=monitor.peak_device_vram_bytes,
                peak_process_ram_bytes=monitor.peak_process_ram_bytes,
                mean_step_reward=rewards_sum / cfg.horizon,
                curriculum_level=level,
                completed_episodes=len(episodes),
                falls=sum(e["fall"] for e in episodes),
                action_std=float(runner.alg.actor.output_std.mean()),
                **components,
                **losses,
            )
            append_json(path / "metrics.jsonl", record)
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, transitions)
            # Log manager reward rates, each with its own scalar.
            for name, values in env.env.reward_manager._episode_sums.items():
                writer.add_scalar("reward_accumulator/" + name, float(values.mean()), transitions)
            print(json.dumps(record), flush=True)
            save_checkpoint(path / "last.pt", runner, cfg, updates, transitions, level)
            if updates % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    path / f"update_{updates:06d}.pt", runner, cfg, updates, transitions, level
                )
        save_checkpoint(path / "final.pt", runner, cfg, updates, transitions, level)
    except BaseException as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        # Preserve the last completed update if an optimizer fails partway through.
        if runner is not None and not (path / "last.pt").exists():
            save_checkpoint(
                path / "diagnostic_partial.pt", runner, cfg, updates, transitions, level
            )
        raise
    finally:
        writer.close()
        monitor.close()
        write_json(
            path / "status.json",
            dict(
                status=status,
                updates=updates,
                transitions=transitions,
                policy_steps_per_world=updates * cfg.horizon,
                aggregate_physics_substeps=transitions * 10,
                gradient_steps=updates * 5 * 4,
                ppo_epochs_per_update=5,
                learning_rate=runner.alg.learning_rate if runner is not None else None,
                wall_s=time.monotonic() - start,
                policy_claim="unvalidated; no reliable walking claim",
            ),
        )
        if env is not None:
            env.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return path
