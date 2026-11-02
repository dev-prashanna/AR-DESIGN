from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
import torch
from tensordict import TensorDict
from .config import ROOT, RunConfig, read, observation_dimension
from .tracking import sha, write_json, append_json, seed_all
from .training import construct, load_checkpoint


def command_observations(env):
    # External command changes must invalidate mjlab's cached observation frame.
    return TensorDict(
        env.env.observation_manager.compute(update_history=True), batch_size=[env.num_envs]
    )


def evaluate(checkpoint, suite_path):
    checkpoint = Path(checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["project"]
    suite = read(suite_path)
    if suite["split"] == "test":
        selection = checkpoint.parent / "validation_selection.json"
        if not selection.exists() or json.loads(selection.read_text())["checkpoint_sha"] != sha(
            checkpoint
        ):
            raise ValueError("Test requires the checkpoint selected by validation")
    cases = [
        dict(evaluation_seed=seed, command=command, friction=friction, replica=replica)
        for seed in suite["seeds"]
        for command in suite["commands"]
        for friction in suite["friction"]
        for replica in range(4)
    ]
    cfg = replace(
        RunConfig(**state["config"]),
        num_envs=len(cases),
        randomization=suite.get("randomization", False),
        seed=suite.get("environment_seed", state["config"]["seed"]),
        curriculum=False,
        episode_seconds=suite["episode_seconds"],
    )
    env, runner = construct(cfg)
    load_checkpoint(checkpoint, runner)
    runner.alg.eval_mode()
    env.critic = None
    result_dir = checkpoint.parent / ("evaluation_" + suite["split"] + "_" + str(time.time_ns()))
    result_dir.mkdir()
    rows = []
    try:
        from mjlab.envs.mdp import dr
        from mjlab.managers import SceneEntityCfg

        term = env.env.command_manager.get_term("twist")
        term.cfg.rel_standing_envs = 0.0
        term.cfg.resampling_time_range = (100.0, 100.0)
        # Identical seeded initial-condition groups across every method/policy.
        for seed in suite["seeds"]:
            ids = torch.tensor(
                [i for i, c in enumerate(cases) if c["evaluation_seed"] == seed], device=env.device
            )
            seed_all(seed)
            env.env.reset(seed=seed, env_ids=ids)
        for friction in suite["friction"]:
            ids = torch.tensor(
                [i for i, c in enumerate(cases) if c["friction"] == friction], device=env.device
            )
            dr.geom_friction(
                env.env,
                ids,
                ranges=(friction, friction),
                asset_cfg=SceneEntityCfg("robot", geom_names=(".*collision.*",)),
            )
        term.vel_command_b[:] = torch.tensor(
            [c["command"] for c in cases], device=env.device, dtype=torch.float32
        )
        term.is_standing_env[:] = False
        env.acc.zero_()
        env.last_x[:] = env.env.scene["robot"].data.root_link_pos_w[:, :2]
        env.completed = []
        obs = command_observations(env)
        active = torch.ones(len(cases), device=env.device, dtype=torch.bool)
        selected = [
            i
            for i, c in enumerate(cases)
            if c["evaluation_seed"] == suite["seeds"][0]
            and c["friction"] == suite["friction"][0]
            and c["replica"] == 0
        ]
        trajectories = {i: [] for i in selected}
        forward_distance = torch.zeros(len(cases), device=env.device)
        foot_sites, _ = env.env.scene["robot"].find_sites(("l_foot_contact", "r_foot_contact"))
        airborne_time = torch.zeros(len(cases), 2, device=env.device)
        peak_clearance = torch.zeros_like(airborne_time)
        lift_count = torch.zeros_like(airborne_time, dtype=torch.long)
        previous_airborne = torch.zeros_like(airborne_time, dtype=torch.bool)
        stop_at = suite.get("stop_at_s")
        stopped = False
        stop_time = torch.zeros(len(cases), device=env.device)
        stop_speed_sq = torch.zeros(len(cases), device=env.device)
        mass = json.loads(
            (ROOT / read(cfg.robot_path).get("audit_path", "docs/model_audit.json")).read_text()
        )["mass_kg"]
        with torch.inference_mode():
            for step in range(round(suite["episode_seconds"] / env.env.step_dt) + 1):
                if stop_at is not None and not stopped and step * env.env.step_dt >= stop_at:
                    term.vel_command_b[:] = 0.0
                    obs = command_observations(env)
                    stopped = True
                obs, _, done, extras = env.step(runner.alg.actor(obs))
                ids = extras["terminal_env_ids"].tolist()
                velocity = env.env.scene["robot"].data.root_link_lin_vel_b.clone()
                if ids:
                    velocity[extras["terminal_env_ids"]] = extras["terminal_physics"][
                        "linear_velocity"
                    ]
                height = env.env.scene["robot"].data.site_pos_w[:, foot_sites, 2].clone()
                contact = env.env.scene["feet"].data.found.clone()
                if ids:
                    height[extras["terminal_env_ids"]] = extras["terminal_physics"]["foot_height"]
                    contact[extras["terminal_env_ids"]] = extras["terminal_physics"]["foot_contact"]
                airborne = (height > 0.003) & (contact == 0)
                airborne_time += airborne * active[:, None] * env.env.step_dt
                peak_clearance = torch.maximum(peak_clearance, height * active[:, None])
                lift_count += (airborne & ~previous_airborne) * active[:, None]
                previous_airborne = airborne
                forward_distance += velocity[:, 0] * env.env.step_dt * active.float()
                if stopped:
                    stop_time += env.env.step_dt * active.float()
                    stop_speed_sq += (
                        velocity[:, :2].square().sum(-1) * env.env.step_dt * active.float()
                    )
                for selected_id in selected:
                    if not active[selected_id] or not (step % 5 == 0 or done[selected_id]):
                        continue
                    data = env.env.scene["robot"].data
                    if selected_id in ids:
                        j = ids.index(selected_id)
                        physics = extras["terminal_physics"]
                        pose = physics["root_pose"][j].tolist()
                        q = physics["joint_position"][j].tolist()
                        vel = physics["linear_velocity"][j].tolist()
                    else:
                        pose = data.root_link_pose_w[selected_id].tolist()
                        q = data.joint_pos[selected_id].tolist()
                        vel = data.root_link_lin_vel_b[selected_id].tolist()
                    trajectories[selected_id].append(
                        dict(
                            t_s=(step + 1) * env.env.step_dt,
                            root_pose=pose,
                            joint_pos=q,
                            velocity=vel,
                            terminal=bool(done[selected_id]),
                        )
                    )
                finished = env.completed
                env.completed = []
                for i, episode in zip(ids, finished):
                    if not active[i]:
                        continue
                    distance = float(forward_distance[i])
                    cot = (
                        episode["mechanical_work_J"] / (mass * 9.81 * distance)
                        if distance > 0.02 and cases[i]["command"][0] > 0.001
                        else None
                    )
                    success = (
                        not episode["fall"]
                        and episode["duration_s"]
                        >= suite["success_survival_fraction"] * suite["episode_seconds"]
                        and episode["velocity_rmse"] <= suite["success_velocity_rmse"]
                        and episode["yaw_rmse"] <= suite["success_yaw_rmse"]
                    )
                    row = dict(
                        checkpoint_sha=sha(checkpoint),
                        training_seed=state["config"]["seed"],
                        environment=i,
                        split=suite["split"],
                        success=success,
                        mechanical_cot=cot,
                        forward_distance_m=distance,
                        foot_lifts=lift_count[i].tolist(),
                        airborne_fraction=(airborne_time[i] / episode["duration_s"]).tolist(),
                        peak_foot_clearance_m=peak_clearance[i].tolist(),
                        walking_success=success
                        and (
                            abs(cases[i]["command"][0]) <= 0.001 or bool((lift_count[i] >= 2).all())
                        ),
                        **cases[i],
                        **episode,
                    )
                    if stop_at is not None:
                        row["stop_at_s"] = stop_at
                        row["stop_reached"] = episode["duration_s"] >= stop_at
                        row["post_stop_velocity_rms_m_s"] = (
                            float((stop_speed_sq[i] / stop_time[i]).sqrt())
                            if stop_time[i] > 0
                            else None
                        )
                    rows.append(row)
                    append_json(result_dir / "episodes.jsonl", row)
                    active[i] = False
                if not active.any():
                    break
        if active.any():
            raise RuntimeError("Evaluation ended with unfinished episodes")
        write_json(
            result_dir / "trajectories.json",
            dict(
                selection_rule="First configured evaluation seed, first friction, replica 0, every command; retain successes and failures",
                cases=[
                    dict(environment=i, case=cases[i], trajectory=trajectories[i]) for i in selected
                ],
            ),
        )
        command_groups = [
            [r for r in rows if r["command"] == command] for command in suite["commands"]
        ]
        minimum_command_success = min(
            float(np.mean([r["walking_success"] for r in group])) for group in command_groups
        )
        worst_yaw_rmse = max(
            float(np.mean([r["yaw_rmse"] for r in group])) for group in command_groups
        )
        summary = dict(
            minimum_command_success_rate=minimum_command_success,
            maximum_command_yaw_rmse=worst_yaw_rmse,
            evaluator_sha=sha(Path(__file__)),
            wrapper_sha=sha(ROOT / "src/biped_ppo/training.py"),
            episodes=len(rows),
            success_rate=float(np.mean([r["success"] for r in rows])),
            walking_success_rate=float(np.mean([r["walking_success"] for r in rows])),
            fall_rate=float(np.mean([r["fall"] for r in rows])),
            mean_duration_s=float(np.mean([r["duration_s"] for r in rows])),
            mean_velocity_rmse=float(np.mean([r["velocity_rmse"] for r in rows])),
            independent_training_seeds=1,
            training_seed=state["config"]["seed"],
            training_updates=state["updates"],
            training_transitions=state["transitions"],
            checkpoint_sha=sha(checkpoint),
            suite_sha=sha(ROOT / suite_path),
            split=suite["split"],
            config=state["config"],
            evaluation_parallel_worlds=len(cases),
        )
        write_json(result_dir / "summary.json", summary)
        if suite["split"] == "validation":
            import shutil

            selection = checkpoint.parent / "validation_selection.json"
            candidate = dict(
                selection_metric=suite.get("selection_metric", "success_rate"),
                selection_score=summary[suite.get("selection_metric", "success_rate")],
                checkpoint_sha=sha(checkpoint),
                checkpoint=str(checkpoint),
                success_rate=summary["success_rate"],
                walking_success_rate=summary["walking_success_rate"],
                worst_yaw_rmse=summary["maximum_command_yaw_rmse"],
                velocity_rmse=summary["mean_velocity_rmse"],
                suite_sha=summary["suite_sha"],
                evaluation=str(result_dir),
            )
            old = json.loads(selection.read_text()) if selection.exists() else None
            if old is not None and old["suite_sha"] != candidate["suite_sha"]:
                write_json(
                    checkpoint.parent / ("validation_selection_" + old["suite_sha"][:16] + ".json"),
                    old,
                )
                old = None
            if old is None or selection_quality(candidate) > selection_quality(old):
                write_json(selection, candidate)
                if checkpoint.resolve() != (checkpoint.parent / "best_validation.pt").resolve():
                    shutil.copy2(checkpoint, checkpoint.parent / "best_validation.pt")
    finally:
        env.close()
    return result_dir


def inference_runner(checkpoint):
    from tensordict import TensorDict
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from .training import runner_config

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    project = state["project"]
    cfg = RunConfig(**project["config"])
    if project["robot_hash"] != sha(ROOT / cfg.robot_path) or project["urdf_hash"] != sha(
        ROOT / read(cfg.robot_path)["urdf"]
    ):
        raise ValueError("Robot assets changed since checkpoint")

    dimension = observation_dimension(cfg.task_path)

    class InferenceEnv:
        num_envs = 1
        num_actions = 6
        device = "cpu"
        cfg = {}

        def get_observations(self):
            return TensorDict(
                {"actor": torch.zeros(1, dimension), "critic": torch.zeros(1, dimension)},
                batch_size=[1],
            )

    runner = MjlabOnPolicyRunner(
        InferenceEnv(), runner_config(RunConfig(**project["config"])), device="cpu"
    )
    runner.alg.load(state, None, True)
    runner.alg.eval_mode()
    return runner


def export(checkpoint, fmt="onnx"):
    import onnxruntime as ort

    checkpoint = Path(checkpoint).resolve()
    runner = inference_runner(checkpoint)
    try:
        directory = checkpoint.parent / "export"
        directory.mkdir(exist_ok=True)
        # Inference export embeds empirical observation normalization. Output is
        # latent Gaussian mean; deployment applies tanh, scale, offset and limits.
        if fmt == "onnx":
            runner.export_policy_to_onnx(str(directory), "policy.onnx")
            policy = runner.alg.actor.as_onnx(verbose=False).cpu().eval()
            inputs = policy.get_dummy_inputs()
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            session = ort.InferenceSession(
                str(directory / "policy.onnx"), providers=["CPUExecutionProvider"]
            )
            max_error = 0.0
            with torch.no_grad():
                for scale in [0.0, 0.1, 1.0, 3.0]:
                    normalizer = runner.alg.actor.obs_normalizer
                    xs = tuple(
                        normalizer.mean + normalizer.std * torch.randn_like(x) * scale
                        for x in inputs
                    )
                    ref = policy(*xs).numpy()
                    actual = session.run(
                        None, {v.name: x.numpy() for v, x in zip(session.get_inputs(), xs)}
                    )[0]
                    max_error = max(max_error, float(np.max(np.abs(ref - actual))))
            if max_error > 1e-5:
                raise AssertionError(f"Export parity failed: {max_error}")
        else:
            runner.export_policy_to_jit(str(directory), "policy.pt")
            policy = runner.alg.actor.as_jit().cpu().eval()
            traced = torch.jit.load(str(directory / "policy.pt"))
            config = RunConfig(
                **torch.load(checkpoint, map_location="cpu", weights_only=False)["project"][
                    "config"
                ]
            )
            x = torch.randn(1, observation_dimension(config.task_path))
            with torch.no_grad():
                max_error = float((policy(x) - traced(x)).abs().max())
            if max_error > 1e-5:
                raise AssertionError("JIT parity failed")
        cfg = RunConfig(
            **torch.load(checkpoint, map_location="cpu", weights_only=False)["project"]["config"]
        )
        robot = read(cfg.robot_path)
        write_json(
            directory / "deployment.json",
            dict(
                robot=robot,
                actuator_limits={
                    j["name"]: j["limits"]
                    for j in json.loads(
                        (ROOT / robot.get("audit_path", "docs/model_audit.json")).read_text()
                    )["joints"]
                    if j["type"] == "revolute"
                },
                checkpoint_sha=sha(checkpoint),
                output="latent mean; apply through MotorTargetController.advance with persistent state"
                if read(cfg.task_path).get("motor_target_filter")
                else "latent mean; q_target=clip(reference(time,command)+scale*tanh(output),joint_lower,joint_upper)"
                if read(cfg.task_path).get("reference_gait", False)
                else "latent mean; q_target=clip(q_nominal+scale*tanh(output),joint_lower,joint_upper)",
                motor_target_controller=read(cfg.task_path)
                if read(cfg.task_path).get("motor_target_filter")
                else None,
                controller_implementation="biped_ppo.control.MotorTargetController"
                if read(cfg.task_path).get("motor_target_filter")
                else None,
                reference_gait=read(cfg.task_path).get("reference_gait", False),
                reference_implementation="biped_ppo.gait.reference_targets"
                if read(cfg.task_path).get("reference_gait", False)
                else None,
                observation_fields=[
                    ["body_angular_velocity", 3, 0.25],
                    ["projected_gravity", 3, 1],
                    ["relative_joint_position", 6, 1],
                    ["joint_velocity", 6, 0.05],
                    ["previous_tanh_action", 6, 1],
                    ["command_vx_vy_wz", 3, 1],
                ]
                + ([["gait_sin_cos", 2, 1]] if "gait_frequency_hz" in read(cfg.task_path) else [])
                + (
                    [["applied_target_relative", 6, 1], ["target_velocity", 6, 0.1]]
                    if read(cfg.task_path).get("motor_target_filter")
                    else []
                ),
                frames="body frame; x forward y left z up; radians, seconds, SI",
                gait_phase_hz=read(cfg.task_path).get("gait_frequency_hz"),
                gait_phase_fields=["sin(phase)", "cos(phase)"]
                if "gait_frequency_hz" in read(cfg.task_path)
                else [],
                history=1,
                frequency_hz=50,
                normalization="embedded",
                format=fmt,
                max_absolute_error=max_error,
                tolerance=1e-5,
                hardware_deployment_authorized=False,
            ),
        )
    finally:
        del runner
    if read(cfg.task_path).get("motor_target_filter"):
        import shutil

        runtime_source = directory / "runtime_source/biped_ppo"
        runtime_source.mkdir(parents=True, exist_ok=True)
        (runtime_source / "__init__.py").write_text("")
        for filename in ("control.py", "gait.py", "runtime.py"):
            shutil.copy2(ROOT / "src/biped_ppo" / filename, runtime_source / filename)
    return directory


def selection_quality(candidate):
    if candidate.get("selection_metric") == "minimum_command_success_rate":
        return (
            candidate["selection_score"],
            candidate.get("walking_success_rate", 0.0),
            -candidate.get("worst_yaw_rmse", float("inf")),
            -candidate["velocity_rmse"],
        )
    return (
        candidate.get("selection_score", candidate["success_rate"]),
        candidate["success_rate"],
        -candidate["velocity_rmse"],
    )
