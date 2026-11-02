"""Generate artifact-linked futuristic results and the paper extension."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from biped_ppo.config import ROOT, read
from biped_ppo.tracking import sha, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--test", type=Path, required=True)
parser.add_argument("--playback", type=Path, required=True)
args = parser.parse_args()
run = args.run.resolve()
args.test = args.test.resolve()
args.playback = args.playback.resolve()
selection = json.loads((run / "validation_selection.json").read_text())
summary = json.loads((args.test / "summary.json").read_text())
if summary["checkpoint_sha"] != selection["checkpoint_sha"]:
    raise ValueError("Test does not correspond to validation-selected checkpoint")
robot = read(summary["config"]["robot_path"])
task = read(summary["config"]["task_path"])
rows = [json.loads(s) for s in (args.test / "episodes.jsonl").read_text().splitlines()]
metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
status = json.loads((run / "status.json").read_text())
playback = json.loads(args.playback.read_text())
ancestry = []
resume_ancestry = []
ancestor = run
visited = set()
while ancestor not in visited:
    visited.add(ancestor)
    if (ancestor / "warm_start.json").exists():
        entry = json.loads((ancestor / "warm_start.json").read_text())
        ancestry.append(entry)
        checkpoint = Path(entry["checkpoint"])
    else:
        manifest = json.loads((ancestor / "manifest.json").read_text())
        if not manifest.get("parent_checkpoint"):
            break
        checkpoint = Path(manifest["parent_checkpoint"])
        resume_ancestry.append(dict(checkpoint=str(checkpoint), counters_already_included=True))
    ancestor = (checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).resolve().parent
else:
    raise RuntimeError("Training ancestry cycle")
commands = []
names = {
    (0, 0, 0): "Stop",
    (0.05, 0, 0): "W",
    (-0.05, 0, 0): "S",
    (0, 0, 0.25): "A",
    (0, 0, -0.25): "D",
    (0.05, 0, 0.25): "W+A",
    (0.05, 0, -0.25): "W+D",
}
for command in sorted(set(tuple(e["command"]) for e in rows)):
    cases = [e for e in rows if tuple(e["command"]) == command]
    commands.append(
        dict(
            key=names[command],
            command=command,
            episodes=len(cases),
            success_rate=float(np.mean([e["success"] for e in cases])),
            fall_rate=float(np.mean([e["fall"] for e in cases])),
            walking_success_rate=float(np.mean([e.get("walking_success", False) for e in cases])),
            mean_foot_lifts=np.mean([e.get("foot_lifts", [0, 0]) for e in cases], axis=0).tolist(),
            velocity_rmse=float(np.mean([e["velocity_rmse"] for e in cases])),
            yaw_rmse=float(np.mean([e["yaw_rmse"] for e in cases])),
            forward_distance_m=float(np.mean([e["forward_distance_m"] for e in cases])),
        )
    )
result = dict(
    run=str(run),
    status=status,
    selected_checkpoint=selection,
    test_summary=summary,
    commands=commands,
    command_tracking_passed=all(c["success_rate"] >= 0.95 for c in commands),
    gpu_walking_gate_passed=all(c["walking_success_rate"] >= 0.95 for c in commands),
    cpu_keyboard_gate_passed=playback.get("command_tracking_passed", False),
    reliable_wasd=all(c["walking_success_rate"] >= 0.95 for c in commands)
    and playback.get("command_tracking_passed", False),
    warm_start_ancestry=ancestry,
    resume_ancestry=resume_ancestry,
    selected_policy_total_transitions=summary["training_transitions"]
    + sum(e["previous_transitions"] for e in ancestry),
    cpu_keyboard={k: v for k, v in playback.items() if k != "trajectory"},
    independent_training_seeds=1,
    physical_hardware_tested=False,
    hardware_calibration_complete=False,
    hardware_contract="docs/physical_policy_contract.json",
    sources={
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            run / "manifest.json",
            run / "metrics.jsonl",
            run / "status.json",
            run / "validation_selection.json",
            args.test / "summary.json",
            args.test / "episodes.jsonl",
            args.playback,
            ROOT / robot["audit_path"],
            ROOT / "assets/robot/meshes/futuristic/manifest.json",
        ]
        + ([run / "gpu_samples.jsonl"] if (run / "gpu_samples.jsonl").exists() else [])
    },
)
write_json(ROOT / "results/futuristic_results.json", result)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
x = [r["transitions"] / 1e6 for r in metrics]
axes[0].plot(x, [r["mean_step_reward"] for r in metrics], linewidth=0.7)
axes[0].set(
    xlabel="Million transitions",
    ylabel="Reward / policy step",
    title="Futuristic training: actual raw data",
)
axes[1].bar([c["key"] for c in commands], [100 * c["success_rate"] for c in commands])
axes[1].set(ylabel="Success (%)", ylim=(0, 105), title="Held-out commands: one trained policy")
axes[2].plot(x, [r["action_std"] for r in metrics], linewidth=0.7)
axes[2].set(xlabel="Million transitions", ylabel="Gaussian latent std", title="Exploration")
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
figure = ROOT / "results/figures/futuristic_training.png"
fig.savefig(figure, dpi=180)
plt.close(fig)
lines = [
    r"\section{Futuristic CAD and keyboard-control extension}",
    r"This extension is separate from the initial matched study. The actual 122 futuristic FreeCAD surfaces are attached to an explicitly assumed six-joint assembly using CAD hip, knee and ankle landmarks. Millimetres are converted to metres and the design faces the positive longitudinal axis. The nominal stance is bent, the soles are 52 by 34 mm, and the estimated mass remains 0.500 kg. Masses, rotor inertia, joint axes, rigid cable mappings and contact proxies are provisional; this is not calibrated hardware.",
    r"A small-target airborne diagnostic exposed unstable explicit PD with omitted motor inertia: peak joint speed was 39.0 rad/s, with 972 saturated joint-steps in 0.5 s. An assumed reflected rotor inertia of $10^{-4}$ kg m$^2$ per joint reduced peak speed to 0.774 rad/s with no saturation. This assumption is applied in CPU and GPU models and is covered by a regression test. It is not a measured actuator specification.",
    r"Exploratory runs first established standing but failed commanded locomotion. The revised task adds a sine/cosine gait phase to the original observation (29 fields), alternating foot-clearance and contact rewards, and a true-fall penalty. In the residual variant, a documented inverse-kinematics gait supplies nominal targets, and PPO learns six joint corrections. Directional linear/yaw progress terms supplement exponential tracking; contact-conditioned slip and bilateral clearance rewards discourage dragging. A regression test aligns the reward swing phase with the reference. The prior itself is not learned performance. Backward velocity and positive/negative yaw commands are included. Reward revisions and these diagnostic runs are not pooled with the earlier factorial comparison.",
    f"The learning state reached {status['updates']:,} final-objective PPO updates and {status['transitions']:,} aggregate transitions using {summary['config']['num_envs']} GPU worlds. This continuation ran for {status['wall_s']:.1f} seconds. Earlier warm-start stages contributed {sum(e['previous_transitions'] for e in ancestry):,} additional transitions. The validation-selected checkpoint has {result['selected_policy_total_transitions']:,} total training transitions including those stages. There is one independent training seed (42); initial-condition replicas do not estimate across-training-seed uncertainty. Checkpoints were selected using validation success followed by planar RMSE. Held-out test seeds were used after that selection.",
    r"\begin{table}[ht]\centering\begin{tabular}{lrrrrr}\toprule",
    r"Command & Episodes & Track (\%) & Step (\%) & Speed RMSE & Yaw RMSE\\\midrule",
]
if task.get("motor_target_filter"):
    lines[3] = (
        "The transfer-oriented revision uses 41 inputs: the original 29 plus twelve applied-target and target-velocity fields. "
        "The CPU viewer and physical sensor-to-position runtime share the same observation encoding, command ramps, inverse-kinematics reference, target filter and target-rate limits. "
        "A 1.5 Hz cadence and 8 mm reference lift replace the earlier faster gait. Training explicitly samples forward, backward, pivot and combined commands. "
        "Yaw error and complete motor-target rate/acceleration costs complement tracking. RSL-RL's mirror-consistency loss discourages lateral gait bias, without duplicating PPO likelihood samples. "
        "The robustness stage randomizes physically consistent mass/inertia, friction, gains, effort, delay and sensor bias/noise. These ranges are provisional uncertainty models, not hardware calibration. "
        "Selection ranks the weakest command group before aggregate success and worst yaw error; a good forward result cannot qualify a backward failure. "
        "No physical hardware performance or calibrated actuator equivalence is claimed."
    )
for c in commands:
    lines.append(
        f"{c['key']} & {c['episodes']} & {100 * c['success_rate']:.1f} & {100 * c['walking_success_rate']:.1f} & {c['velocity_rmse']:.4f} & {c['yaw_rmse']:.4f}"
        + r"\\"
    )
lines += [
    r"\bottomrule\end{tabular}\caption{Actual futuristic-policy held-out cases. Speed error is in m/s and yaw error in rad/s. Step success additionally requires at least two measured contact-free lifts above 3 mm by each foot on translating commands. Acceptance requires no fall, at least 95\% of ten seconds, planar RMSE at most 0.025 m/s and yaw RMSE at most 0.1 rad/s.}\end{table}",
    r"\begin{figure}[ht]\centering\includegraphics[width=.43\linewidth]{../results/figures/futuristic_robot.png}\caption{Actual futuristic CAD surfaces rendered by MuJoCo in the articulated nominal stance. Visual surfaces use simplified contact proxies.}\end{figure}",
    r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{../results/figures/futuristic_training.png}\caption{Recorded futuristic training and held-out command results. The training reward is specific to the revised task and cannot be compared directly with the initial study reward.}\end{figure}",
    f"The CPU MuJoCo keyboard transition playback survived {playback['survived_seconds']:.2f} of {playback['requested_seconds']:.1f} seconds; fall: {str(playback['fall']).lower()}. The viewer maps W/S to forward/backward speed, A/D to yaw, released keys or Space to zero command, and R to reset. Keyboard input is a request to the policy, not an imposed base motion. CPU segment tracking, after 0.5 s settling at each command change, passed: {str(playback.get('command_tracking_passed', False)).lower()}.",
]
lines.append(
    r"All predefined command groups passed the declared 95\% per-command acceptance gate and CPU keyboard tracking passed."
    if result["reliable_wasd"]
    else "The complete WASD reliability gate was not reached. The command interface is implemented, but failed directions must not be described as reliable walking."
)
lines.append(
    r"Neither real-hardware walking nor calibrated servo dynamics nor terrain generalization was measured. Machine-readable sources and hashes are in \texttt{results/futuristic\_results.json}."
)
latex = "\n\n".join(lines)
table_start = latex.index("\\begin{table}")
table_end = latex.index("\\end{table}") + len("\\end{table}")
latex = latex[:table_start] + latex[table_start:table_end].replace("\n\n", "\n") + latex[table_end:]
(ROOT / "paper/futuristic_results.tex").write_text(latex + "\n")
write_json(
    ROOT / "paper/futuristic_result_manifest.json",
    dict(
        results_sha=sha(ROOT / "results/futuristic_results.json"),
        figure_sha=sha(figure),
        sources=result["sources"],
    ),
)
print(json.dumps({k: result[k] for k in ("reliable_wasd", "commands")}, indent=2))
