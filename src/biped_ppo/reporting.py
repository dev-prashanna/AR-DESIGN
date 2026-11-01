"""Artifact-linked plots and run-level analysis; absent experiments stay absent."""

import json
from pathlib import Path
import subprocess
import numpy as np
from .config import ROOT
from .tracking import write_json, sha

CONDITIONS = ("baseline", "curriculum", "randomization", "both")


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return None
    draws = np.random.default_rng(0).choice(values, (10000, len(values)), replace=True).mean(1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def analyze(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory).resolve()
    out = ROOT / "results/aggregates"
    out.mkdir(exist_ok=True)
    index_path = ROOT / "results/initial_study_runs.json"
    study = json.loads(index_path.read_text()) if index_path.exists() else []
    diagnostics = []
    for path in sorted(directory.glob("*/metrics.jsonl")):
        records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if not records:
            continue
        diagnostics.append(
            dict(
                run=str(path.parent),
                metrics_sha=sha(path),
                updates=len(records),
                final_transitions=records[-1]["transitions"],
                last_reward=records[-1]["mean_step_reward"],
            )
        )
    policies = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for cell in study:
        path = Path(cell["path"])
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "status.json").read_text())
        metrics = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines()]
        condition = cell["condition"]
        ax = axes.flat[CONDITIONS.index(condition)]
        x = np.asarray([r["transitions"] for r in metrics])
        y = np.asarray([r["mean_step_reward"] for r in metrics])
        color = f"C{cell['seed']}"
        ax.plot(x, y, color=color, alpha=0.18, linewidth=0.7)
        window = min(20, len(y))
        if window:
            ax.plot(
                x[window - 1 :],
                np.convolve(y, np.ones(window) / window, mode="valid"),
                color=color,
                label=f"seed {cell['seed']}",
                linewidth=1.2,
            )
        ax.set(title=condition, ylabel="Task reward / policy step")
        ax.grid(alpha=0.15)
        row = dict(
            condition=condition,
            training_seed=cell["seed"],
            run_id=path.name,
            path=str(path),
            reused=cell.get("reused", False),
            status=status,
            config=manifest["config"],
            config_sha=manifest["config_hash"],
            metrics_sha=sha(path / "metrics.jsonl"),
            checkpoint_sha=sha(path / "final.pt") if (path / "final.pt").exists() else None,
            validation=None,
            test=None,
            stopping=None,
        )
        for split in ["validation", "test", "stopping"]:
            candidates = sorted(path.glob(f"evaluation_{split}_*/summary.json"))
            eligible = [
                p
                for p in candidates
                if json.loads(p.read_text())["checkpoint_sha"] == row["checkpoint_sha"]
            ]
            if eligible:
                p = eligible[-1]
                summary = json.loads(p.read_text())
                episodes = [
                    json.loads(x) for x in (p.parent / "episodes.jsonl").read_text().splitlines()
                ]
                moving = [e for e in episodes if e["command"][0] > 0.001]
                summary.update(
                    summary_sha=sha(p),
                    path=str(p.parent),
                    moving_success_rate=float(np.mean([e["success"] for e in moving])),
                    mean_saturation_fraction=float(
                        np.mean([e["saturation_fraction"] for e in episodes])
                    ),
                    mean_joint_limit_fraction=float(
                        np.mean([e["joint_limit_fraction"] for e in episodes])
                    ),
                    mean_foot_slip_rms_m_s=float(
                        np.mean([e["foot_slip_rms_m_s"] for e in episodes])
                    ),
                )
                row[split] = summary
        policies.append(row)
    for ax in axes.flat:
        ax.set_xlabel("Aggregate environment transitions")
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("Within-condition learning curves: raw (faint), trailing 20-update mean (solid)")
    fig.tight_layout()
    if policies:
        fig.savefig(ROOT / "results/figures/learning.png", dpi=160)
    plt.close(fig)
    if not policies:
        (ROOT / "results/figures/learning.png").unlink(missing_ok=True)
    aggregates = []
    for condition in CONDITIONS:
        selected = [p for p in policies if p["condition"] == condition and p["test"] is not None]
        if not selected:
            continue
        result = dict(
            condition=condition,
            n_independent_training_seeds=len(selected),
            seeds=[p["training_seed"] for p in selected],
            run_ids=[p["run_id"] for p in selected],
        )
        for metric in [
            "success_rate",
            "moving_success_rate",
            "fall_rate",
            "mean_duration_s",
            "mean_velocity_rmse",
            "mean_saturation_fraction",
            "mean_joint_limit_fraction",
            "mean_foot_slip_rms_m_s",
        ]:
            vals = [p["test"][metric] for p in selected]
            result[metric] = dict(
                seed_values=vals,
                mean=float(np.mean(vals)),
                sample_std=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                bootstrap_95_ci=interval(vals),
            )
        aggregates.append(result)
    if aggregates:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for index, a in enumerate(aggregates):
            for ax, metric in zip(axes, ["success_rate", "moving_success_rate"]):
                vals = a[metric]["seed_values"]
                ax.scatter(
                    np.full(len(vals), index), np.array(vals) * 100, color="C0", alpha=0.6, zorder=3
                )
                mean = a[metric]["mean"] * 100
                ci = a[metric]["bootstrap_95_ci"]
                if ci:
                    ax.errorbar(
                        index,
                        mean,
                        yerr=[[mean - ci[0] * 100], [ci[1] * 100 - mean]],
                        color="black",
                        fmt="s",
                        capsize=4,
                        zorder=4,
                    )
                ax.set(
                    xticks=range(len(aggregates)),
                    xticklabels=[a["condition"] for a in aggregates],
                    ylim=(-3, 103),
                    ylabel="Success (%)",
                    title=metric.replace("_", " "),
                )
                ax.tick_params(axis="x", rotation=20)
                ax.grid(axis="y", alpha=0.15)
        fig.suptitle(
            "Held-out test: dots are independent training seeds; bars are run-level bootstrap 95% CI"
        )
        fig.tight_layout()
        fig.savefig(ROOT / "results/figures/test_success.png", dpi=160)
        plt.close(fig)
    if not aggregates:
        (ROOT / "results/figures/test_success.png").unlink(missing_ok=True)
    complete = len(policies) == 12 and all(
        p["test"] is not None and p["status"]["status"] == "budget_completed" for p in policies
    )
    result = dict(
        runs=diagnostics,
        study_policies=policies,
        condition_aggregates=aggregates,
        figures={
            name: sha(ROOT / "results/figures" / name)
            for name in ["learning.png", "test_success.png"]
            if (ROOT / "results/figures" / name).exists()
        },
        study_complete=complete,
        independent_policies=len(policies),
        uncertainty="Percentile bootstrap of independent training-run means; 10000 draws; RNG seed 0; no episode-level pseudoreplication",
        smoothing="Raw per-update rewards and trailing 20-update arithmetic mean; no interpolation across seeds",
        results_status="completed short-budget exploratory study"
        if complete
        else "preliminary methods; study incomplete",
    )
    write_json(out / "results.json", result)
    return out


def build_paper(results):
    results = Path(results).resolve()
    source = results / "results.json"
    if not source.exists():
        raise ValueError("Run biped analyze first; results.json is absent")
    data = json.loads(source.read_text())
    for name, expected_hash in data.get("figures", {}).items():
        figure = ROOT / "results/figures" / name
        if not figure.exists() or sha(figure) != expected_hash:
            raise ValueError(f"Figure provenance mismatch: {name}; rerun analyze")
    audit = json.loads((ROOT / "docs/model_audit.json").read_text())
    numbers = (
        f"\\newcommand{{\\RobotMass}}{{{audit['mass_kg']:.3f}}}\n"
        f"\\newcommand{{\\CompletedRuns}}{{{len(data['runs'])}}}\n"
        f"\\newcommand{{\\StandingHeight}}{{{audit['standing']['min_height_m']:.4f}}}\n"
        f"\\newcommand{{\\StudyRuns}}{{{len(data['study_policies'])}}}\n"
        f"\\newcommand{{\\StudyStatus}}{{{data['results_status']}}}\n"
    )
    (ROOT / "paper/generated.tex").write_text(numbers)
    lines = [
        "\\begin{tabular}{lrrr}\\toprule",
        "Condition & Seeds & Success (\\%) & Planar RMSE (m/s)\\\\\\midrule",
    ]
    for a in data["condition_aggregates"]:
        lines.append(
            f"{a['condition']} & {a['n_independent_training_seeds']} & {100 * a['success_rate']['mean']:.1f} & {a['mean_velocity_rmse']['mean']:.4f}\\\\"
        )
    if not data["condition_aggregates"]:
        lines.append("\\multicolumn{4}{c}{NOT MEASURED: no held-out test artifacts}\\\\")
    lines += ["\\bottomrule\\end{tabular}"]
    (ROOT / "paper/study_results.tex").write_text("\n".join(lines) + "\n")
    findings = ""
    if data["study_complete"]:
        if all(a["moving_success_rate"]["mean"] == 0 for a in data["condition_aggregates"]):
            findings += "No policy passed the full moving-command acceptance criterion on the fixed held-out suite. Standing cases account for the observed aggregate successes. This does not imply an absence of forward displacement; heading error, speed error, or falls can each cause failure. "
        findings += "The sample contains three independent training seeds per condition. Degenerate bootstrap intervals from identical seed outcomes do not establish certainty outside these sampled policies and test conditions. "
    selection = ROOT / "results/selected_policy.json"
    if selection.exists():
        selected = json.loads(selection.read_text())
        candidates = [
            p for p in data["study_policies"] if p["checkpoint_sha"] == selected["checkpoint_sha"]
        ]
        if candidates and candidates[0]["test"] is not None:
            episodes_path = Path(candidates[0]["test"]["path"]) / "episodes.jsonl"
            episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
            forward = [e for e in episodes if e["command"] == [0.045, 0, 0]]
            if forward:
                distance = np.mean([e["forward_distance_m"] for e in forward])
                yaw = np.mean([e["yaw_rmse"] for e in forward])
                findings += f"For the validation-selected policy, the 0.045 m/s command produced mean forward displacement {distance:.3f} m and yaw-rate RMSE {yaw:.3f} rad/s over the corresponding held-out cases. Thus forward motion occurred, but heading control failed the declared criterion. "
    if not findings:
        findings = "NOT MEASURED: study outcomes are incomplete."
    (ROOT / "paper/study_findings.tex").write_text(findings + "\n")

    claims = dict(
        robot_mass={"source": "docs/model_audit.json", "sha": sha(ROOT / "docs/model_audit.json")},
        standing_height={
            "source": "docs/model_audit.json",
            "sha": sha(ROOT / "docs/model_audit.json"),
        },
        diagnostic_runs=data["runs"],
        study_policies=data["study_policies"],
        condition_aggregates=data["condition_aggregates"],
        study_complete=data["study_complete"],
        analysis_source_sha=sha(Path(__file__)),
        results_sha=sha(source),
    )
    write_json(ROOT / "paper/result_manifest.json", claims)
    proc = subprocess.run(
        ["tectonic", "main.tex"], cwd=ROOT / "paper", text=True, capture_output=True
    )
    (ROOT / "paper/build.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError("LaTeX build failed; see paper/build.log")
    return ROOT / "paper/main.pdf"
