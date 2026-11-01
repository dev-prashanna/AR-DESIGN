"""Figures/tables only from actual run artifacts; missing data stay missing."""

import json
from pathlib import Path
import subprocess
from .config import ROOT
from .tracking import write_json, sha


def analyze(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory).resolve()
    out = ROOT / "results/aggregates"
    out.mkdir(exist_ok=True)
    manifests = []
    summaries = []
    fig, ax = plt.subplots(figsize=(7, 4))
    found = False
    for path in sorted(directory.rglob("metrics.jsonl")):
        records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if not records:
            continue
        x = [r["transitions"] for r in records]
        y = [r["mean_step_reward"] for r in records]
        ax.plot(x, y, alpha=0.6, label=path.parent.name[-8:])
        manifests.append(
            dict(
                run=str(path.parent),
                metrics_sha=sha(path),
                updates=len(records),
                final_transitions=x[-1],
                last_reward=y[-1],
            )
        )
        found = True
    if found:
        ax.set(
            xlabel="Aggregate environment transitions",
            ylabel="Mean task reward per policy step",
            title="Observed runs; raw curves, no smoothing",
        )
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(ROOT / "results/figures/learning.png", dpi=160)
    plt.close(fig)
    for p in directory.rglob("summary.json"):
        if "evaluation_" in p.parent.name:
            summaries.append(dict(path=str(p), sha=sha(p), **json.loads(p.read_text())))
    result = dict(
        runs=manifests,
        evaluations=summaries,
        uncertainty="No across-seed interval until matched independent runs exist",
        results_status="preliminary methods/pipeline verification; not a completed ablation study",
    )
    write_json(out / "results.json", result)
    return out


def build_paper(results):
    results = Path(results).resolve()
    source = results / "results.json"
    if not source.exists():
        raise ValueError("Run biped analyze first; no validated results.json found")
    data = json.loads(source.read_text())
    audit = json.loads((ROOT / "docs/model_audit.json").read_text())
    numbers = (
        f"\\newcommand{{\\RobotMass}}{{{audit['mass_kg']:.3f}}}\n"
        f"\\newcommand{{\\CompletedRuns}}{{{len(data['runs'])}}}\n"
        f"\\newcommand{{\\StandingHeight}}{{{audit['standing']['min_height_m']:.4f}}}\n"
    )
    (ROOT / "paper/generated.tex").write_text(numbers)
    claims = {
        "robot_mass": {
            "source": "docs/model_audit.json",
            "sha": sha(ROOT / "docs/model_audit.json"),
        },
        "standing_height": {
            "source": "docs/model_audit.json",
            "sha": sha(ROOT / "docs/model_audit.json"),
        },
        "completed_runs": data["runs"],
        "evaluations": data["evaluations"],
        "ablation_comparison": "NOT MEASURED: full matched seed matrix absent",
    }
    write_json(ROOT / "paper/result_manifest.json", claims)
    proc = subprocess.run(
        ["tectonic", "main.tex"], cwd=ROOT / "paper", text=True, capture_output=True
    )
    (ROOT / "paper/build.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError("LaTeX build failed; see paper/build.log")
    return ROOT / "paper/main.pdf"
