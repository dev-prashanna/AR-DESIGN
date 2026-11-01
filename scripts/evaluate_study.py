"""Validate all eligible finals, freeze the selection, then run untouched tests."""

import json
from pathlib import Path
import shutil
from biped_ppo.config import ROOT
from biped_ppo.evaluation import evaluate, export
from biped_ppo.tracking import write_json, sha

index = json.loads((ROOT / "results/initial_study_runs.json").read_text())
expected = {
    (condition, seed)
    for condition in ["baseline", "curriculum", "randomization", "both"]
    for seed in [0, 1, 2]
}
if {(c["condition"], c["seed"]) for c in index} != expected:
    raise RuntimeError("The matched study is incomplete; finish the missing cells first")
validated = []
for cell in index:
    run = Path(cell["path"])
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "budget_completed" or status["transitions"] != 1536000:
        raise RuntimeError(f"Incomplete fixed-budget run: {run}")
    directory = evaluate(run / "final.pt", "configs/evaluation/validation.yaml")
    summary = json.loads((directory / "summary.json").read_text())
    validated.append(dict(**cell, validation=str(directory), summary=summary))
    print("VALIDATED", cell["condition"], cell["seed"], summary["success_rate"], flush=True)
# This choice happens before any test evaluation and uses validation metrics only.
best = max(
    validated, key=lambda c: (c["summary"]["success_rate"], -c["summary"]["mean_velocity_rmse"])
)
write_json(
    ROOT / "results/selected_policy.json",
    dict(
        condition=best["condition"],
        seed=best["seed"],
        checkpoint=str(Path(best["path"]) / "best_validation.pt"),
        checkpoint_sha=sha(Path(best["path"]) / "best_validation.pt"),
        rule="highest validation success, then lowest validation planar RMSE; finals only eligible",
        validation=best["validation"],
        test_used_for_selection=False,
    ),
)
for cell in validated:
    run = Path(cell["path"])
    directory = evaluate(run / "best_validation.pt", "configs/evaluation/test.yaml")
    summary = json.loads((directory / "summary.json").read_text())
    print("TESTED", cell["condition"], cell["seed"], summary["success_rate"], flush=True)
for cell in validated:
    directory = evaluate(
        Path(cell["path"]) / "best_validation.pt", "configs/evaluation/stopping.yaml"
    )
    print("STOPPING", cell["condition"], cell["seed"], directory, flush=True)
output = export(Path(best["path"]) / "best_validation.pt", "onnx")
artifacts = ROOT / "artifacts/selected_policy"
artifacts.mkdir(parents=True, exist_ok=True)
for name in ["policy.onnx", "deployment.json"]:
    shutil.copy2(output / name, artifacts / name)
write_json(
    artifacts / "selection.json", json.loads((ROOT / "results/selected_policy.json").read_text())
)
