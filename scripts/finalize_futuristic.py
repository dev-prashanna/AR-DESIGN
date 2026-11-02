"""Wait for the long run, freeze validation selection, then test/export/report.

Never changes rewards or chooses checkpoints using test outcomes. Failures are
written to a status file; a completed compute budget is not a walking claim.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from biped_ppo.config import ROOT
from biped_ppo.evaluation import evaluate, export, selection_quality
from biped_ppo.play import play
from biped_ppo.reporting import build_paper
from biped_ppo.tracking import sha, write_json, append_json

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument(
    "--updates", type=int, nargs="+", default=[1000, 2500, 5000, 10000, 15000, 20000]
)
parser.add_argument(
    "--validation-suite", default="configs/evaluation/futuristic_walking_validation.yaml"
)
parser.add_argument("--test-suite", default="configs/evaluation/futuristic_walking_test.yaml")
parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/futuristic_policy")
args = parser.parse_args()
run = args.run.resolve()
status_path = ROOT / "results/futuristic_pipeline_status.json"
validation_suite = args.validation_suite
test_suite = args.test_suite


def status(stage, **values):
    record = dict(
        stage=stage, run=str(run), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values
    )
    write_json(status_path, record)
    print(json.dumps(record), flush=True)


def existing_evaluation(checkpoint, suite, split):
    checkpoint_hash = sha(checkpoint)
    suite_hash = sha(ROOT / suite)
    for path in sorted(checkpoint.parent.glob(f"evaluation_{split}_*/summary.json"), reverse=True):
        summary = json.loads(path.read_text())
        if summary["checkpoint_sha"] == checkpoint_hash and summary["suite_sha"] == suite_hash:
            return path.parent
    return None


def gpu_sample():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        name, util, memory, power, temp = [
            s.strip() for s in result.stdout.splitlines()[0].split(",")
        ]
        return dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            name=name,
            utilization_percent=float(util),
            memory_MiB=float(memory),
            power_W=float(power),
            temperature_C=float(temp),
        )
    return dict(error=result.stderr)


try:
    while not (run / "status.json").exists():
        sample = gpu_sample()
        append_json(run / "gpu_samples.jsonl", sample)
        latest = None
        if (run / "metrics.jsonl").exists():
            lines = (run / "metrics.jsonl").read_text().splitlines()
            if lines:
                try:
                    latest = json.loads(lines[-1])
                except json.JSONDecodeError:
                    pass
        status(
            "training_in_progress",
            gpu=sample,
            update=latest["update"] if latest else None,
            transitions=latest["transitions"] if latest else None,
            target_updates=max(args.updates),
        )
        time.sleep(30)
    training_status = json.loads((run / "status.json").read_text())
    if training_status["status"] != "budget_completed" or training_status["updates"] < max(
        args.updates
    ):
        raise RuntimeError(
            "Long training stopped before its requested budget: " + json.dumps(training_status)
        )
    status("validation_selection")
    candidates = []
    inherited = run / "inherited_validation_selection.json"
    if inherited.exists():
        old = json.loads(inherited.read_text())
        if old["suite_sha"] == sha(ROOT / validation_suite):
            candidates.append(old)
    for update in args.updates:
        checkpoint = run / f"update_{update:06d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        directory = existing_evaluation(checkpoint, validation_suite, "validation")
        if directory is None:
            directory = evaluate(checkpoint, validation_suite)
        summary = json.loads((directory / "summary.json").read_text())
        candidates.append(
            dict(
                checkpoint=str(checkpoint),
                checkpoint_sha=sha(checkpoint),
                selection_metric=summary.get(
                    "selection_metric",
                    "minimum_command_success_rate"
                    if "transfer" in validation_suite
                    else "walking_success_rate",
                ),
                selection_score=summary["minimum_command_success_rate"]
                if "transfer" in validation_suite
                else summary["walking_success_rate"],
                walking_success_rate=summary["walking_success_rate"],
                worst_yaw_rmse=summary["maximum_command_yaw_rmse"],
                success_rate=summary["success_rate"],
                velocity_rmse=summary["mean_velocity_rmse"],
                suite_sha=summary["suite_sha"],
                evaluation=str(directory),
            )
        )
    selected = max(candidates, key=selection_quality)
    selected["rule"] = (
        "weakest command success, aggregate walking success, worst yaw, then planar RMSE; test unused"
        if "transfer" in validation_suite
        else "predefined checkpoints; validation bilateral-step success, tracking success, then planar RMSE; test unused"
    )
    best = run / "best_validation.pt"
    if Path(selected["checkpoint"]).resolve() != best.resolve():
        shutil.copy2(selected["checkpoint"], best)
    write_json(run / "validation_selection.json", selected)
    assert sha(best) == selected["checkpoint_sha"]
    status("held_out_test", selection=selected)
    test = existing_evaluation(best, test_suite, "test")
    if test is None:
        test = evaluate(best, test_suite)
    status("export_and_keyboard_playback", test=str(test))
    export_directory = export(best, "onnx")
    if "transfer" in validation_suite:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_policy_runtime.py"),
                "--checkpoint",
                str(best),
            ],
            cwd=ROOT,
            check=True,
        )
    artifact = args.artifact_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, artifact / "policy.pt")
    for name in ("policy.onnx", "deployment.json"):
        shutil.copy2(export_directory / name, artifact / name)
    if (export_directory / "runtime_source").exists():
        shutil.copytree(
            export_directory / "runtime_source", artifact / "runtime_source", dirs_exist_ok=True
        )
    if "transfer" in validation_suite:
        shutil.copy2(
            ROOT / "docs/physical_policy_contract.json", artifact / "physical_policy_contract.json"
        )
    write_json(artifact / "selection.json", selected)
    playback = artifact / "keyboard_playback.json"
    play(artifact / "policy.pt", seconds=30, headless=True, output=playback)
    status("rendering_and_paper")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/render_futuristic.py"), "--evaluation", str(test)],
        cwd=ROOT,
        check=True,
    )
    for index, name in enumerate(
        (
            "standing",
            "walking_forward",
            "walking_backward",
            "turn_left",
            "turn_right",
            "forward_left",
            "forward_right",
        )
    ):
        shutil.copy2(test / f"command_{index:02d}.mp4", artifact / f"{name}.mp4")
        shutil.copy2(test / f"command_{index:02d}_final.png", artifact / f"{name}_final.png")
    shutil.copy2(test / "video_manifest.json", artifact / "video_manifest.json")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_futuristic.py"),
            "--run",
            str(run),
            "--test",
            str(test),
            "--playback",
            str(playback),
        ],
        cwd=ROOT,
        check=True,
    )
    pdf = build_paper(ROOT / "results/aggregates")
    results = json.loads((ROOT / "results/futuristic_results.json").read_text())
    (artifact / "README.md").write_text(
        "# Futuristic MicroDuck policy\n\n"
        "Launch with `scripts/play_futuristic` from the repository. W/S walk; A/D turn; release or Space stops; R resets.\n\n"
        f"Full WASD acceptance passed: {results['reliable_wasd']}. Read results/futuristic_results.json for every direction and failure.\n\n"
        "The checkpoint includes the learned residual actor and normalization. Deployment also requires the documented inverse-kinematics prior and provisional MuJoCo model. Hardware operation is untested.\n"
    )
    status(
        "pipeline_completed",
        walking_gate_passed=results["reliable_wasd"],
        test=str(test),
        checkpoint=str(artifact / "policy.pt"),
        checkpoint_sha=sha(artifact / "policy.pt"),
        paper=str(pdf),
        paper_sha=sha(pdf),
    )
except BaseException as exc:
    status("pipeline_failed", error=f"{type(exc).__name__}: {exc}")
    (run / "finalization_error.log").write_text(traceback.format_exc())
    raise
