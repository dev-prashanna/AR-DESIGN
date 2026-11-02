"""Launch durable GPU training, interval validation and final reporting."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import torch
from biped_ppo.config import ROOT, RunConfig
from biped_ppo.tracking import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=Path)
parser.add_argument("--warm-start", type=Path)
parser.add_argument("--config", type=Path)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument(
    "--validation-suite", default="configs/evaluation/futuristic_walking_validation.yaml"
)
parser.add_argument("--test-suite", default="configs/evaluation/futuristic_walking_test.yaml")
parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/futuristic_policy")
args = parser.parse_args()
if bool(args.resume) == bool(args.warm_start):
    raise ValueError("Choose exactly one resume or warm-start checkpoint")
checkpoint = (args.resume or args.warm_start).resolve()
state = torch.load(checkpoint, map_location="cpu", weights_only=False)["project"]
cfg = RunConfig.load(args.config) if args.config else RunConfig(**state["config"])
start_update = state["updates"] if args.resume else 0
if args.iterations <= start_update:
    raise ValueError("Requested update cap must exceed the saved update count")
logs = ROOT / "runs" / ("long_jobs_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
logs.mkdir(parents=True)
env = dict(os.environ)
env.pop("PYTHONPATH", None)
env.pop("VIRTUAL_ENV", None)
env["MUJOCO_GL"] = "egl"


def launch(arguments, name):
    with (logs / (name + ".log")).open("w") as stream:
        return subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), *arguments],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


transitions = args.iterations * cfg.num_envs * cfg.horizon
command = ["-m", "biped_ppo", "train"]
if args.warm_start:
    if args.config is None:
        raise ValueError("Warm-start requires an experiment configuration")
    command += ["--config", str(args.config), "--warm-start", str(checkpoint)]
else:
    command += ["--resume", str(checkpoint)]
command += [
    "--iterations",
    str(args.iterations),
    "--max-transitions",
    str(transitions),
    "--max-wall-time",
    str(cfg.max_wall_time),
]
training = launch(command, "training")
run = None
for _ in range(120):
    if training.poll() is not None:
        raise RuntimeError(f"Training exited during startup: {logs}/training.log")
    for manifest in sorted((ROOT / "runs").glob("*/manifest.json"), reverse=True):
        data = json.loads(manifest.read_text())
        if (
            data.get("parent_checkpoint") == str(checkpoint)
            and manifest.parent.stat().st_mtime >= time.time() - 120
        ):
            run = manifest.parent
            break
    if run is not None:
        break
    time.sleep(0.5)
if run is None:
    training.terminate()
    raise RuntimeError("Could not identify new training run")
if args.resume:
    for name in ("best_validation.pt", "validation_selection.json"):
        if (checkpoint.parent / name).exists():
            shutil.copy2(checkpoint.parent / name, run / name)
    if (checkpoint.parent / "validation_selection.json").exists():
        shutil.copy2(
            checkpoint.parent / "validation_selection.json",
            run / "inherited_validation_selection.json",
        )
updates = [
    u for u in (500, 1000, 2500, 5000, 10000, 15000, 20000) if start_update < u <= args.iterations
]
if args.iterations not in updates:
    updates.append(args.iterations)
update_args = [str(u) for u in updates]
validation = launch(
    [
        "scripts/watch_futuristic_validation.py",
        "--run",
        str(run),
        "--updates",
        *update_args,
        "--suite",
        args.validation_suite,
    ],
    "validation",
)
finalization = launch(
    [
        "scripts/finalize_futuristic.py",
        "--run",
        str(run),
        "--updates",
        *update_args,
        "--validation-suite",
        args.validation_suite,
        "--test-suite",
        args.test_suite,
        "--artifact-dir",
        str(args.artifact_dir),
    ],
    "finalization",
)
time.sleep(3)
for name, process in [
    ("training", training),
    ("validation", validation),
    ("finalization", finalization),
]:
    if process.poll() is not None:
        raise RuntimeError(f"{name} exited: {logs}/{name}.log")
jobs = dict(
    profile=cfg.profile,
    run=str(run),
    source_checkpoint=str(checkpoint),
    target_updates=args.iterations,
    target_transitions=transitions,
    validation_updates=updates,
    validation_suite=args.validation_suite,
    test_suite=args.test_suite,
    artifact_dir=str(args.artifact_dir),
    logs=str(logs),
    training_pid=training.pid,
    validation_pid=validation.pid,
    finalization_pid=finalization.pid,
    launched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    detached_sessions=True,
)
write_json(ROOT / "results/futuristic_jobs.json", jobs)
print(json.dumps(jobs, indent=2))
