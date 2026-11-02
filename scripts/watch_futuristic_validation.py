"""Evaluate immutable interval checkpoints while a training process runs."""

import argparse
import json
import time
from pathlib import Path
from biped_ppo.config import ROOT
from biped_ppo.evaluation import evaluate
from biped_ppo.tracking import write_json

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--updates", type=int, nargs="+", default=[1000, 2500, 5000, 7500, 10000])
parser.add_argument("--suite", default="configs/evaluation/futuristic_validation.yaml")
args = parser.parse_args()
history = []
for update in args.updates:
    checkpoint = args.run / f"update_{update:06d}.pt"
    while not checkpoint.exists():
        if (args.run / "status.json").exists():
            print("Training ended before requested update", update, flush=True)
            break
        time.sleep(3)
    if not checkpoint.exists():
        break
    directory = evaluate(checkpoint, args.suite)
    summary = json.loads((directory / "summary.json").read_text())
    episodes = [json.loads(s) for s in (directory / "episodes.jsonl").read_text().splitlines()]
    commands = []
    for command in sorted(set(tuple(e["command"]) for e in episodes)):
        cases = [e for e in episodes if tuple(e["command"]) == command]
        commands.append(
            dict(
                command=command,
                success_rate=sum(e["success"] for e in cases) / len(cases),
                velocity_rmse=sum(e["velocity_rmse"] for e in cases) / len(cases),
                yaw_rmse=sum(e["yaw_rmse"] for e in cases) / len(cases),
            )
        )
    entry = dict(update=update, evaluation=str(directory), summary=summary, commands=commands)
    history.append(entry)
    write_json(args.run / "validation_history.json", history)
    write_json(ROOT / "results/futuristic_validation_history.json", history)
    print("VALIDATION_RESULT=" + json.dumps(entry), flush=True)
