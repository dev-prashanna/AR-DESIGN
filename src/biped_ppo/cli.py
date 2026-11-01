import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .config import ROOT, RunConfig, read
from .tracking import write_json


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser(prog="biped")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    for name in ["validate-model", "view"]:
        p = sub.add_parser(name)
        p.add_argument("--robot", default="configs/robot/custom.yaml")
    p = sub.add_parser("benchmark")
    p.add_argument("--num-envs", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("train")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-wall-time", type=float)
    p.add_argument("--max-transitions", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--resume", type=Path)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--suite", default="configs/evaluation/validation.yaml")
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format", choices=["onnx", "jit"], default="onnx")
    p = sub.add_parser("sweep")
    p.add_argument("--config", default="configs/experiments/ablation-matrix.yaml")
    p.add_argument("--max-wall-time", type=float, required=True)
    for name in ["analyze", "paper"]:
        p = sub.add_parser(name)
        p.add_argument("--runs" if name == "analyze" else "--results", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        from .diagnostics import doctor

        result = doctor()
        write_json(ROOT / "docs/hardware.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "validate-model":
        from .robot import validate

        result = validate(args.robot)
        print(json.dumps({k: result[k] for k in ["mass_kg", "standing", "gates_passed"]}, indent=2))
        if not result["gates_passed"]:
            raise SystemExit(2)
    elif args.command == "view":
        import mujoco
        import mujoco.viewer
        from .robot import generate
        from .diagnostics import initialize

        xml, _ = generate(args.robot, ground=True, motors=True)
        model = mujoco.MjModel.from_xml_string(xml)
        data = initialize(model, read(args.robot))
        mujoco.viewer.launch(model, data)
    elif args.command == "train":
        import torch
        from .training import train

        if args.resume and args.config is None:
            cfg = RunConfig(
                **torch.load(args.resume, map_location="cpu", weights_only=False)["project"][
                    "config"
                ]
            )
        else:
            cfg = RunConfig.load(args.config or "configs/experiments/smoke.yaml")
        for key in ["seed", "max_wall_time", "max_transitions", "iterations"]:
            if getattr(args, key) is not None:
                setattr(cfg, key, getattr(args, key))
        print(train(cfg, args.resume))
    elif args.command == "benchmark":
        from .benchmark import benchmark_one

        if args.worker:
            result = benchmark_one(args.num_envs[0], args.steps)
            print("BENCHMARK_JSON=" + json.dumps(result), flush=True)
        else:
            rows = []
            for n in args.num_envs:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "biped_ppo",
                        "benchmark",
                        "--worker",
                        "--num-envs",
                        str(n),
                        "--steps",
                        str(args.steps),
                    ],
                    capture_output=True,
                    text=True,
                )
                lines = [
                    line.split("=", 1)[1]
                    for line in proc.stdout.splitlines()
                    if line.startswith("BENCHMARK_JSON=")
                ]
                row = (
                    json.loads(lines[-1])
                    if proc.returncode == 0 and lines
                    else dict(num_envs=n, error=proc.stderr[-3000:], returncode=proc.returncode)
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_json(ROOT / "results/benchmark.json", rows)
                if proc.returncode != 0:
                    break
    elif args.command == "evaluate":
        from .evaluation import evaluate

        print(evaluate(args.checkpoint, args.suite))
    elif args.command == "export":
        from .evaluation import export

        print(export(args.checkpoint, args.format))
    elif args.command == "sweep":
        from .training import train

        spec = read(args.config)
        start = time.monotonic()
        for condition, values in spec["conditions"].items():
            for seed in spec["seeds"]:
                remaining = args.max_wall_time - (time.monotonic() - start)
                if remaining <= 0:
                    return
                cfg = RunConfig.load(spec["base"])
                for key, value in values.items():
                    setattr(cfg, key, value)
                cfg.seed = seed
                cfg.max_wall_time = min(cfg.max_wall_time, remaining)
                print(condition, train(cfg), flush=True)
    else:
        from .reporting import analyze, build_paper

        print(analyze(args.runs) if args.command == "analyze" else build_paper(args.results))
