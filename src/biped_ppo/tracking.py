import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
import numpy as np
import torch
from .config import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def append_json(path, data):
    with open(path, "a") as f:
        f.write(json.dumps(data, default=str, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def new_run(cfg, kind="train", parent=None):
    from .diagnostics import doctor

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{kind}_{uuid.uuid4().hex[:8]}"
    path = Path(os.getenv("BIPED_RUNS", ROOT / "runs")) / run_id
    path.mkdir(parents=True)
    manifest = dict(
        run_id=run_id,
        created_utc=stamp,
        kind=kind,
        config=asdict(cfg),
        hardware=doctor(),
        code_sha=git("rev-parse", "HEAD"),
        working_tree=git("status", "--short"),
        diff=git("diff", "HEAD"),
        command=sys.argv,
        parent_checkpoint=parent,
        assets={
            str(p.relative_to(ROOT)): sha(p)
            for p in (ROOT / "assets/robot").rglob("*")
            if p.is_file()
        },
        config_hash=hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest(),
        config_files={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "configs").rglob("*.yaml")},
        lock_hash=sha(ROOT / "uv.lock"),
        simulation_state_resume="reset at resume; not bitwise restoration",
    )
    write_json(path / "manifest.json", manifest)
    write_json(ROOT / "experiments/manifests" / f"{run_id}.json", manifest)
    (path / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())
    (path / "robot.yaml").write_bytes((ROOT / "configs/robot/custom.yaml").read_bytes())
    return path
