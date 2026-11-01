from dataclasses import dataclass, fields
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    path = Path(path)
    if not path.exists():
        path = ROOT / path
    return yaml.safe_load(path.read_text())


@dataclass
class RunConfig:
    profile: str = "smoke"
    num_envs: int = 32
    horizon: int = 24
    iterations: int = 3
    max_wall_time: float = 180
    max_transitions: int = 2304
    checkpoint_interval: int = 1
    episode_seconds: float = 6
    curriculum: bool = False
    randomization: bool = False
    seed: int = 0

    @classmethod
    def load(cls, path):
        data = read(path)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {unknown}")
        result = cls(**data)
        if result.num_envs <= 0 or result.horizon <= 0 or result.num_envs * result.horizon % 4:
            raise ValueError("Rollout batch must be positive and divisible by four minibatches")
        if min(result.max_wall_time, result.max_transitions, result.iterations) <= 0:
            raise ValueError("Budgets must be positive")
        return result
