import time
import torch
import psutil
from dataclasses import replace
from .config import RunConfig
from .task import make_env
from .training import TerminalAwareWrapper
from .resources import ResourceMonitor


def benchmark_one(count, steps=100):
    monitor = ResourceMonitor().start()
    started = time.monotonic()
    env = TerminalAwareWrapper(make_env(replace(RunConfig(), num_envs=count)))
    action = torch.zeros(count, 6, device="cuda:0")
    try:
        with torch.inference_mode():
            for _ in range(20):
                env.step(action)
            torch.cuda.synchronize()
            initialization = time.monotonic() - started
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            for _ in range(steps):
                env.step(action)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - t
        return dict(
            num_envs=count,
            policy_steps=steps,
            aggregate_transitions=count * steps,
            simulation_including_task_transitions_per_second=count * steps / elapsed,
            aggregate_physics_substeps_per_second=count * steps * 10 / elapsed,
            measured_wall_s=elapsed,
            initialization_and_warmup_s=initialization,
            peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),
            gpu_free_bytes=torch.cuda.mem_get_info()[0],
            ram_rss_bytes=psutil.Process().memory_info().rss,
            peak_device_vram_bytes=monitor.peak_device_vram_bytes,
            peak_process_ram_bytes=monitor.peak_process_ram_bytes,
            precision="float32",
            note="No PPO optimizer; not end-to-end training throughput",
        )
    finally:
        env.close()
        monitor.close()
