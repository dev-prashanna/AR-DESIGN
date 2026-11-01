# Performance protocol

Measured simulator/task benchmarks use 20 policy-step warmup then 50 timed policy
steps per world; CUDA is synchronized at timing boundaries. Initialization includes
warmup and cached/uncached kernel loading. Float32, 2 ms physics and 20 ms policy.
The aggregate physics-substep rate is transitions/s * 10. No PPO optimizer is
included in simulator numbers. Full records are results/benchmark.json.

VRAM is sampled from nvidia-smi every 0.5 s and includes all device usage, plus a
separate PyTorch-allocator high water mark. The former includes Warp/context and
other processes; the latter excludes allocations outside PyTorch. Short peaks
between samples can be missed. RAM uses process RSS sampled at the same cadence.
The benchmark process per environment count releases its context before the next.

The measured aggregate task/simulator rate rose from about 4,449 transitions/s
(32 worlds) to 8,794 (64), 16,262 (128), and 31,120 (256). The sampled device
peak was about 228 MiB and free device memory about 7.40 GiB in these short
benchmarks. Select 256 worlds for the pilot, with a new explicit configuration
and PPO batch 256*24=6,144. This does not establish the PPO training rate; the
32-world smoke training measured roughly 1,233--1,519 transitions/s including
rollout and optimization. Recheck training resource use at 256 worlds.
