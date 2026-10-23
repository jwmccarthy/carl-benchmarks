# CARL Benchmarks

End-to-end PPO throughput benchmarks for RLGym/RocketSim and CARL. CARL and JARL (an experimental Torch-based RL framework that enables GPU-resident learning) are included as Git submodules.

## Quickstart

Clone the repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/jwmccarthy/carl-benchmarks.git
cd carl-benchmarks
```

CARL requires Python 3.11, a CUDA toolkit with `nvcc`, and a compatible NVIDIA driver. The build detects the local GPU compute capability automatically.

```bash
uv sync
```

Verify CARL's current-policy 1v1 self-play runner:

```bash
uv run python benchmarks/jarl_carl.py --test --n-sim 256
```

Print the Python and C++ RLGymPPO reference profiles and derived CARL configurations:

```bash
uv run python benchmarks/reference_runner.py
```

Run five seeded CARL measurements using the C++ parity profile:

```bash
uv run python benchmarks/jarl_carl.py --profile rlgymppo_cpp --n-sim 1024
```

Results are appended to `results/benchmarks.jsonl`. Use `--runs 1` for a single measurement.

## Reference-Parity Profiles

`benchmarks/reference_configs.py` defines immutable RLGymPPO reference profiles and derived CARL configurations. Python uses 32 processes and 50,000 collection/minibatch transitions. C++ uses 16 x 24 games (384 environments), 100,000 collection transitions, and 25,000 minibatches. Both use one epoch, `(256, 256, 256)` policy and critic networks, and frameskip 8. CARL sweeps `n_sim` over 512, 1024, 2048, 4096, and 8192; every 1v1 simulation contributes two decision environments.

```bash
uv run python benchmarks/reference_runner.py
uv run python benchmarks/reference_runner.py --profile rlgymppo_cpp --n-sim 1024 --collection-seconds 12.5 --ppo-update-seconds 0.8
```

CARL uses the current shared policy for both players, with no snapshot pool. Its results report `collection_ticks_per_second`, which excludes PPO update time, and `total_ticks_per_second`, which includes it. The Python and C++ reference submodules retain their upstream build/run requirements; native reference result collection is not yet wired into this repository's JSONL format.
