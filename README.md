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

Verify each benchmark:

```bash
uv run python benchmarks/sb3_rocketsim.py --test
uv run python benchmarks/jarl_carl.py --test --n-sim 256
```

Run five seeded measurements by default:

```bash
uv run python benchmarks/sb3_rocketsim.py --n-sim 1
uv run python benchmarks/jarl_carl.py --n-sim 1024
```

| Argument | RocketSim default | CARL default |
| --- | ---: | ---: |
| `--n-sim` | 1 | 1,024 |
| `--frameskip` | 4 | 4 |
| `--max-ticks` | 36,000 | 36,000 |
| `--timesteps` | 1,000,000 | 1,000,000 |
| `--total-batch-size` | 32,768 | 32,768 |
| `--minibatch-size` | 32,768 | 32,768 |
| `--epochs` | 8 | 8 |
| `--seed` | 0 | 0 |
| `--runs` | 5 | 5 |

Results are appended to `results/benchmarks.jsonl`. Use `--runs 1` for a single measurement.

## Reference-Parity Profiles

`benchmarks/reference_configs.py` defines immutable RLGymPPO reference profiles and
derived CARL configurations. Python uses 32 processes, 50,000 collection/minibatch
transitions, and C++ uses 16 x 24 games (384 environments), 100,000 collection
transitions, and 25,000 minibatches. Both use one epoch, `(256, 256, 256)` policy
and critic networks, and frameskip 8. CARL sweeps `n_sim` over 512, 1024, 2048,
4096, and 8192; every 1v1 simulation contributes two decision environments.

```bash
uv run python benchmarks/reference_runner.py
uv run python benchmarks/reference_runner.py --profile rlgymppo_cpp --n-sim 1024 --collection-seconds 12.5 --ppo-update-seconds 0.8
```

The reported `collection_ticks_per_second` divides collection ticks by collection
time only. `total_ticks_per_second` divides the same ticks by collection plus PPO
update time. Self-play and snapshot-pool parity are deliberately not claimed: the
CLI rejects `--self-play` until opponent scheduling is implemented and validated.
