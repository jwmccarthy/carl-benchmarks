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
uv run python benchmarks/jarl_rocketsim.py --test
uv run python benchmarks/jarl_carl.py --test --n-sim 256
```

Run five seeded measurements by default:

```bash
uv run python benchmarks/sb3_rocketsim.py --n-sim 1
uv run python benchmarks/jarl_rocketsim.py --n-sim 16
uv run python benchmarks/jarl_carl.py --n-sim 1024
```

| Argument | RocketSim default | CARL default |
| --- | ---: | ---: |
| `--n-sim` | 1 | 256 |
| `--frameskip` | 4 | 4 |
| `--max-ticks` | 36,000 | 36,000 |
| `--timesteps` | 1,000,000 | 1,000,000 |
| `--total-batch-size` | 32,768 | 32,768 |
| `--minibatch-size` | 32,768 | 32,768 |
| `--epochs` | 8 | 8 |
| `--seed` | 0 | 0 |
| `--runs` | 5 | 5 |

Results are appended to `results/benchmarks.jsonl`. Use `--runs 1` for a single measurement.
