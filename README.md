# CARL Benchmarks

Throughput benchmarks comparing SB3 PPO with RLGym/RocketSim against JARL PPO with CARL. CARL and JARL are included as Git submodules.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/jwmccarthy/carl-benchmarks.git
cd carl-benchmarks
uv sync
```

CARL builds require Python 3.11, CUDA `nvcc`, and a compatible NVIDIA driver. GPU architecture is detected automatically.

Run interface tests:

```bash
uv run python benchmarks/sb3_rocketsim.py --test
uv run python benchmarks/jarl_carl.py --test --n-sim 1024
```

Run five seeded measurements:

```bash
uv run python benchmarks/sb3_rocketsim.py --n-sim 1
uv run python benchmarks/jarl_carl.py --n-sim 1024
```

## Defaults

| Argument | RocketSim | CARL |
| --- | ---: | ---: |
| `--n-sim` | 1 | 1,024 |
| `--frameskip` | 4 | 4 |
| `--max-ticks` | 36,000 | 36,000 |
| `--timesteps` | 1,000,000 | 1,000,000 |
| `--total-batch-size` | 32,768 | 32,768 |
| `--minibatch-size` | 32,768 | 32,768 |
| `--epochs` | 8 | 8 |
| `--runs` | 5 | 5 |

Results are appended to `results/benchmarks.jsonl`. Use `--runs 1` for a single run.
