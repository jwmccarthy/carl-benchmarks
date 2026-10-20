# CARL Benchmarks

Throughput benchmarks comparing SB3 PPO with RLGym/RocketSim against JARL PPO with CARL. CARL and JARL are included as Git submodules. RL runners live in `benchmarks/rl`; simulation-only runners live in `benchmarks/sim`.

## Quickstart

```bash
git clone --recurse-submodules https://github.com/jwmccarthy/carl-benchmarks.git
cd carl-benchmarks
uv sync
```

CARL builds require Python 3.11, CUDA `nvcc`, and a compatible NVIDIA driver. GPU architecture is detected automatically.

Run interface tests:

```bash
uv run python -m benchmarks.rl.sb3_rocketsim --test
uv run python -m benchmarks.rl.jarl_carl --test --n-sim 1024
```

Run five seeded measurements:

```bash
uv run python -m benchmarks.rl.sb3_rocketsim --n-sim 1
uv run python -m benchmarks.rl.jarl_carl --n-sim 1024
```

The RocketSim RL runner uses the fixed Nexto-style RLGym reward (`shaping_scale=1.0`, `goal_scale=10.0`); there is no annealing. CARL retains its existing reward path. For simulation-only measurements, use `uv run python benchmarks/sim/rocketsim.py` or `uv run python benchmarks/sim/carl.py`. Each random action is held for 120 physics ticks. RocketSim `--n-sim` environments are stepped concurrently with a thread pool in both runners.

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
