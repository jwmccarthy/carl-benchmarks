# CARL Benchmarks

Throughput benchmarks comparing SB3 PPO with RLGym/RocketSim against JARL PPO with CARL. CARL and JARL are included as Git submodules.

## Quickstart

Clone and install:

```bash
git clone --recurse-submodules https://github.com/jwmccarthy/carl-benchmarks.git
cd carl-benchmarks
uv sync
```

CARL builds require Python 3.11, CUDA `nvcc`, and a compatible NVIDIA driver. GPU architecture is detected automatically.

The RL benchmarks use fixed Nexto-style reward settings: `shaping_scale=1.0`, `goal_scale=10.0`, with no reward annealing.

Test the runners:

```bash
uv run python -m benchmarks.rl.sb3_rocketsim --test
uv run python -m benchmarks.rl.jarl_carl --test --n-sim 1024
```

Run the benchmarks:

```bash
uv run python -m benchmarks.rl.sb3_rocketsim --n-sim 1
uv run python -m benchmarks.rl.jarl_carl --n-sim 1024
```

Simulation-only benchmarks hold random actions for 120 physics ticks:

```bash
uv run python -m benchmarks.sim.rocketsim --n-sim 4
uv run python -m benchmarks.sim.carl --n-sim 1024
```

RL benchmarks run once by default. Use `--runs 5` for repeated seeds. Results are written to `results/benchmarks.jsonl`; episode logs default to engine/config-specific files such as `results/episodes_carl_jarl_n1024.jsonl`. Pass `--episode-log-file` to override the path.
