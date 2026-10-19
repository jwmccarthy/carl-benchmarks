# CARL Benchmarks

End-to-end PPO throughput benchmarks for RLGym/RocketSim and CARL. CARL and JARL are included as Git submodules.

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
uv run python benchmarks/sb3_rocketsim.py --smoke-test
uv run python benchmarks/jarl_rocketsim.py --smoke-test
uv run python benchmarks/jarl_carl.py --smoke-test --n-sim 256
```

Run five seeded measurements by default:

```bash
uv run python benchmarks/sb3_rocketsim.py --num-envs 1
uv run python benchmarks/jarl_rocketsim.py --num-envs 16
uv run python benchmarks/jarl_carl.py --n-sim 1024
```

Each run uses a 32,768-transition PPO collection, 10 epochs, and logs individual results to `results/benchmarks.jsonl`. Use `--runs 1` for a single measurement.
