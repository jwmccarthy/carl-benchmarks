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

## Benchmark Results

Measurements ran on an Intel Core i5-13600K CPU, NVIDIA GeForce RTX 3060 GPU (12 GiB VRAM, driver 560.94), and 15 GiB system RAM reported by the host. All results use 1,015,808 transitions, 31 PPO collections, 310 gradient updates, a 32,768-transition total/minibatch, 10 epochs, and frameskip 4. Values with five runs are reported as mean +/- sample standard deviation.

### RocketSim

| Learner | `n_sim` | Runs | Time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| SB3 PPO, CPU | 1 | 1 | 418.91 s | 2,425 transitions/s |
| SB3 PPO, CPU | 4 | 5 | 385.78 +/- 2.66 s | 2,633 +/- 18 transitions/s |
| SB3 PPO, CPU | 8 | 5 | 326.03 +/- 4.76 s | 3,116 +/- 45 transitions/s |
| SB3 PPO, CPU | 16 | 5 | 286.84 +/- 2.16 s | 3,542 +/- 27 transitions/s |
| JARL PPO, CUDA learner | 4 | 5 | 789.40 +/- 6.83 s | 1,287 +/- 11 transitions/s |
| JARL PPO, CUDA learner | 8 | 5 | 460.88 +/- 12.36 s | 2,205 +/- 58 transitions/s |
| JARL PPO, CUDA learner | 16 | 1 | 272.18 s | 3,732 transitions/s |

### CARL

| Learner | `n_sim` | Runs | Time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| JARL PPO, CUDA | 256 | 2 | 14.57 +/- 1.90 s | 69,732 +/- 1,901 transitions/s |
| JARL PPO, CUDA | 1,024 | 1 | 6.16 s | 164,824 transitions/s |
| JARL PPO, CUDA | 2,048 | 2 | 4.52 +/- 0.02 s | 224,796 +/- 1,197 transitions/s |
| JARL PPO, CUDA | 4,096 | 1 | 3.73 s | 272,032 transitions/s |
