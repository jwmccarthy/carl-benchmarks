# CARL Benchmarks

Unified 1v1 PPO throughput benchmarks for CARL and RLGymPPO C++. Python dependencies are locked in `uv.lock`; generated JSONL measurements under `results/` are gitignored.

## Prerequisites

CARL needs Python 3.11, CUDA with `nvcc`, and a compatible NVIDIA driver. The C++ reference additionally needs a C++20 compiler supported by CUDA, LibTorch with CUDA support, and RLGym collision meshes. The commands below assume those tools are already available in the active environment.

## Setup

```bash
git clone --recurse-submodules https://github.com/jwmccarthy/carl-benchmarks.git
cd carl-benchmarks
git -C rlgym-ppo-cpp submodule update --init --recursive
uv sync
```

## Run CARL

```bash
uv run python benchmarks/jarl_carl.py --n-sim 1024 --test
uv run python benchmarks/jarl_carl.py --n-sim 1024 --runs 1
```

Use `--epochs`, `--warmup-collections`, `--timesteps`, and `--n-sim` to change a run. The default is five seeded measurements with one warm-up collection.

## Run RLGymPPO C++

Set `TORCH_DIR` to LibTorch's `Torch` CMake package and `RLGYM_COLLISION_MESHES_DIR` to RLGym's `collision_meshes` directory:

```bash
cmake -S benchmarks/rlgymppo_cpp -B build/rlgymppo_cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR="$TORCH_DIR" \
  -DRLGYM_COLLISION_MESHES_DIR="$RLGYM_COLLISION_MESHES_DIR"
cmake --build build/rlgymppo_cpp -j 1
./build/rlgymppo_cpp/rlgymppo_cpp_benchmark --iterations 5 --log-file results/benchmarks.jsonl
```

Use `--threads`, `--games-per-thread`, `--epochs`, and `--warmup-iterations` to change C++ scaling runs.

## Benchmark Configuration

All runners use current shared-policy 1v1 self-play, lookup actions, kickoff resets, goal reward `10`, touch reward `0.1`, goal/no-touch-10-second terminals, frameskip `8`, 3x256 policy/critic networks, 100,000 collection transitions, 25,000 minibatches, entropy `0.01`, and learning rate `2e-4`.

| Runner | Native scale | Decision actors |
| --- | --- | ---: |
| RLGymPPO C++ | 16 threads x 24 games | 768 |
| CARL | `n_sim=512` | 1,024 |
| CARL | `n_sim=1,024` | 2,048 |
| CARL | `n_sim=2,048` | 4,096 |
| CARL | `n_sim=4,096` | 8,192 |
| CARL | `n_sim=8,192` | 16,384 |
| CARL | `n_sim=16,384` | 32,768 |

## Benchmark Results

Five warmed 1M-decision-step trials per row unless noted. Rates are physics ticks per second; total includes PPO updates. The C++ one-epoch scaling rows and CARL's 32,768-actor row are single warmed runs.

### One Epoch

| Method | Decision actors | Collection ticks/s | Total ticks/s |
| --- | ---: | ---: | ---: |
| RLGymPPO C++ | 256 | 213,085 | 185,711 |
| RLGymPPO C++ | 512 | 343,324 | 283,170 |
| RLGymPPO C++ | 1,024 | 636,481 | 498,684 |
| RLGymPPO C++ | 2,048 | 918,495 | 658,993 |
| RLGymPPO C++ | 4,096 | 1,110,675 | 766,028 |
| RLGymPPO C++ | 8,192 | 1,022,951 | 726,324 |
| CARL/JARL | 1,024 | 1,737,281 +/- 34,789 | 1,505,551 +/- 25,932 |
| CARL/JARL | 2,048 | 2,825,148 +/- 91,142 | 2,255,980 +/- 59,983 |
| CARL/JARL | 4,096 | 5,276,434 +/- 109,103 | 3,713,999 +/- 55,275 |
| CARL/JARL | 8,192 | 8,554,713 +/- 72,822 | 5,073,206 +/- 29,150 |
| CARL/JARL | 16,384 | 13,154,913 +/- 131,165 | 6,398,822 +/- 37,637 |
| CARL/JARL | 32,768 | 15,954,427 | 7,121,066 |

### Four Epochs

| Method | Decision actors | Collection ticks/s | Total ticks/s |
| --- | ---: | ---: | ---: |
| RLGymPPO C++ | 768 | 463,242 +/- 15,115 | 282,109 +/- 8,282 |
| CARL/JARL | 1,024 | 1,758,148 +/- 27,363 | 1,096,414 +/- 8,462 |
| CARL/JARL | 2,048 | 2,882,179 +/- 64,828 | 1,445,961 +/- 18,404 |
| CARL/JARL | 4,096 | 5,123,973 +/- 76,823 | 1,928,446 +/- 16,776 |
| CARL/JARL | 8,192 | 8,296,552 +/- 27,213 | 2,258,364 +/- 26,487 |
| CARL/JARL | 16,384 | 13,083,825 +/- 150,279 | 2,505,046 +/- 31,008 |

### Eight Epochs

| Method | Decision actors | Collection ticks/s | Total ticks/s |
| --- | ---: | ---: | ---: |
| RLGymPPO C++ | 768 | 458,270 +/- 9,571 | 212,377 +/- 3,533 |
| CARL/JARL | 1,024 | 1,708,000 +/- 35,539 | 782,865 +/- 12,176 |
| CARL/JARL | 2,048 | 2,890,048 +/- 37,146 | 973,398 +/- 8,292 |
| CARL/JARL | 4,096 | 5,134,322 +/- 120,745 | 1,195,552 +/- 6,362 |
| CARL/JARL | 8,192 | 8,341,513 +/- 126,597 | 1,296,777 +/- 12,883 |
| CARL/JARL | 16,384 | 13,040,980 +/- 207,565 | 1,384,278 +/- 10,692 |
