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

All runners use current shared-policy 1v1 self-play, lookup actions, kickoff resets, goal reward `10`, touch reward `0.1`, goal/no-touch-10-second terminals, frameskip `8`, 3x256 policy/critic networks, 100,000 collection transitions, 25,000 minibatches, one PPO epoch, entropy `0.01`, and learning rate `2e-4`.

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
