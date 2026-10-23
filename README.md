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

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th align="center">Decision actors</th>
      <th align="center">Collection ticks/s</th>
      <th align="center">Total ticks/s</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="6">RLGymPPO C++</td><td align="right">256</td><td align="right">213,085</td><td align="right">185,711</td></tr>
    <tr><td align="right">512</td><td align="right">343,324</td><td align="right">283,170</td></tr>
    <tr><td align="right">1,024</td><td align="right">636,481</td><td align="right">498,684</td></tr>
    <tr><td align="right">2,048</td><td align="right">918,495</td><td align="right">658,993</td></tr>
    <tr><td align="right">4,096</td><td align="right">1,110,675</td><td align="right">766,028</td></tr>
    <tr><td align="right">8,192</td><td align="right">1,022,951</td><td align="right">726,324</td></tr>
    <tr><td rowspan="6">CARL/JARL</td><td align="right">1,024</td><td align="right">1,737,281 +/- 34,789</td><td align="right">1,505,551 +/- 25,932</td></tr>
    <tr><td align="right">2,048</td><td align="right">2,825,148 +/- 91,142</td><td align="right">2,255,980 +/- 59,983</td></tr>
    <tr><td align="right">4,096</td><td align="right">5,276,434 +/- 109,103</td><td align="right">3,713,999 +/- 55,275</td></tr>
    <tr><td align="right">8,192</td><td align="right">8,554,713 +/- 72,822</td><td align="right">5,073,206 +/- 29,150</td></tr>
    <tr><td align="right">16,384</td><td align="right">13,154,913 +/- 131,165</td><td align="right">6,398,822 +/- 37,637</td></tr>
    <tr><td align="right">32,768</td><td align="right">15,954,427</td><td align="right">7,121,066</td></tr>
  </tbody>
</table>
