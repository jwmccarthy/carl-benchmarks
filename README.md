# CARL Benchmarks

Matched 1v1 shared-policy PPO benchmarks for:

- Stable-Baselines3 PPO with RLGym and RocketSim.
- JARL PPO with CARL.

Both runners use RLGym's discrete 90-action lookup table, a four-tick action repeat by default, and matching Nexto reward scales. Each physical match produces two actor slots, one for each car, sharing policy parameters.

## Install

Run the RocketSim baseline:

```bash
uv sync
```

CARL and JARL are local source dependencies expected at `../CARL` and `../JARL`.

Building CARL requires a CUDA toolkit with `nvcc`. The RocketSim baseline does not require CARL or a CUDA toolkit.

Run the CARL benchmark with the `rlenv` Conda environment after its CARL extension has been rebuilt:

```bash
conda run -n rlenv python benchmarks/jarl_carl.py --smoke-test --n-sim 256
```

## Smoke Tests

```bash
uv run python benchmarks/sb3_rocketsim.py --smoke-test --frameskip 4
```

Train the minimal RocketSim baseline with a random lookup-table opponent:

```bash
uv run python benchmarks/sb3_rocketsim.py \
  --num-envs 8 --frameskip 4 --timesteps 1000000
```

SB3 uses CPU policy and update execution because its vector-environment API exchanges NumPy arrays.

## JARL RocketSim

Use JARL to keep RocketSim simulation on CPU while policy inference, rollout storage, and PPO updates use CUDA:

```bash
uv run python benchmarks/jarl_rocketsim.py --num-envs 16
```

The runner holds total samples per PPO collection constant. Its defaults collect 32,768 actor transitions per collection, run 10 epochs of full-batch PPO, and time a requested 1,000,000 transitions. It prints actual transitions, elapsed time, collections, and gradient updates when finished.

Completed runs append one JSON object per line to `results/benchmarks.jsonl`. Override the destination with `--log-file`.

## Logged Results

The results below use 1,015,808 transitions, 31 PPO collections, 310 gradient updates, a 32,768-transition total/minibatch, 10 epochs, frameskip 4, and seed 0. Repeated CARL configurations are averaged.

| Simulator and learner | Parallelism | Time | Throughput | Speedup vs. SB3 RocketSim |
| --- | ---: | ---: | ---: | ---: |
| RocketSim + SB3 PPO (CPU) | 1 environment | 418.91 s | 2,425 transitions/s | 1.0x |
| RocketSim + JARL PPO (CUDA learner) | 16 environments | 272.18 s | 3,732 transitions/s | 1.5x |
| CARL + JARL PPO | 256 simulations | 14.57 s | 69,732 transitions/s | 28.8x |
| CARL + JARL PPO | 1,024 simulations | 6.16 s | 164,824 transitions/s | 68.0x |
| CARL + JARL PPO | 2,048 simulations | 4.52 s | 224,796 transitions/s | 92.7x |

These are end-to-end throughput measurements. RocketSim and CARL still use different observation layouts, so they are not yet controlled learning-quality results.

For a serial run:

```bash
uv run python benchmarks/sb3_rocketsim.py --num-envs 1
```

For a 16-environment rollout with the same algorithmic batch:

```bash
uv run python benchmarks/sb3_rocketsim.py --num-envs 16
```

## CARL

Run a CARL smoke test:

```bash
conda run -n rlenv python benchmarks/jarl_carl.py --smoke-test --n-sim 256
```

The CARL runner trains blue against a random lookup-table opponent and uses CARL's native goal reward, matching the RocketSim baseline. It uses the same fixed PPO defaults and timing summary. For the default 32,768-transition total batch:

| `n_sim` | Rollout steps |
| ---: | ---: |
| 256 | 128 |
| 512 | 64 |
| 1,024 | 32 |
| 2,048 | 16 |

```bash
conda run -n rlenv python benchmarks/jarl_carl.py --n-sim 512
```
