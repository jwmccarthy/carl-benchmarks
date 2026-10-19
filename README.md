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
