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

CARL and JARL are local optional source dependencies expected at `../CARL` and `../JARL`:

```bash
uv sync --extra carl
```

Building CARL requires a CUDA toolkit with `nvcc`. The RocketSim baseline does not require CARL or a CUDA toolkit.

## Smoke Tests

```bash
uv run python -m carl_benchmarks.sb3_rocketsim --smoke-test
uv run --extra carl python -m carl_benchmarks.jarl_carl --smoke-test
```

## Matched PPO Updates

Match actor rollout size, minibatch size, and epoch count. This pair collects 65,536 actor decisions and performs 40 gradient updates per PPO collection:

```bash
uv run python -m carl_benchmarks.sb3_rocketsim \
  --num-matches 16 --workers 1 --frameskip 4 \
  --rollout-steps 2048 --batch-size 16384 --epochs 10 \
  --total-timesteps 100000000 --seed 0

uv run --extra carl python -m carl_benchmarks.jarl_carl \
  --n-sim 512 --frameskip 4 \
  --rollout-steps 64 --batch-size 16384 --epochs 10 \
  --total-timesteps 100000000 --seed 0
```

For RocketSim throughput, vary `--workers` while keeping its PPO settings fixed. For CARL throughput, sweep `--n-sim` with a rollout length that preserves the desired global actor rollout:

| `n_sim` | Actor slots | Rollout steps for 65,536 actor decisions |
| ---: | ---: | ---: |
| 64 | 128 | 512 |
| 128 | 256 | 256 |
| 256 | 512 | 128 |
| 512 | 1,024 | 64 |
| 1,024 | 2,048 | 32 |

The runners currently compare end-to-end stacks. RLGym's `DefaultObs` and CARL's native observation layouts differ, so this is not yet a controlled observation-equivalence experiment.
