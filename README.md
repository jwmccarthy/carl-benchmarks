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
uv run python -m carl_benchmarks.sb3_rocketsim --smoke-test --frameskip 4
```

Train the minimal RocketSim baseline with a random lookup-table opponent:

```bash
uv run python -m carl_benchmarks.sb3_rocketsim \
  --num-envs 8 --frameskip 4 --timesteps 1000000
```

The runner holds total samples per PPO collection constant. Its defaults collect 32,768 actor transitions per collection, run 10 epochs of full-batch PPO, and time a requested 10,000,000 transitions. It prints actual transitions, elapsed time, collections, and gradient updates when finished.

For a serial run:

```bash
uv run python -m carl_benchmarks.sb3_rocketsim --num-envs 1
```

For a 16-environment rollout with the same algorithmic batch:

```bash
uv run python -m carl_benchmarks.sb3_rocketsim --num-envs 16
```

## Matched PPO Updates

The CARL/JARL runner will be added next. It will use the same 90-action table and reward configuration, then expose matched PPO rollout, minibatch, and epoch settings.
