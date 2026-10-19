import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from jarl.collect import CriticCapture, LogProbCapture, Runner
from jarl.envs.gym import SyncGymEnv
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.modules import MLP
from jarl.modules.base import CompositeNet
from jarl.modules.encoder import FlattenEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import CategoricalPolicy
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from results import append_result
from sb3_rocketsim import RocketSimEnv


class FeedForwardCategoricalPolicy(CategoricalPolicy):

    def _logits(self, observation: torch.Tensor) -> torch.Tensor:
        return CompositeNet.forward(self, observation)

    def dist(self, observation: torch.Tensor):
        return torch.distributions.Categorical(logits=self._logits(observation))

    def action(self, observation: torch.Tensor) -> torch.Tensor:
        return self._logits(observation).argmax(dim=-1)

    def sample(self, observation: torch.Tensor) -> torch.Tensor:
        return self.dist(observation).sample()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JARL PPO in RocketSim")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--total-batch-size", type=int, default=32_768)
    parser.add_argument("--minibatch-size", type=int, default=32_768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("results/benchmarks.jsonl"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if min(
        args.num_envs,
        args.frameskip,
        args.max_ticks,
        args.timesteps,
        args.total_batch_size,
        args.minibatch_size,
        args.epochs,
    ) < 1:
        parser.error("environment and training sizes must be positive")
    if args.total_batch_size % args.num_envs:
        parser.error("--total-batch-size must divide evenly across --num-envs")
    if args.total_batch_size % args.minibatch_size:
        parser.error("--minibatch-size must divide --total-batch-size")
    if not torch.cuda.is_available():
        parser.error("JARL RocketSim requires CUDA")

    torch.manual_seed(args.seed)
    rollout_steps = args.total_batch_size // args.num_envs
    environment_index = 0

    def make_environment():
        nonlocal environment_index
        environment = RocketSimEnv(
            args.frameskip,
            args.max_ticks,
            args.seed + environment_index,
        )
        environment_index += 1
        return environment

    env = SyncGymEnv(make_environment, args.num_envs)
    observation = env.reset()

    if args.smoke_test:
        next_observation, _, _, _, _ = env.step(np.zeros(args.num_envs, dtype=np.int64))
        if next_observation.shape != observation.shape:
            raise RuntimeError("RocketSim observation shape changed after a step")
        print(f"smoke passed: observation {observation.shape}, actions [0, 89]")
        return

    device = torch.device("cuda")
    policy = FeedForwardCategoricalPolicy(
        foot=FlattenEncoder(),
        body=MLP(dims=[64, 64], func=nn.Tanh),
    ).build(env).to(device)
    critic = Critic(
        foot=FlattenEncoder(),
        body=MLP(dims=[64, 64], func=nn.Tanh),
    ).build(env).to(device)

    buffer = RolloutBuffer(rollout_steps, env.n_envs, device, copy_on_finish=False)
    runner = Runner(env, policy, buffer, captures=(LogProbCapture(), CriticCapture(critic)))
    update = Update(
        transforms=(GAE(gamma=0.99, lambda_=0.95),),
        sampler=RolloutMinibatches(batch_size=args.minibatch_size, epochs=args.epochs),
        loss=PPOLoss(
            policy,
            critic,
            PPOConfig(clip=0.2, value_clip=None, value_coef=0.5, entropy_coef=0.0),
        ),
        optimizer_step=IndependentOptimizerSteps(
            OptimizerStep(policy, Adam(policy.parameters(), lr=3e-4), max_grad_norm=0.5),
            OptimizerStep(critic, Adam(critic.parameters(), lr=3e-4), max_grad_norm=0.5),
        ),
        section="PPO",
    )

    started = time.perf_counter()
    Trainer(runner, buffer, Algorithm(update), OnPolicySchedule()).run(args.timesteps)
    elapsed = time.perf_counter() - started

    collections = math.ceil(args.timesteps / args.total_batch_size)
    transitions = collections * args.total_batch_size
    gradient_updates = collections * args.epochs * (
        args.total_batch_size // args.minibatch_size
    )
    transitions_per_second = transitions / elapsed

    append_result(args.log_file, {
        "engine": "rocketsim_jarl",
        "num_envs": args.num_envs,
        "frameskip": args.frameskip,
        "total_batch_size": args.total_batch_size,
        "minibatch_size": args.minibatch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "elapsed_seconds": elapsed,
        "transitions": transitions,
        "transitions_per_second": transitions_per_second,
        "collections": collections,
        "gradient_updates": gradient_updates,
    })

    print(
        f"completed {transitions} transitions in {elapsed:.2f}s "
        f"({transitions_per_second:.0f} transitions/s); "
        f"{collections} collections, {gradient_updates} gradient updates",
    )


if __name__ == "__main__":
    main()
