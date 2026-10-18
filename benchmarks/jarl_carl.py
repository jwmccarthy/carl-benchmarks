import argparse
import math
import time

import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium.vector.utils import batch_space
from torch.optim import Adam

from carl.gymnasium import CARLTorchVectorEnv
from jarl.collect import CriticCapture, LogProbCapture, Runner
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

from actions import carl_lookup_table


class LookupCARLEnv:

    def __init__(self, env: CARLTorchVectorEnv, seed: int) -> None:
        self.env = env
        self.n_envs = env.n_sim
        self.device = env.device
        self.generator = th.Generator(device=self.device).manual_seed(seed)
        self.lookup = carl_lookup_table(self.device)

        self.blue_indices = th.arange(env.n_sim, device=self.device) * env.n_cars
        self.orange_indices = self.blue_indices + 1

        self.single_observation_space = env.single_observation_space
        self.single_action_space = gym.spaces.Discrete(90)
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)
        self.action_space = batch_space(self.single_action_space, self.n_envs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[self.blue_indices]

    def step(self, action):
        action = th.as_tensor(action, dtype=th.long, device=self.device).reshape(-1)
        if action.shape != (self.n_envs,):
            raise ValueError(f"expected {self.n_envs} action IDs, got {tuple(action.shape)}")
        if ((action < 0) | (action >= 90)).any():
            raise ValueError("action IDs must be in [0, 89]")

        opponent = th.randint(
            90,
            (self.n_envs,),
            generator=self.generator,
            device=self.device,
        )
        actions = th.empty((self.env.n_envs,), dtype=th.long, device=self.device)
        actions[self.blue_indices] = action
        actions[self.orange_indices] = opponent

        observation, reward, terminated, truncated, _ = self.env.step(self.lookup[actions])
        return (
            observation[self.blue_indices],
            reward[self.blue_indices],
            terminated[self.blue_indices],
            truncated[self.blue_indices],
            {},
        )

    def close(self):
        self.env.close()


class FeedForwardCategoricalPolicy(CategoricalPolicy):

    def _logits(self, observation: th.Tensor) -> th.Tensor:
        return CompositeNet.forward(self, observation)

    def dist(self, observation: th.Tensor):
        return th.distributions.Categorical(logits=self._logits(observation))

    def action(self, observation: th.Tensor) -> th.Tensor:
        return self._logits(observation).argmax(dim=-1)

    def sample(self, observation: th.Tensor) -> th.Tensor:
        return self.dist(observation).sample()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JARL PPO in CARL")
    parser.add_argument("--n-sim", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--total-batch-size", type=int, default=32_768)
    parser.add_argument("--minibatch-size", type=int, default=32_768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if min(
        args.n_sim,
        args.frameskip,
        args.max_ticks,
        args.timesteps,
        args.total_batch_size,
        args.minibatch_size,
        args.epochs,
    ) < 1:
        parser.error("simulation and training sizes must be positive")

    if args.total_batch_size % args.n_sim:
        parser.error("--total-batch-size must divide evenly across --n-sim")
    if args.total_batch_size % args.minibatch_size:
        parser.error("--minibatch-size must divide --total-batch-size")

    th.manual_seed(args.seed)
    rollout_steps = args.total_batch_size // args.n_sim

    base = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        normalize=True,
        discrete_actions=True,
    )
    env = LookupCARLEnv(base, args.seed)

    observation = env.reset()

    if args.smoke_test:
        next_observation, _, _, _, _ = env.step(
            th.zeros(env.n_envs, dtype=th.long, device=env.device),
        )
        if next_observation.shape != observation.shape:
            raise RuntimeError("CARL observation shape changed after a step")
        print(f"smoke passed: observation {tuple(observation.shape)}, actions [0, 89]")
        env.close()
        return

    policy = FeedForwardCategoricalPolicy(
        foot=FlattenEncoder(),
        body=MLP(dims=[64, 64], func=nn.Tanh),
    ).build(env).to(env.device)

    critic = Critic(
        foot=FlattenEncoder(),
        body=MLP(dims=[64, 64], func=nn.Tanh),
    ).build(env).to(env.device)

    buffer = RolloutBuffer(rollout_steps, env.n_envs, env.device, copy_on_finish=False)
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

    gradient_steps = collections * args.epochs * (
        args.total_batch_size // args.minibatch_size
    )
    print(
        f"completed {transitions} transitions in {elapsed:.2f}s "
        f"({transitions / elapsed:.0f} transitions/s); "
        f"{collections} collections, {gradient_steps} gradient updates",
    )
    env.close()


if __name__ == "__main__":
    main()
