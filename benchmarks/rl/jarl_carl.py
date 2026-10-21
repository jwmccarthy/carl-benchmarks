import argparse
import math
import statistics
import time
from pathlib import Path

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
from jarl.log.logger import Logger
from jarl.runtime import OnPolicySchedule, Trainer
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from benchmarks.rl.actions import carl_lookup_table
from benchmarks.rl.carl_nexto_reward import NextoReward
from benchmarks.results import append_episode, append_result


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
            raise ValueError(
                f"expected {self.n_envs} action IDs, got {tuple(action.shape)}"
            )
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

        observation, reward, terminated, truncated, _ = self.env.step(
            self.lookup[actions]
        )
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


class EpisodeLogger(Logger):
    def __init__(self, path: Path, run_index: int) -> None:
        super().__init__()
        self.path = path
        self.run_index = run_index
        self.started = time.perf_counter()

    def update(self, metrics, step=None):
        super().update(metrics, step)
        reward = metrics.get("episode", {}).get("reward")
        if reward is not None:
            append_episode(
                self.path,
                "carl_jarl",
                step or 0,
                float(reward),
                time.perf_counter() - self.started,
                self.run_index,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JARL PPO in CARL")
    parser.add_argument("--n-sim", type=int, default=1024)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--total-batch-size", type=int, default=32_768)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--minibatch-size", type=int, default=32_768)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    parser.add_argument("--episode-log-file", type=Path)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    if args.episode_log_file is None:
        args.episode_log_file = Path(f"results/episodes_carl_jarl_n{args.n_sim}.jsonl")

    if args.rollout_steps is not None:
        if args.rollout_steps < 1:
            parser.error("--rollout-steps must be positive")
        args.total_batch_size = args.rollout_steps * args.n_sim
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
        reward_funcs=(NextoReward(1, 1, shaping_scale=1.0, goal_scale=10.0),),
    )
    env = LookupCARLEnv(base, args.seed)

    observation = env.reset()

    if args.test:
        next_observation, _, _, _, _ = env.step(
            th.zeros(env.n_envs, dtype=th.long, device=env.device),
        )
        if next_observation.shape != observation.shape:
            raise RuntimeError("CARL observation shape changed after a step")
        print(f"test passed: observation {tuple(observation.shape)}, actions [0, 89]")
        env.close()
        return

    env.close()
    elapsed_times = []
    throughputs = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        th.manual_seed(seed)
        base = CARLTorchVectorEnv(
            n_sim=args.n_sim,
            n_blue=1,
            n_orange=1,
            seed=seed,
            frameskip=args.frameskip,
            max_ticks=args.max_ticks,
            normalize=True,
            discrete_actions=True,
            reward_funcs=(NextoReward(1, 1, shaping_scale=1.0, goal_scale=10.0),),
        )
        env = LookupCARLEnv(base, seed)
        policy = (
            FeedForwardCategoricalPolicy(
                foot=FlattenEncoder(),
                body=MLP(dims=[64, 64], func=nn.Tanh),
            )
            .build(env)
            .to(env.device)
        )

        critic = (
            Critic(
                foot=FlattenEncoder(),
                body=MLP(dims=[64, 64], func=nn.Tanh),
            )
            .build(env)
            .to(env.device)
        )

        buffer = RolloutBuffer(
            rollout_steps, env.n_envs, env.device, copy_on_finish=False
        )
        runner = Runner(
            env, policy, buffer, captures=(LogProbCapture(), CriticCapture(critic))
        )

        update = Update(
            transforms=(GAE(gamma=0.99, lambda_=0.95),),
            sampler=RolloutMinibatches(
                batch_size=args.minibatch_size, epochs=args.epochs
            ),
            loss=PPOLoss(
                policy,
                critic,
                PPOConfig(clip=0.2, value_clip=None, value_coef=0.5, entropy_coef=0.0),
            ),
            optimizer_step=IndependentOptimizerSteps(
                OptimizerStep(
                    policy, Adam(policy.parameters(), lr=3e-4), max_grad_norm=0.5
                ),
                OptimizerStep(
                    critic, Adam(critic.parameters(), lr=3e-4), max_grad_norm=0.5
                ),
            ),
            section="PPO",
        )

        logger = EpisodeLogger(args.episode_log_file, run_index)
        logger.register_progress_metric(
            "episode",
            "reward",
            label="episode reward",
            format_spec=".3f",
        )

        th.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        Trainer(
            runner,
            buffer,
            Algorithm(update),
            OnPolicySchedule(),
            logger=logger,
        ).run(args.timesteps)
        th.cuda.synchronize()
        elapsed = time.perf_counter() - started

        peak_allocated_mib = th.cuda.max_memory_allocated() / 2**20
        peak_reserved_mib = th.cuda.max_memory_reserved() / 2**20
        allocated_mib = th.cuda.memory_allocated() / 2**20

        collections = math.ceil(args.timesteps / args.total_batch_size)
        transitions = collections * args.total_batch_size

        gradient_steps = (
            collections * args.epochs * (args.total_batch_size // args.minibatch_size)
        )
        transitions_per_second = transitions / elapsed

        append_result(
            args.log_file,
            {
                "engine": "carl_jarl",
                "n_sim": args.n_sim,
                "frameskip": args.frameskip,
                "total_batch_size": args.total_batch_size,
                "minibatch_size": args.minibatch_size,
                "epochs": args.epochs,
                "seed": seed,
                "run_index": run_index,
                "elapsed_seconds": elapsed,
                "transitions": transitions,
                "transitions_per_second": transitions_per_second,
                "collections": collections,
                "gradient_updates": gradient_steps,
                "peak_allocated_mib": peak_allocated_mib,
                "peak_reserved_mib": peak_reserved_mib,
                "allocated_mib": allocated_mib,
            },
        )

        print(
            f"run {run_index}: completed {transitions} transitions in {elapsed:.2f}s "
            f"({transitions_per_second:.0f} transitions/s); "
            f"{collections} collections, {gradient_steps} gradient updates; "
            f"peak VRAM {peak_allocated_mib:.0f} MiB allocated / "
            f"{peak_reserved_mib:.0f} MiB reserved",
        )

        elapsed_times.append(elapsed)
        throughputs.append(transitions_per_second)
        env.close()

    elapsed_stdev = statistics.stdev(elapsed_times) if args.runs > 1 else 0.0
    throughput_stdev = statistics.stdev(throughputs) if args.runs > 1 else 0.0

    print(
        f"summary ({args.runs} runs): elapsed {statistics.mean(elapsed_times):.2f}s "
        f"+/- {elapsed_stdev:.2f}s; throughput {statistics.mean(throughputs):.0f} "
        f"+/- {throughput_stdev:.0f} transitions/s",
    )


if __name__ == "__main__":
    main()
