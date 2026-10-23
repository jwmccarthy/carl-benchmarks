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
from jarl.sample import RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.transform import GAE

from actions import carl_lookup_table
from reference_configs import REFERENCE_PROFILES, carl_rollout
from results import append_result


class LookupCARLEnv:

    def __init__(self, env: CARLTorchVectorEnv) -> None:
        self.env = env
        self.n_envs = env.n_envs
        self.device = env.device
        self.lookup = carl_lookup_table(self.device)

        self.single_observation_space = env.single_observation_space
        self.single_action_space = gym.spaces.Discrete(90)
        self.observation_space = batch_space(self.single_observation_space, self.n_envs)
        self.action_space = batch_space(self.single_action_space, self.n_envs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        action = th.as_tensor(action, dtype=th.long, device=self.device).reshape(-1)
        if action.shape != (self.n_envs,):
            raise ValueError(
                f"expected {self.n_envs} action IDs, got {tuple(action.shape)}"
            )
        if ((action < 0) | (action >= 90)).any():
            raise ValueError("action IDs must be in [0, 89]")

        return self.env.step(self.lookup[action])

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
    parser.add_argument("--profile", choices=REFERENCE_PROFILES, default="rlgymppo_cpp")
    parser.add_argument("--n-sim", type=int, default=1024)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    profile = REFERENCE_PROFILES[args.profile]
    rollout = carl_rollout(profile, args.n_sim)

    base = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=profile.frameskip,
        max_ticks=args.max_ticks,
        normalize=True,
        discrete_actions=True,
    )
    env = LookupCARLEnv(base)

    observation = env.reset()

    if args.test:
        next_observation, _, _, _, _ = env.step(
            th.zeros(env.n_envs, dtype=th.long, device=env.device),
        )
        if next_observation.shape != observation.shape:
            raise RuntimeError("CARL observation shape changed after a step")
        print(
            f"test passed: observation {tuple(observation.shape)}, actions [0, 89]; "
            f"{env.n_envs} current-policy self-play actors"
        )
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
            frameskip=profile.frameskip,
            max_ticks=args.max_ticks,
            normalize=True,
            discrete_actions=True,
        )
        env = LookupCARLEnv(base)
        policy = (
            FeedForwardCategoricalPolicy(
                foot=FlattenEncoder(),
                body=MLP(dims=list(profile.policy_layers), func=nn.Tanh),
            )
            .build(env)
            .to(env.device)
        )

        critic = (
            Critic(
                foot=FlattenEncoder(),
                body=MLP(dims=list(profile.critic_layers), func=nn.Tanh),
            )
            .build(env)
            .to(env.device)
        )

        buffer = RolloutBuffer(
            rollout.rollout_steps, env.n_envs, env.device, copy_on_finish=False
        )
        runner = Runner(
            env, policy, buffer, captures=(LogProbCapture(), CriticCapture(critic))
        )

        update = Update(
            transforms=(GAE(gamma=0.99, lambda_=0.95),),
            sampler=RolloutMinibatches(
                batch_size=profile.minibatch_size, epochs=profile.epochs
            ),
            loss=PPOLoss(
                policy,
                critic,
                PPOConfig(
                    clip=0.2,
                    value_clip=None,
                    value_coef=0.5,
                    entropy_coef=profile.entropy_coefficient,
                ),
            ),
            optimizer_step=IndependentOptimizerSteps(
                OptimizerStep(
                    policy,
                    Adam(policy.parameters(), lr=profile.learning_rate),
                    max_grad_norm=0.5,
                ),
                OptimizerStep(
                    critic,
                    Adam(critic.parameters(), lr=profile.learning_rate),
                    max_grad_norm=0.5,
                ),
            ),
            section="PPO",
        )

        th.cuda.reset_peak_memory_stats()
        algorithm = Algorithm(update)
        runner.reset()
        collections = math.ceil(args.timesteps / rollout.actual_collection_transitions)
        collection_seconds = 0.0
        ppo_update_seconds = 0.0
        for _ in range(collections):
            th.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(rollout.rollout_steps):
                runner.step()
            th.cuda.synchronize()
            collection_seconds += time.perf_counter() - started

            th.cuda.synchronize()
            started = time.perf_counter()
            algorithm.update(buffer.finish())
            buffer.clear()
            th.cuda.synchronize()
            ppo_update_seconds += time.perf_counter() - started

        elapsed = collection_seconds + ppo_update_seconds

        peak_allocated_mib = th.cuda.max_memory_allocated() / 2**20
        peak_reserved_mib = th.cuda.max_memory_reserved() / 2**20
        allocated_mib = th.cuda.memory_allocated() / 2**20

        transitions = collections * rollout.actual_collection_transitions
        collection_ticks = transitions * profile.frameskip
        gradient_steps = collections * profile.epochs * math.ceil(
            rollout.actual_collection_transitions / profile.minibatch_size
        )
        collection_ticks_per_second = collection_ticks / collection_seconds
        total_ticks_per_second = collection_ticks / elapsed

        append_result(
            args.log_file,
            {
                "engine": "carl_jarl",
                "profile": profile.name,
                "n_sim": args.n_sim,
                "n_cars": rollout.n_cars,
                "decision_environments": rollout.decision_environments,
                "rollout_steps": rollout.rollout_steps,
                "target_collection_transitions": profile.collection_transitions,
                "actual_collection_transitions": rollout.actual_collection_transitions,
                "frameskip": profile.frameskip,
                "minibatch_size": profile.minibatch_size,
                "epochs": profile.epochs,
                "seed": seed,
                "run_index": run_index,
                "elapsed_seconds": elapsed,
                "collection_seconds": collection_seconds,
                "ppo_update_seconds": ppo_update_seconds,
                "transitions": transitions,
                "collection_ticks": collection_ticks,
                "collection_ticks_per_second": collection_ticks_per_second,
                "total_ticks_per_second": total_ticks_per_second,
                "collections": collections,
                "gradient_updates": gradient_steps,
                "peak_allocated_mib": peak_allocated_mib,
                "peak_reserved_mib": peak_reserved_mib,
                "allocated_mib": allocated_mib,
            },
        )

        print(
            f"run {run_index}: completed {collection_ticks} ticks in {elapsed:.2f}s "
            f"({collection_ticks_per_second:.0f} collection ticks/s, "
            f"{total_ticks_per_second:.0f} total ticks/s); "
            f"{collections} collections, {gradient_steps} gradient updates; "
            f"peak VRAM {peak_allocated_mib:.0f} MiB allocated / "
            f"{peak_reserved_mib:.0f} MiB reserved",
        )
        elapsed_times.append(elapsed)
        throughputs.append(total_ticks_per_second)
        env.close()

    elapsed_stdev = statistics.stdev(elapsed_times) if args.runs > 1 else 0.0
    throughput_stdev = statistics.stdev(throughputs) if args.runs > 1 else 0.0
    print(
        f"summary ({args.runs} runs): elapsed {statistics.mean(elapsed_times):.2f}s "
        f"+/- {elapsed_stdev:.2f}s; total throughput {statistics.mean(throughputs):.0f} "
        f"+/- {throughput_stdev:.0f} ticks/s",
    )


if __name__ == "__main__":
    main()
