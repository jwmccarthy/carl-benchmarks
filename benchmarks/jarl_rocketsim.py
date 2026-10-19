import argparse
import math
import statistics
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
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.modules import MLP
from jarl.modules.base import CompositeNet
from jarl.modules.encoder import FlattenEncoder
from jarl.modules.layer import orthogonal_init
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
    parser.add_argument("--n-sim", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--total-batch-size", type=int, default=32_768)
    parser.add_argument("--minibatch-size", type=int, default=32_768)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    rollout_steps = args.total_batch_size // args.n_sim

    def environment_factory(seed: int):
        environment_index = 0

        def make_environment():
            nonlocal environment_index
            environment = RocketSimEnv(
                args.frameskip,
                args.max_ticks,
                seed + environment_index,
            )
            environment_index += 1
            return environment

        return make_environment

    env = SyncGymEnv(environment_factory(args.seed), args.n_sim)
    observation = env.reset()

    if args.test:
        next_observation, _, _, _, _ = env.step(np.zeros(args.n_sim, dtype=np.int64))
        if next_observation.shape != observation.shape:
            raise RuntimeError("RocketSim observation shape changed after a step")
        print(f"test passed: observation {observation.shape}, actions [0, 89]")
        return

    device = torch.device("cuda")
    elapsed_times = []
    throughputs = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        torch.manual_seed(seed)
        env = SyncGymEnv(environment_factory(seed), args.n_sim)
        policy = (
            FeedForwardCategoricalPolicy(
                foot=FlattenEncoder(),
                body=MLP(
                    dims=[64, 64],
                    func=nn.Tanh,
                    out_init_func=orthogonal_init(std=0.01),
                ),
            )
            .build(env)
            .to(device)
        )
        critic = (
            Critic(
                foot=FlattenEncoder(),
                body=MLP(
                    dims=[64, 64],
                    func=nn.Tanh,
                    out_init_func=orthogonal_init(std=1.0),
                ),
            )
            .build(env)
            .to(device)
        )

        buffer = RolloutBuffer(rollout_steps, env.n_envs, device, copy_on_finish=False)
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
            optimizer_step=OptimizerStep(
                (policy, critic),
                Adam((*policy.parameters(), *critic.parameters()), lr=3e-4),
                max_grad_norm=0.5,
            ),
            section="PPO",
        )

        started = time.perf_counter()
        Trainer(runner, buffer, Algorithm(update), OnPolicySchedule()).run(
            args.timesteps
        )
        elapsed = time.perf_counter() - started

        collections = math.ceil(args.timesteps / args.total_batch_size)
        transitions = collections * args.total_batch_size
        gradient_updates = (
            collections * args.epochs * (args.total_batch_size // args.minibatch_size)
        )
        transitions_per_second = transitions / elapsed

        append_result(
            args.log_file,
            {
                "engine": "rocketsim_jarl",
                "n_sim": args.n_sim,
                "frameskip": args.frameskip,
                "total_batch_size": args.total_batch_size,
                "minibatch_size": args.minibatch_size,
                "epochs": args.epochs,
                "seed": seed,
                "run_index": run_index,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "elapsed_seconds": elapsed,
                "transitions": transitions,
                "transitions_per_second": transitions_per_second,
                "collections": collections,
                "gradient_updates": gradient_updates,
            },
        )

        print(
            f"run {run_index}: completed {transitions} transitions in {elapsed:.2f}s "
            f"({transitions_per_second:.0f} transitions/s); "
            f"{collections} collections, {gradient_updates} gradient updates",
        )
        elapsed_times.append(elapsed)
        throughputs.append(transitions_per_second)

    elapsed_stdev = statistics.stdev(elapsed_times) if args.runs > 1 else 0.0
    throughput_stdev = statistics.stdev(throughputs) if args.runs > 1 else 0.0
    print(
        f"summary ({args.runs} runs): elapsed {statistics.mean(elapsed_times):.2f}s "
        f"+/- {elapsed_stdev:.2f}s; throughput {statistics.mean(throughputs):.0f} "
        f"+/- {throughput_stdev:.0f} transitions/s",
    )


if __name__ == "__main__":
    main()
