import argparse
import statistics
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from results import append_result


class RocketSimEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, frameskip: int = 4, max_ticks: int = 36_000, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.env = RLGym(
            state_mutator=MutatorSequence(
                FixedTeamSizeMutator(1, 1),
                KickoffMutator(),
            ),
            obs_builder=DefaultObs(),
            action_parser=RepeatAction(LookupTableAction(), repeats=frameskip),
            reward_fn=GoalReward(),
            transition_engine=RocketSimEngine(),
            termination_cond=GoalCondition(),
            truncation_cond=TimeoutCondition(max_ticks / 120),
        )

        initial = self.env.reset()
        self.player, self.opponent = self.env.agents
        _, observation_size = self.env.observation_space(self.player)
        _, action_count = self.env.action_space(self.player)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(action_count)
        self._observation = initial[self.player]

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        observations = self.env.reset()
        self._observation = observations[self.player]
        return self._observation, {}

    def step(self, action: int):
        actions = {
            self.player: np.asarray([action], dtype=np.int64),
            self.opponent: np.asarray(
                [self.rng.integers(self.action_space.n)],
                dtype=np.int64,
            ),
        }

        observations, rewards, terminated, truncated = self.env.step(actions)
        self._observation = observations[self.player]
        return (
            self._observation,
            float(rewards[self.player]),
            bool(terminated[self.player]),
            bool(truncated[self.player]),
            {},
        )

    def close(self):
        self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 PPO in RocketSim")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--total-batch-size", type=int, default=32_768)
    parser.add_argument("--minibatch-size", type=int, default=32_768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if (
        min(
            args.num_envs,
            args.frameskip,
            args.max_ticks,
            args.timesteps,
            args.total_batch_size,
            args.minibatch_size,
            args.epochs,
            args.runs,
        )
        < 1
    ):
        parser.error("environment and training sizes must be positive")
    if args.total_batch_size % args.num_envs:
        parser.error("--total-batch-size must divide evenly across --num-envs")
    if args.total_batch_size % args.minibatch_size:
        parser.error("--minibatch-size must divide --total-batch-size")

    rollout_steps = args.total_batch_size // args.num_envs

    def make_env(seed: int, index: int):
        return lambda: RocketSimEnv(args.frameskip, args.max_ticks, seed + index)

    env = make_vec_env(make_env(args.seed, 0), n_envs=args.num_envs, seed=args.seed)

    if args.smoke_test:
        observations = env.reset()
        observations, _, _, _ = env.step(np.zeros(args.num_envs, dtype=np.int64))
        print(f"smoke passed: observation {observations.shape}, actions [0, 89]")
        env.close()
        return

    env.close()
    elapsed_times = []
    throughputs = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        env = make_vec_env(make_env(seed, 0), n_envs=args.num_envs, seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=rollout_steps,
            batch_size=args.minibatch_size,
            n_epochs=args.epochs,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [64, 64], "activation_fn": nn.Tanh},
            device="cpu",
            seed=seed,
            verbose=1,
        )
        started = time.perf_counter()
        model.learn(args.timesteps)
        elapsed = time.perf_counter() - started
        collections = model.num_timesteps // args.total_batch_size
        gradient_steps = (
            collections * args.epochs * (args.total_batch_size // args.minibatch_size)
        )
        transitions_per_second = model.num_timesteps / elapsed
        append_result(
            args.log_file,
            {
                "engine": "rocketsim_sb3",
                "num_envs": args.num_envs,
                "frameskip": args.frameskip,
                "total_batch_size": args.total_batch_size,
                "minibatch_size": args.minibatch_size,
                "epochs": args.epochs,
                "seed": seed,
                "run_index": run_index,
                "device": "cpu",
                "elapsed_seconds": elapsed,
                "transitions": model.num_timesteps,
                "transitions_per_second": transitions_per_second,
                "collections": collections,
                "gradient_updates": gradient_steps,
            },
        )
        print(
            f"run {run_index}: completed {model.num_timesteps} transitions in {elapsed:.2f}s "
            f"({transitions_per_second:.0f} transitions/s); "
            f"{collections} collections, {gradient_steps} gradient updates",
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
