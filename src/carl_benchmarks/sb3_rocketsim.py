"""Minimal SB3 PPO baseline using RLGym and RocketSim."""

import argparse

import gymnasium as gym
import numpy as np
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


class RocketSimEnv(gym.Env):
    """Train one car against a uniformly sampled lookup-table opponent."""

    metadata = {"render_modes": []}

    def __init__(self, frameskip: int = 4, max_ticks: int = 36_000, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.env = RLGym(
            state_mutator=MutatorSequence(
                FixedTeamSizeMutator(1, 1), KickoffMutator(),
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
            self.opponent: np.asarray([self.rng.integers(self.action_space.n)], dtype=np.int64),
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if min(args.num_envs, args.frameskip, args.max_ticks, args.timesteps) < 1:
        parser.error("environment and training sizes must be positive")

    def make_env(index: int):
        return lambda: RocketSimEnv(args.frameskip, args.max_ticks, args.seed + index)

    env = make_vec_env(make_env(0), n_envs=args.num_envs, seed=args.seed)
    if args.smoke_test:
        observations = env.reset()
        observations, _, _, _ = env.step(np.zeros(args.num_envs, dtype=np.int64))
        print(f"smoke passed: observation {observations.shape}, actions [0, 89]")
        env.close()
        return

    model = PPO("MlpPolicy", env, device="cpu", seed=args.seed, verbose=1)
    model.learn(args.timesteps)
    env.close()


if __name__ == "__main__":
    main()
