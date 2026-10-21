import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from benchmarks.results import append_episode, append_result
from benchmarks.rl.nexto_reward import NextoReward


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
            reward_fn=NextoReward(),
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


class ThreadedRocketSimVecEnv(VecEnv):
    def __init__(self, n_envs: int, frameskip: int, max_ticks: int, seed: int):
        self.envs = [
            RocketSimEnv(frameskip, max_ticks, seed + index) for index in range(n_envs)
        ]
        super().__init__(
            n_envs, self.envs[0].observation_space, self.envs[0].action_space
        )
        self.executor = ThreadPoolExecutor(max_workers=n_envs)
        self.actions = None
        self._seeds = [seed + index for index in range(n_envs)]
        self.episode_returns = np.zeros(n_envs, dtype=np.float32)

    def reset(self):
        observations = [
            env.reset(seed=seed)[0] for env, seed in zip(self.envs, self._seeds)
        ]
        self._seeds = [None] * self.num_envs
        return np.stack(observations)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = list(
            self.executor.map(
                lambda pair: pair[0].step(int(pair[1])),
                zip(self.envs, self.actions),
            )
        )
        observations, rewards, dones, infos = [], [], [], []
        for env, (observation, reward, terminated, truncated, info) in zip(
            self.envs, results
        ):
            done = terminated or truncated
            info = dict(info)
            if done:
                info["terminal_observation"] = observation
                observation = env.reset()[0]
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            self.episode_returns[len(observations) - 1] += reward
            if done:
                info["episode"] = {
                    "r": float(self.episode_returns[len(observations) - 1]),
                }
                self.episode_returns[len(observations) - 1] = 0
            infos.append(info)
        return np.stack(observations), np.asarray(rewards), np.asarray(dones), infos

    def close(self):
        self.executor.shutdown()
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(env, attr_name) for env in self.envs]

    def set_attr(self, attr_name, value, indices=None):
        for env in self.envs:
            setattr(env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(env, method_name)(*args, **kwargs) for env in self.envs]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        seed = 0 if seed is None else seed
        self._seeds = [seed + index for index in range(self.num_envs)]
        return self._seeds


class EpisodeCallback(BaseCallback):
    def __init__(self, path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.path = path

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                append_episode(
                    self.path,
                    "rocketsim_sb3",
                    self.num_timesteps,
                    episode["r"],
                )
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 PPO in RocketSim")
    parser.add_argument("--n-sim", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=36_000)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--total-batch-size", type=int, default=32_768)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--minibatch-size", type=int, default=32_768)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--episode-log-file", type=Path, default=Path("results/episodes.jsonl")
    )
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.n_sim < 1:
        parser.error("--n-sim must be positive")
    rollout_steps = args.rollout_steps or args.total_batch_size // args.n_sim
    if rollout_steps * args.n_sim != args.total_batch_size:
        parser.error("--rollout-steps * --n-sim must equal --total-batch-size")

    def make_vector_env(seed: int):
        return ThreadedRocketSimVecEnv(args.n_sim, args.frameskip, args.max_ticks, seed)

    env = make_vector_env(args.seed)

    if args.test:
        observations = env.reset()
        observations, _, _, _ = env.step(np.zeros(args.n_sim, dtype=np.int64))
        print(f"test passed: observation {observations.shape}, actions [0, 89]")
        env.close()
        return

    env.close()
    elapsed_times = []
    throughputs = []

    for run_index in range(args.runs):
        seed = args.seed + run_index
        env = make_vector_env(seed)
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
        model.learn(args.timesteps, callback=EpisodeCallback(args.episode_log_file))
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
                "n_sim": args.n_sim,
                "frameskip": args.frameskip,
                "total_batch_size": args.total_batch_size,
                "minibatch_size": args.minibatch_size,
                "epochs": args.epochs,
                "seed": seed,
                "run_index": run_index,
                "threaded": args.n_sim > 1,
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
