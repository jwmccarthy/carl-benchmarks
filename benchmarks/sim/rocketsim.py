import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.results import append_result
from benchmarks.rl.sb3_rocketsim import RocketSimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-action RocketSim throughput")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--n-sim", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    args = parser.parse_args()
    if args.n_sim < 1:
        parser.error("--n-sim must be positive")
    envs = [RocketSimEnv(frameskip=120, seed=args.seed + i) for i in range(args.n_sim)]
    for i, env in enumerate(envs):
        env.reset(seed=args.seed + i)
    rngs = [np.random.default_rng(args.seed + i) for i in range(args.n_sim)]

    started = time.perf_counter()
    for _ in range(args.steps):
        for index, env in enumerate(envs):
            _, _, terminated, truncated, _ = env.step(
                rngs[index].integers(env.action_space.n)
            )
            if terminated or truncated:
                env.reset()

    elapsed = time.perf_counter() - started

    for env in envs:
        env.close()

    transitions = args.steps * args.n_sim
    steps_per_second = transitions / elapsed

    append_result(
        args.log_file,
        {
            "engine": "rocketsim_random",
            "steps": transitions,
            "n_sim": args.n_sim,
            "physics_ticks": transitions * 120,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_per_second,
            "seed": args.seed,
        },
    )
    print(
        f"{steps_per_second:.0f} steps/s ({transitions * 120 / elapsed:.0f} physics ticks/s)"
    )


if __name__ == "__main__":
    main()
