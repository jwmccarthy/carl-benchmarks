import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.results import append_result
from benchmarks.rl.sb3_rocketsim import RocketSimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-action RocketSim throughput")
    parser.add_argument("--total-ticks", type=int, default=1_000_000)
    parser.add_argument("--n-sim", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    args = parser.parse_args()

    if min(args.n_sim, args.total_ticks) < 1:
        parser.error("--n-sim and --total-ticks must be positive")

    steps = math.ceil(args.total_ticks / (120 * args.n_sim))
    envs = [RocketSimEnv(frameskip=120, seed=args.seed + i) for i in range(args.n_sim)]

    for i, env in enumerate(envs):
        env.reset(seed=args.seed + i)

    rngs = [np.random.default_rng(args.seed + i) for i in range(args.n_sim)]

    started = time.perf_counter()

    for _ in range(steps):
        for index, env in enumerate(envs):
            _, _, terminated, truncated, _ = env.step(
                rngs[index].integers(env.action_space.n)
            )
            if terminated or truncated:
                env.reset()

    elapsed = time.perf_counter() - started

    for env in envs:
        env.close()

    transitions = steps * args.n_sim
    physics_ticks = transitions * 120
    steps_per_second = transitions / elapsed

    append_result(
        args.log_file,
        {
            "engine": "rocketsim_random",
            "steps": transitions,
            "n_sim": args.n_sim,
            "target_ticks": args.total_ticks,
            "physics_ticks": physics_ticks,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_per_second,
            "seed": args.seed,
        },
    )
    print(
        f"{steps_per_second:.0f} steps/s ({physics_ticks / elapsed:.0f} physics ticks/s)"
    )


if __name__ == "__main__":
    main()
