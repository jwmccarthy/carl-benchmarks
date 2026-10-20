import argparse
import time
from pathlib import Path

import numpy as np

from benchmarks.results import append_result
from benchmarks.rl.sb3_rocketsim import RocketSimEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-action RocketSim throughput")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    args = parser.parse_args()
    env = RocketSimEnv(frameskip=120, seed=args.seed)
    observation, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    for _ in range(args.steps):
        observation, _, terminated, truncated, _ = env.step(
            rng.integers(env.action_space.n)
        )
        if terminated or truncated:
            observation, _ = env.reset()
    elapsed = time.perf_counter() - started
    env.close()
    steps_per_second = args.steps / elapsed
    append_result(
        args.log_file,
        {
            "engine": "rocketsim_random",
            "steps": args.steps,
            "physics_ticks": args.steps * 120,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_per_second,
            "seed": args.seed,
        },
    )
    print(
        f"{steps_per_second:.0f} steps/s ({args.steps * 120 / elapsed:.0f} physics ticks/s)"
    )


if __name__ == "__main__":
    main()
