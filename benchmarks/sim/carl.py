import argparse
import math
import sys
import time
from pathlib import Path

import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.results import append_result
from benchmarks.rl.jarl_carl import LookupCARLEnv
from carl.gymnasium import CARLTorchVectorEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-action CARL throughput")
    parser.add_argument("--n-sim", type=int, default=1024)
    parser.add_argument("--total-ticks", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    args = parser.parse_args()
    if min(args.n_sim, args.total_ticks, args.warmup_steps) < 1:
        parser.error("n-sim, total-ticks, and warmup-steps must be positive")

    base = CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=120,
        discrete_actions=True,
    )
    env = LookupCARLEnv(base, args.seed)
    env.reset()

    for _ in range(args.warmup_steps):
        env.step(th.randint(90, (args.n_sim,), device=env.device))

    th.cuda.synchronize(env.device)
    started = time.perf_counter()

    steps = math.ceil(args.total_ticks / (120 * args.n_sim))
    for _ in range(steps):
        env.step(th.randint(90, (args.n_sim,), device=env.device))

    th.cuda.synchronize(env.device)
    elapsed = time.perf_counter() - started

    env.close()

    transitions = steps * args.n_sim
    physics_ticks = transitions * 120
    steps_per_second = transitions / elapsed

    append_result(
        args.log_file,
        {
            "engine": "carl_random",
            "n_sim": args.n_sim,
            "warmup_steps": args.warmup_steps,
            "steps": transitions,
            "target_ticks": args.total_ticks,
            "physics_ticks": physics_ticks,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_per_second,
            "seed": args.seed,
        },
    )
    print(f"{steps_per_second:.0f} env steps/s ({physics_ticks / elapsed:.0f} ticks/s)")


if __name__ == "__main__":
    main()
