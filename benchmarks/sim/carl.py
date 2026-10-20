import argparse
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
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file", type=Path, default=Path("results/benchmarks.jsonl")
    )
    args = parser.parse_args()
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
    th.cuda.synchronize(env.device)
    started = time.perf_counter()

    for _ in range(args.steps):
        env.step(th.randint(90, (args.n_sim,), device=env.device))

    th.cuda.synchronize(env.device)
    elapsed = time.perf_counter() - started

    env.close()

    transitions = args.steps * args.n_sim
    steps_per_second = transitions / elapsed

    append_result(
        args.log_file,
        {
            "engine": "carl_random",
            "n_sim": args.n_sim,
            "steps": args.steps,
            "physics_ticks": args.steps * 120,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_per_second,
            "seed": args.seed,
        },
    )
    print(
        f"{steps_per_second:.0f} env steps/s ({args.steps * 120 * args.n_sim / elapsed:.0f} ticks/s)"
    )


if __name__ == "__main__":
    main()
