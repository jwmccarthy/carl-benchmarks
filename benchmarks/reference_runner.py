from __future__ import annotations

import argparse
import json

try:
    from .reference_configs import CARL_N_SIM_SWEEP, REFERENCE_PROFILES, carl_rollout
except ImportError:  # Support execution as `python benchmarks/reference_runner.py`.
    from reference_configs import CARL_N_SIM_SWEEP, REFERENCE_PROFILES, carl_rollout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print native RLGymPPO and derived CARL benchmark configurations."
    )
    parser.add_argument("--profile", choices=REFERENCE_PROFILES, action="append")
    parser.add_argument("--n-sim", type=int, action="append")
    parser.add_argument(
        "--collection-seconds",
        type=float,
        help="Measured time spent collecting one rollout, excluding PPO updates.",
    )
    parser.add_argument(
        "--ppo-update-seconds",
        type=float,
        help="Measured time spent updating PPO for that rollout.",
    )
    args = parser.parse_args()

    if (args.collection_seconds is None) != (args.ppo_update_seconds is None):
        parser.error("--collection-seconds and --ppo-update-seconds must be provided together")
    if args.collection_seconds is not None and (
        args.collection_seconds <= 0 or args.ppo_update_seconds < 0
    ):
        parser.error("collection seconds must be > 0 and PPO update seconds must be >= 0")

    profiles = args.profile or list(REFERENCE_PROFILES)
    n_sim_values = args.n_sim or list(CARL_N_SIM_SWEEP)
    records: list[dict[str, object]] = []
    for profile_name in profiles:
        profile = REFERENCE_PROFILES[profile_name]
        records.append({"kind": "reference_profile", **profile.as_dict()})
        for n_sim in n_sim_values:
            rollout = carl_rollout(profile, n_sim)
            record = {"kind": "carl_configuration", **rollout.as_dict()}
            if args.collection_seconds is not None:
                # Collection rate excludes PPO updates; total rate includes both phases.
                record["collection_ticks_per_second"] = (
                    rollout.collection_ticks / args.collection_seconds
                )
                record["total_ticks_per_second"] = rollout.collection_ticks / (
                    args.collection_seconds + args.ppo_update_seconds
                )
            records.append(record)

    print(json.dumps(records, indent=2, sort_keys=True))
    if args.collection_seconds is None:
        print(
            "Rate definitions: collection_ticks_per_second = collection_ticks / "
            "collection_seconds; total_ticks_per_second = collection_ticks / "
            "(collection_seconds + ppo_update_seconds)."
        )


if __name__ == "__main__":
    main()
