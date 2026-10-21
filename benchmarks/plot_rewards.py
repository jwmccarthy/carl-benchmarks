import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def read_records(paths):
    records = []
    for path in paths:
        with path.open() as file:
            for line in file:
                record = json.loads(line)
                record["_source"] = path.stem
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RL episode rewards over wall time"
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--input", action="append", type=Path, dest="inputs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()
    paths = args.paths + (args.inputs or [])
    if not paths:
        parser.error("provide at least one input JSONL file")
    if args.window < 1:
        parser.error("--window must be positive")

    records = read_records(paths)
    groups = {}
    for record in records:
        engine = record.get("engine", record["_source"])
        config = record.get("config") or record.get("n_sim")
        label = f"{engine} (n={config})" if config is not None else engine
        groups.setdefault(label, []).append(record)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in groups.items():
        group.sort(key=lambda record: record.get("time_seconds", 0))
        times = [record.get("time_seconds", 0) for record in group]
        rewards = [
            record["reward"] if "reward" in record else record["episode_reward"]
            for record in group
        ]
        ax.scatter(times, rewards, alpha=0.2, s=14)
        smoothed = [
            sum(rewards[max(0, index - args.window + 1) : index + 1])
            / min(args.window, index + 1)
            for index in range(len(rewards))
        ]
        ax.plot(times, smoothed, label=label)
    ax.set_xlabel("Wall time (seconds)")
    ax.set_ylabel("Episode reward")
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
