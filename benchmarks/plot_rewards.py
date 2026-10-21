import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def exponential_smooth(values, weight):
    smoothed = []
    current = None
    for value in values:
        current = value if current is None else weight * current + (1 - weight) * value
        smoothed.append(current)
    return smoothed


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
    parser.add_argument("--smoothing", type=float, default=0.9)
    args = parser.parse_args()
    paths = args.paths + (args.inputs or [])
    if not paths:
        parser.error("provide at least one input JSONL file")
    if not 0 <= args.smoothing < 1:
        parser.error("--smoothing must be in [0, 1)")

    records = read_records(paths)
    groups = {}
    for record in records:
        engine = record.get("engine", record["_source"])
        config = record.get("config") or record.get("n_sim")
        label = f"{engine} (n={config})" if config is not None else engine
        groups.setdefault(label, []).append(record)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in groups.items():
        group.sort(key=lambda record: record.get("time_seconds", 0))
        times = [record.get("time_seconds", 0) for record in group]
        rewards = [
            record["reward"] if "reward" in record else record["episode_reward"]
            for record in group
        ]
        (line,) = ax.plot(times, rewards, alpha=0.12, linewidth=1)
        smoothed = exponential_smooth(rewards, args.smoothing)
        ax.plot(times, smoothed, color=line.get_color(), linewidth=2.5, label=label)
    ax.set_xlabel("Wall time (seconds)")
    ax.set_ylabel("Episode reward")
    ax.legend(frameon=True, fancybox=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
