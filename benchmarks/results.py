import json
from datetime import datetime, timezone
from pathlib import Path


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    with path.open("a") as file:
        json.dump(result, file, sort_keys=True)
        file.write("\n")


def append_episode(
    path: Path,
    engine: str,
    step: int,
    reward: float,
    time_seconds: float,
    run_index: int | None = None,
) -> None:
    episode = {
        "kind": "episode",
        "engine": engine,
        "step": step,
        "reward": reward,
        "episode_reward": reward,
        "time_seconds": time_seconds,
    }
    if run_index is not None:
        episode["run_index"] = run_index
    append_result(
        path,
        episode,
    )
