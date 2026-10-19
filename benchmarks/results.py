import json
from datetime import datetime, timezone
from pathlib import Path


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    with path.open("a") as file:
        json.dump(result, file, sort_keys=True)
        file.write("\n")
