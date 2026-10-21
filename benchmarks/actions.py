import numpy as np
import torch

from rlgym.rocket_league.action_parsers import LookupTableAction


def lookup_table() -> np.ndarray:
    table = np.asarray(LookupTableAction.make_lookup_table(), dtype=np.int64)
    if table.shape != (90, 8):
        raise RuntimeError(f"expected RLGym's 90x8 action table, got {table.shape}")
    if not np.array_equal(table[:, 1], table[:, 3]):
        raise RuntimeError("RLGym lookup rows must have equal steer and yaw values")
    return table


def carl_lookup_table(device: torch.device) -> torch.Tensor:
    table = lookup_table()

    def axis(values: np.ndarray) -> np.ndarray:
        if not np.isin(values, (-1, 0, 1)).all():
            raise RuntimeError("lookup axes must use -1, 0, or 1")
        return np.where(values < 0, 1, np.where(values > 0, 2, 0))

    # RLGym: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
    # CARL:  [horizontal, vertical, throttle, powerslide, boost, air_roll, jump].
    return torch.as_tensor(
        np.stack(
            (
                axis(table[:, 1]),
                axis(table[:, 2]),
                axis(table[:, 0]),
                table[:, 7],
                table[:, 6],
                axis(table[:, 4]),
                table[:, 5],
            ),
            axis=-1,
        ),
        dtype=torch.int32,
        device=device,
    )
