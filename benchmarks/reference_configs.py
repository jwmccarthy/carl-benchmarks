from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from typing import Final


from unified_configs import CARL_N_SIM_SWEEP, CPP_GAMES_PER_THREAD, CPP_THREADS, PPO, PYTHON_PROCESSES
CARL_CARS_PER_1V1: Final = 2


@dataclass(frozen=True)
class ReferenceProfile:
    name: str
    processes: int | None
    threads: int | None
    games_per_thread: int | None
    collection_transitions: int
    minibatch_size: int
    epochs: int
    policy_layers: tuple[int, ...]
    critic_layers: tuple[int, ...]
    entropy_coefficient: float
    learning_rate: float
    frameskip: int

    @property
    def environment_count(self) -> int:
        if self.processes is not None:
            return self.processes
        assert self.threads is not None and self.games_per_thread is not None
        return self.threads * self.games_per_thread

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"environment_count": self.environment_count}


@dataclass(frozen=True)
class CARLRollout:
    profile: ReferenceProfile
    n_sim: int
    n_cars: int
    decision_environments: int
    rollout_steps: int
    actual_collection_transitions: int

    @property
    def collection_ticks(self) -> int:
        return self.actual_collection_transitions * self.profile.frameskip

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.name,
            "n_sim": self.n_sim,
            "n_cars": self.n_cars,
            "decision_environments": self.decision_environments,
            "target_collection_transitions": self.profile.collection_transitions,
            "rollout_steps": self.rollout_steps,
            "actual_collection_transitions": self.actual_collection_transitions,
            "collection_ticks": self.collection_ticks,
            "frameskip": self.profile.frameskip,
        }


RLGYMPPO_PYTHON: Final = ReferenceProfile(
    name="rlgymppo_python",
    processes=PYTHON_PROCESSES,
    threads=None,
    games_per_thread=None,
    collection_transitions=PPO.collection_transitions,
    minibatch_size=PPO.minibatch_size,
    epochs=PPO.epochs,
    policy_layers=PPO.policy_layers,
    critic_layers=PPO.critic_layers,
    entropy_coefficient=PPO.entropy_coefficient,
    learning_rate=PPO.learning_rate,
    frameskip=PPO.frameskip,
)

RLGYMPPO_CPP: Final = ReferenceProfile(
    name="rlgymppo_cpp",
    processes=None,
    threads=CPP_THREADS,
    games_per_thread=CPP_GAMES_PER_THREAD,
    collection_transitions=PPO.collection_transitions,
    minibatch_size=PPO.minibatch_size,
    epochs=PPO.epochs,
    policy_layers=PPO.policy_layers,
    critic_layers=PPO.critic_layers,
    entropy_coefficient=PPO.entropy_coefficient,
    learning_rate=PPO.learning_rate,
    frameskip=PPO.frameskip,
)

REFERENCE_PROFILES: Final = {
    RLGYMPPO_PYTHON.name: RLGYMPPO_PYTHON,
    RLGYMPPO_CPP.name: RLGYMPPO_CPP,
}


def carl_rollout(profile: ReferenceProfile, n_sim: int) -> CARLRollout:
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")

    decision_environments = n_sim * CARL_CARS_PER_1V1
    rollout_steps = max(
        1,
        floor(profile.collection_transitions / decision_environments + 0.5),
    )
    return CARLRollout(
        profile=profile,
        n_sim=n_sim,
        n_cars=CARL_CARS_PER_1V1,
        decision_environments=decision_environments,
        rollout_steps=rollout_steps,
        actual_collection_transitions=rollout_steps * decision_environments,
    )
