from dataclasses import dataclass
from typing import Final


CARL_N_SIM_SWEEP: Final = (512, 1024, 2048, 4096, 8192, 16384)


@dataclass(frozen=True)
class UnifiedPPO:
    policy_layers:          tuple[int, ...] = (256, 256, 256)
    critic_layers:          tuple[int, ...] = (256, 256, 256)
    collection_transitions: int = 100_000
    minibatch_size:         int             = 25_000
    epochs:                 int             = 1
    entropy_coefficient:    float           = 0.01
    learning_rate:          float           = 2e-4
    frameskip:              int             = 8


PPO:                      Final = UnifiedPPO()
CPP_THREADS:              Final = 16
CPP_GAMES_PER_THREAD:     Final = 24
CPP_DECISION_ACTORS:      Final = CPP_THREADS * CPP_GAMES_PER_THREAD * 2
GOAL_REWARD:              Final = 10.0
TOUCH_REWARD:             Final = 0.1
NO_TOUCH_TIMEOUT_SECONDS: Final = 10.0
