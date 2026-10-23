import argparse
import sys
from pathlib import Path

import numpy as np

from reference_configs import RLGYMPPO_PYTHON
from results import append_result


def build_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    from rlgym_ppo.util import RLGymV2GymWrapper

    return RLGymV2GymWrapper(
        RLGym(
            state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()),
            obs_builder=DefaultObs(
                zero_padding=None,
                pos_coef=np.asarray([
                    1 / common_values.SIDE_WALL_X,
                    1 / common_values.BACK_NET_Y,
                    1 / common_values.CEILING_Z,
                ]),
                ang_coef=1 / np.pi,
                lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            ),
            action_parser=RepeatAction(LookupTableAction(), repeats=RLGYMPPO_PYTHON.frameskip),
            reward_fn=CombinedReward((GoalReward(), 10), (TouchReward(), 0.1)),
            termination_cond=GoalCondition(),
            truncation_cond=NoTouchTimeoutCondition(timeout_seconds=10),
            transition_engine=RocketSimEngine(),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native Python RLGymPPO profile.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("results/benchmarks.jsonl"))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    from rlgym_ppo import Learner
    from rlgym_ppo.util import reporting

    original_report_metrics = reporting.report_metrics

    def report_metrics(loggable_metrics, debug_metrics, wandb_run):
        report = dict(loggable_metrics)
        transitions = int(report["Timesteps Collected"])
        collection_seconds = float(report["Timestep Collection Time"])
        total_seconds = float(report["Total Iteration Time"])
        ticks = transitions * RLGYMPPO_PYTHON.frameskip
        append_result(
            args.log_file,
            {
                "kind": "native_iteration",
                "engine": "rlgymppo_python",
                "profile": RLGYMPPO_PYTHON.name,
                "frameskip": RLGYMPPO_PYTHON.frameskip,
                "transitions": transitions,
                "collection_ticks": ticks,
                "collection_seconds": collection_seconds,
                "ppo_update_seconds": total_seconds - collection_seconds,
                "collection_ticks_per_second": ticks / collection_seconds,
                "total_ticks_per_second": ticks / total_seconds,
            },
        )
        original_report_metrics(loggable_metrics, debug_metrics, wandb_run)

    reporting.report_metrics = report_metrics
    Learner(
        build_env,
        n_proc=RLGYMPPO_PYTHON.processes,
        min_inference_size=29,
        random_seed=args.seed,
        timestep_limit=args.iterations * RLGYMPPO_PYTHON.collection_transitions,
        exp_buffer_size=RLGYMPPO_PYTHON.collection_transitions * 3,
        ts_per_iteration=RLGYMPPO_PYTHON.collection_transitions,
        policy_layer_sizes=RLGYMPPO_PYTHON.policy_layers,
        critic_layer_sizes=RLGYMPPO_PYTHON.critic_layers,
        ppo_epochs=RLGYMPPO_PYTHON.epochs,
        ppo_batch_size=RLGYMPPO_PYTHON.collection_transitions,
        ppo_minibatch_size=RLGYMPPO_PYTHON.minibatch_size,
        ppo_ent_coef=RLGYMPPO_PYTHON.entropy_coefficient,
        policy_lr=RLGYMPPO_PYTHON.learning_rate,
        critic_lr=RLGYMPPO_PYTHON.learning_rate,
        standardize_returns=True,
        standardize_obs=False,
        checkpoint_load_folder=None,
        checkpoints_save_folder=None,
        log_to_wandb=False,
    ).learn()


if __name__ == "__main__":
    sys.exit(main())
