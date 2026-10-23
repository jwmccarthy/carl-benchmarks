import argparse
import sys
from pathlib import Path

import numpy as np

from results import append_result
from unified_configs import PPO, PYTHON_PROCESSES


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
            action_parser=RepeatAction(LookupTableAction(), repeats=PPO.frameskip),
            reward_fn=CombinedReward((GoalReward(), 10), (TouchReward(), 0.1)),
            termination_cond=GoalCondition(),
            truncation_cond=NoTouchTimeoutCondition(timeout_seconds=10),
            transition_engine=RocketSimEngine(),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native Python RLGymPPO profile.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--n-proc", type=int, default=PYTHON_PROCESSES)
    parser.add_argument("--collection-transitions", type=int, default=PPO.collection_transitions)
    parser.add_argument("--minibatch-size", type=int, default=PPO.minibatch_size)
    parser.add_argument("--epochs", type=int, default=PPO.epochs)
    parser.add_argument("--learning-rate", type=float, default=PPO.learning_rate)
    parser.add_argument("--entropy-coefficient", type=float, default=PPO.entropy_coefficient)
    parser.add_argument("--layer-size", type=int, default=256)
    parser.add_argument("--layer-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("results/benchmarks.jsonl"))
    args = parser.parse_args()
    if (args.iterations < 1 or args.warmup_iterations < 0 or args.epochs < 1
            or args.n_proc < 1 or args.collection_transitions < 1
            or args.minibatch_size < 1 or args.layer_size < 1 or args.layer_count < 1):
        parser.error("counts must be positive; warmup iterations cannot be negative")
    layers = (args.layer_size,) * args.layer_count

    from rlgym_ppo import Learner
    import rlgym_ppo.learner as learner_module
    from rlgym_ppo.util import reporting

    original_report_metrics = reporting.report_metrics
    reported_iterations = 0

    def report_metrics(loggable_metrics, debug_metrics, wandb_run):
        nonlocal reported_iterations
        report = dict(loggable_metrics)
        if reported_iterations >= args.warmup_iterations:
            transitions = int(report["Timesteps Collected"])
            collection_seconds = float(report["Timestep Collection Time"])
            total_seconds = float(report["Total Iteration Time"])
            ticks = transitions * PPO.frameskip
            append_result(
                args.log_file,
                {
                    "kind": "native_iteration",
                    "engine": "rlgymppo_python",
                    "profile": "unified",
                    "n_proc": args.n_proc,
                    "decision_environments": args.n_proc * 2,
                    "epochs": args.epochs,
                    "warmup_iterations": args.warmup_iterations,
                    "frameskip": PPO.frameskip,
                    "transitions": transitions,
                    "collection_ticks": ticks,
                    "collection_seconds": collection_seconds,
                    "ppo_update_seconds": total_seconds - collection_seconds,
                    "collection_ticks_per_second": ticks / collection_seconds,
                    "total_ticks_per_second": ticks / total_seconds,
                },
            )
        reported_iterations += 1
        original_report_metrics(loggable_metrics, debug_metrics, wandb_run)

    reporting.report_metrics = report_metrics

    class NoKeyboardInput:
        def kbhit(self):
            return False

    learner_module.KBHit = NoKeyboardInput
    Learner(
        build_env,
        n_proc=args.n_proc,
        min_inference_size=29,
        random_seed=args.seed,
        timestep_limit=(args.iterations + args.warmup_iterations)
        * args.collection_transitions,
        exp_buffer_size=args.collection_transitions * 3,
        ts_per_iteration=args.collection_transitions,
        policy_layer_sizes=layers,
        critic_layer_sizes=layers,
        ppo_epochs=args.epochs,
        ppo_batch_size=args.collection_transitions,
        ppo_minibatch_size=args.minibatch_size,
        ppo_ent_coef=args.entropy_coefficient,
        policy_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        standardize_returns=True,
        standardize_obs=False,
        checkpoint_load_folder=None,
        checkpoints_save_folder=None,
        log_to_wandb=False,
    ).learn()


if __name__ == "__main__":
    sys.exit(main())
