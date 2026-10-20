"""Small fixed Nexto-style reward for RLGym.

The full Goddard reward is tied to CARL's batched state API.  This adapter
keeps the compatible RLGym signals and deliberately has no schedule.
"""

import numpy as np
from rlgym.api import RewardFunction


class FixedNextoReward(RewardFunction):
    """Fixed shaping_scale=1 and goal_scale=10 for one-versus-one RLGym."""

    def reset(self, agents, initial_state, shared_info):
        self.previous = None

    def get_rewards(self, agents, state, is_terminals, is_truncated, shared_info):
        rewards = {agent: 0.0 for agent in agents}
        ball = np.asarray(state.ball.position)
        velocity = np.asarray(state.ball.linear_velocity)
        for agent in agents:
            car = state.cars[agent].physics
            car_position = np.asarray(car.position)
            distance = np.linalg.norm(ball - car_position)
            toward_ball = np.dot(np.asarray(car.linear_velocity), ball - car_position)
            rewards[agent] = (
                0.0025 * np.exp(-distance / 2300.0)
                + 0.00125 * np.tanh(toward_ball / 2300.0)
                + 0.00025 * min(np.linalg.norm(velocity) / 6000.0, 1.0)
            )
        if self.previous is not None:
            for agent in agents:
                rewards[agent] += (
                    5.0 * float(np.linalg.norm(velocity) - self.previous) / 6000.0
                )
        self.previous = np.linalg.norm(velocity)
        return rewards
