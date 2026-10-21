import copy
import math

import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.common_values import BLUE_TEAM

BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
CEILING_Z = 2044.0
GOAL_Y = 5124.25
GOAL_HEIGHT = 642.775
GOAL_DISTANCE_OFFSET = GOAL_Y - 5120.0 + BALL_RADIUS
TOUCH_HEIGHT_SCALE = 2250.0
MATCH_TICKS = 5 * 60 * 120
HISTORICAL_GOAL_WEIGHT = 10.0


class NextoReward(RewardFunction):
    def __init__(self):
        self.previous = None
        self.score_difference = 0
        self.touch_decay = {}
        self.last_touch = {}

    def reset(self, agents, initial_state, shared_info):
        self.previous = copy.deepcopy(initial_state)
        self.score_difference = 0
        self.touch_decay = {agent: 1.0 for agent in agents}
        self.last_touch = {agent: False for agent in agents}

    @staticmethod
    def _unit(value):
        return value / max(float(np.linalg.norm(value)), 1e-8)

    @classmethod
    def _cosine(cls, left, right):
        return float(np.dot(cls._unit(left), cls._unit(right)))

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        if self.previous is None:
            raise RuntimeError("reward used before reset")
        if (
            state.tick_count >= MATCH_TICKS
            and self.score_difference == 0
            and not any(is_truncated.values())
        ):
            raise RuntimeError(
                "Nexto win-probability reward requires an overtime flag, which "
                "RLGym GameState does not expose"
            )
        previous = self.previous
        score_delta = 0
        if state.goal_scored:
            score_delta = 1 if state.scoring_team == BLUE_TEAM else -1
            self.score_difference += score_delta
        cars, old_cars = state.cars, previous.cars
        positions = {agent: cars[agent].physics.position for agent in agents}
        ball, old_ball = state.ball, previous.ball
        touched = {agent: cars[agent].ball_touches > 0 for agent in agents}
        if any(touched.values()):
            self.last_touch = touched.copy()

        rewards = {}
        for agent in agents:
            car, old_car = cars[agent], old_cars[agent]
            sign = 1.0 if car.team_num == BLUE_TEAM else -1.0
            opponent_goal = np.array((0.0, sign * GOAL_Y, 0.0))
            own_goal = -opponent_goal
            to_ball = ball.position - car.physics.position
            old_to_ball = old_ball.position - old_car.physics.position
            distance = np.linalg.norm(to_ball)
            old_distance = np.linalg.norm(old_to_ball)
            ball_to_goal = opponent_goal - ball.position
            old_ball_to_goal = opponent_goal - old_ball.position
            opponents = [a for a in agents if cars[a].team_num != car.team_num]
            defender_distance = np.mean(
                [np.linalg.norm(positions[a] - old_ball.position) for a in opponents]
            )
            scored = max(score_delta * sign, 0.0)
            goal_speed = (
                scored * np.linalg.norm(old_ball.linear_velocity) / BALL_MAX_SPEED
            )
            goal_distance = scored * (
                1.0 - math.exp(-defender_distance / CAR_MAX_SPEED)
            )
            boost = math.sqrt(np.clip(car.boost_amount / 100, 0, 1))
            old_boost = math.sqrt(np.clip(old_car.boost_amount / 100, 0, 1))
            boost_delta = boost - old_boost
            ball_goal_progress = math.exp(
                -np.linalg.norm(ball_to_goal) / BALL_MAX_SPEED
            ) - math.exp(-np.linalg.norm(old_ball_to_goal) / BALL_MAX_SPEED)
            player_ball_progress = math.exp(-distance / 1410) - math.exp(
                -old_distance / 1410
            )
            if touched[agent]:
                self.touch_decay[agent] = max(self.touch_decay[agent] * 0.95, 0.1)
            else:
                self.touch_decay[agent] = min(self.touch_decay[agent] + 0.013, 1.0)
            touch_height = (
                max((ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS), 0) ** 0.2836
            )
            ball_touch = (
                touched[agent]
                * self.touch_decay[agent]
                * touch_height
                * max(ball_goal_progress, 0)
            )
            newly_demoed = float(car.is_demoed and not old_car.is_demoed)
            enemy_demo = np.mean(
                [
                    float(cars[a].is_demoed and not old_cars[a].is_demoed)
                    for a in opponents
                ]
            )
            direction = self._unit(to_ball)
            alignment = 0.5 * (
                self._cosine(to_ball, car.physics.position - own_goal)
                + self._cosine(-to_ball, opponent_goal - car.physics.position)
            )
            old_alignment = 0.5 * (
                self._cosine(old_to_ball, old_car.physics.position - own_goal)
                + self._cosine(-old_to_ball, opponent_goal - old_car.physics.position)
            )
            closest = float(
                distance
                == min(np.linalg.norm(ball.position - positions[a]) for a in agents)
            )
            touched_last = float(self.last_touch[agent])
            ball_speed = np.linalg.norm(ball.linear_velocity)
            flip_reset = float(
                touched[agent]
                and (old_car.has_flipped or old_car.has_double_jumped)
                and not (car.has_flipped or car.has_double_jumped)
                and car.physics.position[2] > 3 * BALL_RADIUS
                and distance < 2 * BALL_RADIUS
                and self._cosine(to_ball, -car.physics.up) > 0.9
            )
            remaining = max(MATCH_TICKS - state.tick_count, 0) / 120
            variance = max(2 * remaining / 60, 1e-6)

            def win_prob(score):
                return 0.5 * (
                    1 + math.erf((score - 0.5) / math.sqrt(variance) / math.sqrt(2))
                )

            win_progress = sign * (
                win_prob(self.score_difference)
                - win_prob(self.score_difference - score_delta)
            )
            velocity = car.physics.linear_velocity
            shaping = (
                2.5 * goal_speed
                + 2.5 * goal_distance
                + max(boost_delta, 0)
                - 0.5
                * max(-boost_delta, 0)
                * np.clip(1 - car.physics.position[2] / GOAL_HEIGHT, 0, 1)
                + 0 * ball_touch
                + 0.00025
                * np.clip(
                    (ball.position[2] - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS), 0, 1
                )
                * touched_last
                + 0.00025 * min(ball_speed / BALL_MAX_SPEED, 1) * touched_last
                + 5 * (enemy_demo - newly_demoed)
                + 0.0025
                * math.exp(-0.5 * max(distance - BALL_RADIUS, 0) / CAR_MAX_SPEED)
                + 0.0025
                * math.exp(
                    -0.5
                    * max(np.linalg.norm(ball_to_goal) - GOAL_DISTANCE_OFFSET, 0)
                    / BALL_MAX_SPEED
                )
                + 0.000625 * np.dot(car.physics.forward, direction)
                + 0.0025 * alignment
                + 0.00125 * closest
                + 0.00025 * touched_last
                + 0.00125
                * float(sign * (ball.position[1] - car.physics.position[1]) > 0)
                + 0.00125 * np.dot(self._unit(velocity), direction)
                + 0.1
                * np.dot(self._unit(velocity), direction)
                * float(np.linalg.norm(ball.position[:2]) < 1)
                + 0.000625 * min(np.linalg.norm(velocity) / CAR_MAX_SPEED, 1)
                + 0.00125 * boost
                + 0.0015 * np.dot(car.physics.forward, velocity) / CAR_MAX_SPEED
                + 5 * ball_goal_progress
                + 0.75 * player_ball_progress
                + 0.5 * (alignment - old_alignment)
                + 0.25
                * touched[agent]
                * np.linalg.norm(ball.linear_velocity - old_ball.linear_velocity)
                / CAR_MAX_SPEED
                + touched[agent] * max(ball.position[2] / TOUCH_HEIGHT_SCALE, 0)
                + 0.01 * np.linalg.norm(car.physics.angular_velocity) / 5.5
                + 10 * flip_reset
                - 0.005 * float(car.on_ground and car.physics.position[2] < BALL_RADIUS)
                + 10 * win_progress
            )
            rewards[agent] = score_delta * sign + shaping / HISTORICAL_GOAL_WEIGHT
        self.previous = copy.deepcopy(state)
        return rewards
