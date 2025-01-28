import math
from typing import Optional

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.registration import register


# Register the custom environments
register(
    id="CustomFrozenLake-v1",
    entry_point="custom_envs:CustomFrozenLakeEnv",
    kwargs={
        "render_mode": None,
        "desc": None,
        "map_name": "4x4",
        "is_slippery": True,
        "slipperiness": 0.8,
    },
    max_episode_steps=100,
)

register(
    id="CustomMountainCar-v0",
    entry_point="custom_envs:CustomMountainCarEnv",
    kwargs={
        "render_mode": None,
        "goal_velocity": 0,
        "custom_gravity": 0.0025,
        "custom_force": 0.001,
        "goal_position": 0.5,
        "reward_type": "default",
    },
    max_episode_steps=200,
)


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        slipperiness: float = 1.0,
    ):
        """
        Custom FrozenLake environment with adjustable slipperiness.

        Parameters:
        - render_mode: Optional rendering mode.
        - desc: Custom map description (2D list of strings).
        - map_name: Predefined map name.
        - is_slippery: Determines if the environment is slippery (original logic).
        - slipperiness: Degree of slipperiness (0 = no slip, 1 = original slippery).
        """
        super().__init__(render_mode=render_mode, desc=desc, map_name=map_name)
        self.is_slippery = is_slippery
        self.slipperiness = slipperiness

        # Override transition probabilities
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    li.clear()

                    if self.desc[row][col] in b"GH":  # Goal or Hole
                        li.append((1.0, s, 0, True))
                    else:
                        if self.is_slippery:
                            # Custom slippery logic controlled by slipperiness
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                if b == a:  # Forward direction
                                    prob = 1 - (self.slipperiness * (2 / 3))
                                else:  # Side directions
                                    prob = self.slipperiness / 3
                                new_row, new_col = self._inc(row, col, b)
                                new_state = self._to_s(new_row, new_col)
                                new_letter = self.desc[new_row][new_col]
                                terminated = bytes(new_letter) in b"GH"
                                reward = float(new_letter == b"G")
                                li.append((prob, new_state, reward, terminated))
                        else:
                            # Deterministic logic when not slippery
                            new_row, new_col = self._inc(row, col, a)
                            new_state = self._to_s(new_row, new_col)
                            new_letter = self.desc[new_row][new_col]
                            terminated = bytes(new_letter) in b"GH"
                            reward = float(new_letter == b"G")
                            li.append((1.0, new_state, reward, terminated))

    def _inc(self, row, col, action):
        """
        Increment row and column based on the action taken.

        Parameters:
        - row: Current row index.
        - col: Current column index.
        - action: Action to take (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP).

        Returns:
        - Tuple (new_row, new_col) after taking the action.
        """
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.nrow - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.ncol - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)
        return row, col

    def _to_s(self, row, col):
        """
        Convert (row, col) to a single state index.

        Parameters:
        - row: Row index.
        - col: Column index.

        Returns:
        - State index as an integer.
        """
        return row * self.ncol + col


class CustomMountainCarEnv(MountainCarEnv):
    """
    Custom version of the MountainCar environment with adjustable gravity, reward, and episode duration.
    """

    def __init__(
            self,
            render_mode: Optional[str] = None,
            goal_velocity=0,
            custom_gravity=0.0025,
            custom_force=0.001,
            # max_episode_steps=200,
            goal_position=0.5,
            reward_type='default',
    ):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)
        # Override gravity and max_episode_steps with custom values
        self.goal_position = goal_position
        if custom_gravity >= self.gravity:
            self.max_speed = custom_gravity / self.gravity * self.max_speed
        self.force = custom_force
        self.gravity = custom_gravity
        # self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.reward_type = reward_type

    def step(self, action: int):
        # Override the step function to include a step counter for custom episode duration
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        # Custom reward function based on reward_type
        if self.reward_type == 'default':
            reward = -1.0 if not terminated else 0.0
        elif self.reward_type == 'distance':
            # Reward based on the distance from the starting position (-0.5)
            reward = abs(position + 0.5) - (
                1.0 if position <= self.min_position or position >= self.max_position else 0.0)
            reward += -1.0 if not terminated else 0.0
        elif self.reward_type == 'progress':
            # Reward based on progress towards the goal, incentivizing movement to the right and higher speed
            reward = (position - self.min_position) if position >= 0.0 else 0.0
            reward += velocity if velocity >= self.goal_velocity else 0.0
            reward += -1.0 if not terminated else 0.0
        elif self.reward_type == 'sparse':
            reward = -0.0 if not terminated else 1.0
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self.state = (position, velocity)
        self.current_step += 1

        # Terminate if the maximum number of steps is reached
        truncated = False
        # truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Override the reset function to reset the step counter
        self.current_step = 0
        return super().reset(seed=seed, options=options)

