import numpy as np
import math
from typing import Optional
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv


class CustomMountainCarEnv(MountainCarEnv):
    """
    Custom version of the MountainCar environment with adjustable gravity, reward, and episode duration.
    """

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0, custom_gravity=0.0025, max_episode_steps=200,
                 reward_type='default'):
        super().__init__(render_mode=render_mode, goal_velocity=goal_velocity)
        # Override gravity and max_episode_steps with custom values
        self.gravity = custom_gravity
        self.max_episode_steps = max_episode_steps
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
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self.state = (position, velocity)
        self.current_step += 1

        # Terminate if the maximum number of steps is reached
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Override the reset function to reset the step counter
        self.current_step = 0
        return super().reset(seed=seed, options=options)


if __name__ == "__main__":
    # Example of how to create and use the custom environment
    env = CustomMountainCarEnv(custom_gravity=0.003, max_episode_steps=300, reward_type='progress')
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action for demonstration purposes
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
