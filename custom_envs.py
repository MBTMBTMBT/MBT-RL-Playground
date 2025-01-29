import math
from typing import Optional
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from numpy import cos, pi, sin
from gymnasium import spaces
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
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

register(
    id="CustomAcrobot-v1",
    entry_point="custom_envs:CustomAcrobotEnv",
    kwargs={
        "render_mode": None,
        "termination_height": 1.0,
        "friction": 0.0,
        "torque_scaling": 1.0,
        "gravity": 9.8,
        "link_lengths": (1.0, 1.0),
        "link_masses": (1.0, 1.0),
        "max_velocities": (4 * pi, 9 * pi),
        "reward_type": "default",
    },
    max_episode_steps=500,
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


class CustomAcrobotEnv(AcrobotEnv):
    """
    Customizable Acrobot environment with user-defined physics parameters.
    """

    def __init__(
            self,
            render_mode: Optional[str] = None,
            termination_height: float = 1.0,  # Height to trigger termination
            friction: float = 0.0,  # Friction factor for joints
            torque_scaling: float = 1.0,  # Scaling factor for torque
            gravity: float = 9.8,  # Gravity value
            link_lengths: tuple = (1.0, 1.0),  # Lengths of the links
            link_masses: tuple = (1.0, 1.0),  # Masses of the links
            max_velocities: tuple = (4 * pi, 9 * pi),  # Max angular velocities
            reward_type: str = "default",
    ):
        super().__init__(render_mode=render_mode)

        # Customizable parameters
        self.termination_height = termination_height
        self.friction = friction
        self.torque_scaling = torque_scaling
        self.gravity = gravity
        self.LINK_LENGTH_1, self.LINK_LENGTH_2 = link_lengths
        self.LINK_MASS_1, self.LINK_MASS_2 = link_masses
        self.MAX_VEL_1, self.MAX_VEL_2 = max_velocities

        # Update observation space with new velocity limits
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Scale available torques
        self.AVAIL_TORQUE = [-1.0 * torque_scaling, 0.0, 1.0 * torque_scaling]
        self.reward_type = reward_type

    def step(self, a):
        obs, reward, terminated, truncated, info = super().step(a)
        if self.reward_type == "sparse":
            reward = 1.0 if terminated else 0.0
        return obs, reward, terminated, truncated, info

    def _terminal(self):
        """Check if the free end of the acrobot has reached the termination height."""
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > self.termination_height)

    def _dsdt(self, s_augmented):
        """Computes the dynamics of the system with the customized parameters."""
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = l1 / 2.0
        lc2 = self.LINK_LENGTH_2 / 2.0
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.gravity
        friction = self.friction
        a = s_augmented[-1]
        s = s_augmented[:-1]

        theta1, theta2, dtheta1, dtheta2 = s
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
                -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
                - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
                + phi2
        )

        # Compute angular accelerations
        if self.book_or_nips == "nips":
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            ddtheta2 = (
                               a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2
                       ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        # Apply friction to angular velocities
        ddtheta1 -= friction * dtheta1
        ddtheta2 -= friction * dtheta2

        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
            self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale]

        termination_y = offset - self.termination_height * scale
        pygame.draw.line(
            surf,
            color=(0, 0, 0),
            start_pos=(-2.2 * scale + offset, termination_y),
            end_pos=(2.2 * scale + offset, termination_y),
        )

        for (x, y), th, llen in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
