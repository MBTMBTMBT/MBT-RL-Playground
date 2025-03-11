import hashlib
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
from gymnasium.envs.box2d.car_racing import CarRacing
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
    max_episode_steps=250,
)


register(
    id="CarRacingFixedMap-v2",
    entry_point="custom_envs:CarRacingFixedMap",
    kwargs={
        "render_mode": None,
        "lap_complete_percent": 0.95,
        "domain_randomize": False,
        "continuous": False,
        "map_seed": 0,
    },
    max_episode_steps=1000,
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
        reward_type="default",
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
        if self.reward_type == "default":
            reward = -1.0 if not terminated else 0.0
        elif self.reward_type == "distance":
            # Reward based on the distance from the starting position (-0.5)
            reward = abs(position + 0.5) - (
                1.0
                if position <= self.min_position or position >= self.max_position
                else 0.0
            )
            reward += -1.0 if not terminated else 0.0
        elif self.reward_type == "progress":
            # Reward based on progress towards the goal, incentivizing movement to the right and higher speed
            reward = (position - self.min_position) if position >= 0.0 else 0.0
            reward += velocity if velocity >= self.goal_velocity else 0.0
            reward += -1.0 if not terminated else 0.0
        elif self.reward_type == "sparse":
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
        high = np.array(
            [1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Scale available torques
        self.AVAIL_TORQUE = [-1.0 * torque_scaling, 0.0, 1.0 * torque_scaling]
        self.reward_type = reward_type

    def step(self, a):
        obs, reward, terminated, truncated, info = super().step(a)
        if self.reward_type == "sparse":
            reward = 1.0 if terminated else 0.0
        elif self.reward_type == "scaled":
            reward = -1.0 / 250 if not terminated else 0.0
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
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )

        # Compute angular accelerations
        if self.book_or_nips == "nips":
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
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

        termination_y = offset + self.termination_height * scale
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


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)
NO_FREEZE = 5000


class CarRacingFixedMap(CarRacing):
    def __init__(
        self,
        render_mode=None,
        verbose=False,
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
        map_seed=0,
    ):
        self.map_seed = map_seed
        super().__init__(
            render_mode=render_mode,
            verbose=verbose,
            lap_complete_percent=lap_complete_percent,
            domain_randomize=domain_randomize,
            continuous=continuous,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        reward /= 1e3

        on_grass = self._check_on_grass()
        if on_grass:
            terminated = True
            reward -= 1

        return obs, reward, terminated, truncated, info

    def _check_on_grass(self):
        """
        Return True if ALL wheels are outside road boundaries (on grass).
        """
        assert self.car is not None
        wheels = [wheel.position for wheel in self.car.wheels]

        if not self.road or len(self.road) == 0:
            print("[DEBUG] No road tiles available!")
            return True  # No tiles, assume on grass

        def point_on_road(pos):
            for tile in self.road:
                if tile.fixtures[0].TestPoint(pos):
                    return True
            return False

        # If any wheel is on the road, we are NOT on grass
        for wheel_pos in wheels:
            if point_on_road(wheel_pos):
                return False

        return True  # All wheels off-road

    import hashlib

    def _create_track(self):
        def derive_new_seed(base_seed, retry_count):
            # Combine the original map_seed and retry_count in a deterministic way
            data = f"{base_seed}-{retry_count}".encode("utf-8")
            hashed = hashlib.sha256(data).hexdigest()
            # Convert hex digest to int32 range
            new_seed = int(hashed, 16) % (2 ** 31)
            return new_seed

        max_retries = np.inf  # To avoid infinite loops
        retries = 0

        while retries < max_retries:
            # Deterministic seed transformation
            current_seed = derive_new_seed(self.map_seed, retries)
            map_random = np.random.RandomState(current_seed)

            # === START OF ORIGINAL GENERATION ===
            CHECKPOINTS = 12
            checkpoints = []
            for c in range(CHECKPOINTS):
                noise = map_random.uniform(0, 2 * np.pi * 1 / CHECKPOINTS)
                alpha = 2 * np.pi * c / CHECKPOINTS + noise
                rad = map_random.uniform(TRACK_RAD / 3, TRACK_RAD)

                if c == 0:
                    alpha = 0
                    rad = 1.5 * TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * np.pi * c / CHECKPOINTS
                    self.start_alpha = 2 * np.pi * (-0.5) / CHECKPOINTS
                    rad = 1.5 * TRACK_RAD

                checkpoints.append((alpha, rad * np.cos(alpha), rad * np.sin(alpha)))

            self.road = []
            x, y, beta = 1.5 * TRACK_RAD, 0, 0
            dest_i = 0
            laps = 0
            track = []
            no_freeze = NO_FREEZE
            visited_other_side = False
            while True:
                alpha = np.arctan2(y, x)
                if visited_other_side and alpha > 0:
                    laps += 1
                    visited_other_side = False
                if alpha < 0:
                    visited_other_side = True
                    alpha += 2 * np.pi

                while True:
                    failed = True
                    while True:
                        dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                        if alpha <= dest_alpha:
                            failed = False
                            break
                        dest_i += 1
                        if dest_i % len(checkpoints) == 0:
                            break

                    if not failed:
                        break

                    alpha -= 2 * np.pi
                    continue

                r1x = np.cos(beta)
                r1y = np.sin(beta)
                p1x = -r1y
                p1y = r1x
                dest_dx = dest_x - x
                dest_dy = dest_y - y
                proj = r1x * dest_dx + r1y * dest_dy

                while beta - alpha > 1.5 * np.pi:
                    beta -= 2 * np.pi
                while beta - alpha < -1.5 * np.pi:
                    beta += 2 * np.pi
                prev_beta = beta
                proj *= SCALE
                if proj > 0.3:
                    beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
                if proj < -0.3:
                    beta += min(TRACK_TURN_RATE, abs(0.001 * proj))

                x += p1x * TRACK_DETAIL_STEP
                y += p1y * TRACK_DETAIL_STEP
                track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

                if laps > 4:
                    break
                no_freeze -= 1
                if no_freeze == 0:
                    break

            # Closed loop detection
            i1, i2 = -1, -1
            i = len(track)
            while True:
                i -= 1
                if i == 0:
                    success = False
                    break
                pass_through_start = (
                        track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
                )
                if pass_through_start and i2 == -1:
                    i2 = i
                elif pass_through_start and i1 == -1:
                    i1 = i
                    break

            if i1 == -1 or i2 == -1:
                success = False
            else:
                track = track[i1: i2 - 1]
                first_beta = track[0][1]
                first_perp_x = np.cos(first_beta)
                first_perp_y = np.sin(first_beta)
                well_glued_together = np.sqrt(
                    (first_perp_x * (track[0][2] - track[-1][2])) ** 2 +
                    (first_perp_y * (track[0][3] - track[-1][3])) ** 2
                )
                if well_glued_together > TRACK_DETAIL_STEP:
                    success = False
                else:
                    success = True

            if success:
                break
            else:
                retries += 1
                if self.verbose:
                    print(f"Retry {retries} for map_seed {self.map_seed} -> seed {current_seed}")

        if not success:
            raise RuntimeError(
                f"Failed to generate valid track after {max_retries} retries for map_seed {self.map_seed}")

        # === REST OF ORIGINAL GENERATION ===
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good

        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * np.cos(beta1), y1 - TRACK_WIDTH * np.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * np.cos(beta1), y1 + TRACK_WIDTH * np.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * np.cos(beta2), y2 - TRACK_WIDTH * np.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * np.cos(beta2), y2 + TRACK_WIDTH * np.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]

            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append((vertices, t.color))
            self.road.append(t)

            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * np.cos(beta1), y1 + side * TRACK_WIDTH * np.sin(beta1))
                b1_r = (
                x1 + side * (TRACK_WIDTH + BORDER) * np.cos(beta1), y1 + side * (TRACK_WIDTH + BORDER) * np.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * np.cos(beta2), y2 + side * TRACK_WIDTH * np.sin(beta2))
                b2_r = (
                x2 + side * (TRACK_WIDTH + BORDER) * np.cos(beta2), y2 + side * (TRACK_WIDTH + BORDER) * np.sin(beta2))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (255, 255, 255) if i % 2 == 0 else (255, 0, 0)))

        self.track = track
        return True
