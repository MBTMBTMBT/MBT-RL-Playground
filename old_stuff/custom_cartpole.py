from typing import Optional
from gymnasium import logger
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class CustomCartPoleEnv(CartPoleEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
    ):
        """
        Initialize the custom CartPole environment.

        Parameters:
            gravity (float): Gravitational constant.
            masscart (float): Mass of the cart.
            masspole (float): Mass of the pole.
            length (float): Half the length of the pole.
            force_mag (float): Magnitude of the applied force.
            render_mode (Optional[str]): Rendering mode.
            max_episode_steps (int): Maximum number of steps per episode.
        """
        super().__init__()

        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns an initial observation.

        Parameters:
            seed (Optional[int]): Random seed.
            options (Optional[dict]): Additional options.

        Returns:
            Tuple[np.ndarray, dict]: The initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.steps = 0
        low, high = -0.05, 0.05
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        """
        Executes a step in the environment.

        Parameters:
            action (int): The action to take.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, reward, terminated, truncated, and info.
        """
        assert action in self.action_space, f"Invalid action: {action}"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        self.steps += 1

        terminated = x < -2.4 or x > 2.4 or theta < -0.2095 or theta > 0.2095

        truncated = self.steps >= self.max_episode_steps

        reward = 1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()

        return (
            np.array(self.state, dtype=np.float32),
            reward,
            terminated,
            truncated,
            {},
        )
