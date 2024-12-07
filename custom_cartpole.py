from typing import Optional
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
