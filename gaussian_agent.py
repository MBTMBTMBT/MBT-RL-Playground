import numpy as np
from typing import Tuple, Optional, Union, Any, Literal
from sbx import SAC
import jax.numpy as jnp
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule


def sample_gaussian_action(
    mean: np.ndarray,
    log_std: np.ndarray,
    low: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    squash: bool = False,
) -> np.ndarray:
    """
    Sample actions from a Gaussian distribution given mean and log_std, using vectorized NumPy.

    Args:
        mean (np.ndarray): Mean of action distribution, shape (batch_size, action_dim)
        log_std (np.ndarray): Log standard deviation, shape (batch_size, action_dim)
        low (Optional[np.ndarray]): Lower bound of action space. If None, no clipping is done.
        high (Optional[np.ndarray]): Upper bound of action space. If None, no clipping is done.
        squash (bool): Whether to apply tanh squashing to actions.

    Returns:
        np.ndarray: Sampled actions, shape (batch_size, action_dim)
    """
    std = np.exp(log_std)
    noise = np.random.randn(*mean.shape)
    action = mean + std * noise

    if squash:
        action = np.tanh(action)

    if low is not None and high is not None:
        action = np.clip(action, low, high)

    return action


def fuse_gaussian_distributions(
    mean1: np.ndarray,
    log_std1: np.ndarray,
    mean2: np.ndarray,
    log_std2: np.ndarray,
    p: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse two Gaussian distributions using linear weighting.

    Args:
        mean1 (np.ndarray): Mean of first distribution, shape (batch_size, action_dim)
        log_std1 (np.ndarray): Log std of first distribution, same shape
        mean2 (np.ndarray): Mean of second distribution
        log_std2 (np.ndarray): Log std of second distribution
        p (float): Weight for the second distribution (0 <= p <= 1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (mean, log_std) of the fused distribution
    """
    std1 = np.exp(log_std1)
    std2 = np.exp(log_std2)

    # Weighted mean
    fused_mean = (1 - p) * mean1 + p * mean2

    # Weighted variance: Var = E[std^2]
    fused_var = (1 - p) * (std1 ** 2) + p * (std2 ** 2)
    fused_log_std = 0.5 * np.log(fused_var + 1e-8)  # avoid log(0)

    return fused_mean, fused_log_std


class SACJax(SAC):
    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[Literal["auto"], float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        param_resets: Optional[
            list[int]
        ] = None,  # List of timesteps after which to reset the params
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_delay=policy_delay,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            param_resets=param_resets,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def predict_action_distribution(
        self, np_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the mean and std of the action distribution given a batch of states.
        Args:
            np_state (np.ndarray): State input with shape (batch_size, obs_dim)
        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and std of actions with shape (batch_size, action_dim)
        """
        # Convert numpy input to JAX array
        jax_state = jnp.asarray(np_state)

        # Get mean and log_std from policy (assumes SBX JAX policy structure)
        mean, log_std = self.policy.actor(jax_state)

        # Convert to std
        std = jnp.exp(log_std)

        # Return as numpy arrays
        return np.array(mean), np.array(std)

    def get_default_action_distribution(
        self, np_state: np.ndarray, strategy: str = "policy_init"  # or "uniform_like"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a default action distribution before the policy is trained.
        Args:
            np_state (np.ndarray): Input states, shape (batch_size, obs_dim)
            strategy (str): "policy_init" or "uniform_like"
        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and std, shape (batch_size, action_dim)
        """
        batch_size = np_state.shape[0]
        action_dim = self.action_space.shape[0]

        if strategy == "policy_init":
            mean = np.zeros((batch_size, action_dim), dtype=np.float32)
            std = np.ones((batch_size, action_dim), dtype=np.float32)
        elif strategy == "uniform_like":
            center = (self.action_space.low + self.action_space.high) / 2.0
            scale = (self.action_space.high - self.action_space.low) / 2.0
            mean = np.tile(center, (batch_size, 1)).astype(np.float32)
            std = np.tile(scale, (batch_size, 1)).astype(np.float32)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return mean, std
