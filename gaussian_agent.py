import numpy as np
from typing import Tuple, Optional, Union, Any, Literal
from sbx import SAC
import jax.numpy as jnp
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import SelfOffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback, TrainFreq


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
    fused_var = (1 - p) * (std1**2) + p * (std2**2)
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
        Return the mean and std of the Gaussian action distribution predicted by the policy network.
        No Tanh applied. This is suitable for analyzing the policy distribution itself.

        Args:
            np_state (np.ndarray): State input with shape (batch_size, obs_dim)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and std of Gaussian distribution before Tanh.
        """
        jax_state = jnp.asarray(np_state)

        # Apply policy actor network
        dist = self.policy.actor.apply(self.policy.actor_state.params, jax_state)

        # Get mean and std directly from the Gaussian distribution
        mean = dist.distribution.loc  # (batch_size, action_dim)
        std = dist.distribution._scale_diag  # (batch_size, action_dim)

        return np.array(mean), np.array(std)

    def get_default_action_distribution(
        self, np_state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the default action distribution for comparison.
        By default, use untrained Gaussian policy distribution: mean=0, std=1.

        Args:
            np_state (np.ndarray): Input states, shape (batch_size, obs_dim)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and std, shape (batch_size, action_dim)
        """
        batch_size = np_state.shape[0]
        action_dim = self.action_space.shape[0]

        # Standard untrained Gaussian Policy
        mean = np.zeros((batch_size, action_dim), dtype=np.float32)
        std = np.ones((batch_size, action_dim), dtype=np.float32)

        return mean, std

    def sample_action_from_distribution(
            self,
            mean: np.ndarray,
            std: np.ndarray,
            deterministic: bool = False,
    ) -> np.ndarray:
        """
        Sample actions from a given Gaussian distribution (mean, std),
        apply Tanh squashing, and rescale to the environment's action space.

        This version matches the training sampling logic.

        Args:
            mean (np.ndarray): shape (batch_size, action_dim)
            std (np.ndarray): shape (batch_size, action_dim)
            deterministic (bool): If True, use mean directly without noise.

        Returns:
            np.ndarray: Actions after rescaling, shape (batch_size, action_dim)
        """
        if deterministic:
            raw_action = mean  # No noise
        else:
            noise = np.random.randn(*mean.shape)
            raw_action = mean + std * noise

        # Squash action to [-1, 1] using tanh
        squashed_action = np.tanh(raw_action)

        # Rescale to action_space
        action = self.policy.unscale_action(squashed_action)

        return action

    def learn(
        self: SelfOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOffPolicyAlgorithm:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # Additional condition: make sure replay buffer is sufficiently filled
                buffer_ready = (
                        self.replay_buffer.size >= min(self.learning_starts, self.replay_buffer.buffer_size)
                )
                if buffer_ready:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                    # Special case when the user passes `gradient_steps=0`
                    if gradient_steps > 0:
                        self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self


class MixPolicySAC:
    def __init__(
            self,
            model: SACJax,
    ):
        self.model = model
        self.action_space = model.action_space
        self.action_dim = model.action_space.shape[0]
        self.obs_dim = model.observation_space.shape[0]
        self.device = model.device

    def predict_action_distribution_with_weight(
            self, np_state: np.ndarray, p: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the mean and std of the mixed Gaussian action distribution.

        The mixed distribution is obtained by linearly interpolating
        between the prior policy distribution and the current policy distribution.

        Args:
            np_state (np.ndarray): shape (batch_size, obs_dim)
            p (float): weight for the current policy, in [0, 1]

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mixed mean and std (before tanh)
        """
        assert 0.0 <= p <= 1.0, "Weight p must be in [0, 1]"

        predicted_mean, predicted_std = self.model.predict_action_distribution(np_state)
        prior_mean, prior_std = self.model.get_default_action_distribution(np_state)

        # Linear interpolation for mean
        mixed_mean = p * predicted_mean + (1.0 - p) * prior_mean

        # Interpolation for variance (more stable)
        mixed_var = p * (predicted_std ** 2) + (1.0 - p) * (prior_std ** 2)
        mixed_std = np.sqrt(mixed_var)

        return mixed_mean, mixed_std

    def predict_with_weight(
            self, np_state: np.ndarray, p: float = 1.0
    ) -> np.ndarray:
        action_mean, action_std = self.predict_action_distribution_with_weight(np_state, p)
        action = self.model.sample_action_from_distribution(action_mean, action_std)
        return action
