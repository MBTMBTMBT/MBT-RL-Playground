import torch
from tianshou.policy import PPOPolicy, SACPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tqdm import tqdm


class TianshouPPO:
    def __init__(self, env, policy="MlpPolicy", learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 n_steps=2048, batch_size=64, verbose=0, device="auto"):
        """
        Tianshou wrapper for PPO.

        :param env: Custom Gymnasium environment (single instance).
        :param policy: Policy type, currently supports only "MlpPolicy".
        :param learning_rate: Learning rate for optimizer.
        :param gamma: Discount factor.
        :param gae_lambda: GAE lambda.
        :param n_steps: Number of steps per update.
        :param batch_size: Batch size for training.
        :param verbose: Verbosity level.
        :param device: Device for computation ("auto", "cpu", or "cuda").
        """
        if policy != "MlpPolicy":
            raise NotImplementedError(
                f"Policy '{policy}' is not implemented. Currently, only 'MlpPolicy' is supported.")

        self.env = env
        self.verbose = verbose
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.n

        # Define actor-critic network
        net = Net(self.state_shape, hidden_sizes=[64, 64], device=self.device)
        actor_critic = ActorCritic(net, self.action_shape, device=self.device)
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)

        # Define PPO policy
        self.policy = PPOPolicy(
            actor=actor_critic.actor,
            critic=actor_critic.critic,
            optim=self.optimizer,
            dist_fn=torch.distributions.Categorical,
            discount_factor=gamma,
            gae_lambda=gae_lambda,
            max_grad_norm=0.5,
        )

        # Replay buffer and collector
        self.buffer = ReplayBuffer(size=n_steps)
        self.collector = Collector(self.policy, env, self.buffer)
        self.batch_size = batch_size

    def predict(self, state, deterministic=True):
        """
        Predict action given state.

        :param state: Current state.
        :param deterministic: Whether to use deterministic policy.
        :return: Predicted action.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy(state, deterministic=deterministic)[0]
        return action.cpu().numpy()

    def learn(self, total_timesteps, progress_bar=False):
        """
        Train PPO for given timesteps.

        :param total_timesteps: Total number of timesteps to train.
        :param progress_bar: Display progress bar during training.
        """
        step = 0
        pbar = None
        if progress_bar:
            pbar = tqdm(total=total_timesteps, desc="Training PPO")
        while step < total_timesteps:
            self.collector.collect(n_step=self.buffer.maxsize)
            losses = self.policy.update(self.buffer, batch_size=self.batch_size)
            self.collector.reset_buffer()
            step += self.buffer.maxsize
            if self.verbose:
                print(f"Step: {step}, Losses: {losses}")
            if progress_bar:
                pbar.update(self.buffer.maxsize)
        if progress_bar:
            pbar.close()


class TianshouSAC:
    def __init__(self, env, policy="MlpPolicy", learning_rate=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=1000000, batch_size=256, verbose=0, device="auto"):
        """
        Tianshou wrapper for SAC.

        :param env: Custom Gymnasium environment (single instance).
        :param policy: Policy type, currently supports only "MlpPolicy".
        :param learning_rate: Learning rate for optimizers.
        :param gamma: Discount factor.
        :param tau: Soft update coefficient.
        :param alpha: Entropy coefficient.
        :param buffer_size: Size of replay buffer.
        :param batch_size: Batch size for training.
        :param verbose: Verbosity level.
        :param device: Device for computation ("auto", "cpu", or "cuda").
        """
        if policy != "MlpPolicy":
            raise NotImplementedError(f"Policy '{policy}' is not implemented. Currently, only 'MlpPolicy' is supported.")

        self.env = env
        self.verbose = verbose
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
        self.state_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape[0]

        # Define networks
        actor = ActorProb(Net(self.state_shape, hidden_sizes=[64, 64], device=self.device), self.action_shape, max_action=1.0)
        critic1 = Critic(Net(self.state_shape + self.action_shape, hidden_sizes=[64, 64], device=self.device))
        critic2 = Critic(Net(self.state_shape + self.action_shape, hidden_sizes=[64, 64], device=self.device))

        # Define optimizers
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=learning_rate)

        # Define SAC policy
        self.policy = SACPolicy(
            actor=actor,
            actor_optim=self.actor_optim,
            critic1=critic1,
            critic1_optim=self.critic_optim,
            critic2=critic2,
            critic2_optim=self.critic_optim,
            tau=tau,
            alpha=alpha,
            gamma=gamma,
            action_space=env.action_space,
        )

        # Replay buffer and collector
        self.buffer = ReplayBuffer(size=buffer_size)
        self.collector = Collector(self.policy, env, self.buffer)
        self.batch_size = batch_size

    def predict(self, state, deterministic=True):
        """
        Predict action given state.

        :param state: Current state.
        :param deterministic: Whether to use deterministic policy.
        :return: Predicted action.
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy(state, deterministic=deterministic)[0]
        return action.cpu().numpy()

    def learn(self, total_timesteps, progress_bar=False):
        """
        Train SAC for given timesteps.

        :param total_timesteps: Total number of timesteps to train.
        :param progress_bar: Display progress bar during training.
        """
        step = 0
        pbar = None
        if progress_bar:
            pbar = tqdm(total=total_timesteps, desc="Training SAC")
        while step < total_timesteps:
            self.collector.collect(n_step=1000)
            losses = self.policy.update(self.buffer, batch_size=self.batch_size)
            step += 1000
            if self.verbose:
                print(f"Step: {step}, Losses: {losses}")
            if progress_bar:
                pbar.update(1000)
        if progress_bar:
            pbar.close()
