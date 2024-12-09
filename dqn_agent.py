import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List
from collections import defaultdict
import random
from torch.utils.data import DataLoader, TensorDataset


class DQNAgent:
    def __init__(self,
                 input_dims: int,
                 action_space: List[int],
                 hidden_layers: List[int],
                 max_q_value_abs: float = 1.0,
                 replay_buffer_size: int = 10000,
                 batch_size: int = 64,
                 train_epochs: int = 1):
        """
        Initialize the DQN agent.

        :param input_dims: Dimension of the input state.
        :param action_space: List of possible values for each action dimension.
        :param hidden_layers: List specifying the number of neurons in each hidden layer.
        :param replay_buffer_size: Maximum size of the replay buffer.
        :param batch_size: Number of samples per training batch.
        :param train_epochs: Number of epochs to train during each training session.
        """
        self.input_dims = input_dims
        self.action_space = action_space
        self.hidden_layers = hidden_layers

        self.max_q_value_abs = abs(max_q_value_abs)

        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        # Replay buffer to store experiences
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.replay_buffer_size = replay_buffer_size

        # Define the DQN model
        self.model = self._build_model().to(self.device)

        self.criterion = nn.MSELoss()

        # Initialize Q-table
        self.q_table = defaultdict(lambda: 0.0)

    def _build_model(self) -> nn.Module:
        """Build a feedforward neural network based on the specified hidden layers."""
        layers = []
        input_size = self.input_dims
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, np.prod(self.action_space)))
        return nn.Sequential(*layers)

    def get_action_probabilities(self, state: np.ndarray, strategy: str = "greedy",
                                 temperature: float = 1.0) -> np.ndarray:
        """
        Calculate action probabilities based on the specified strategy.

        :param state: The current state.
        :param strategy: Strategy to calculate probabilities ("greedy" or "softmax").
        :param temperature: Temperature parameter for softmax.
        :return: An array of action probabilities.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze(0).detach().cpu().numpy()

        if strategy == "greedy":
            probabilities = np.zeros_like(q_values, dtype=float)
            probabilities[np.argmax(q_values)] = 1.0
        elif strategy == "softmax":
            exp_values = np.exp(q_values / temperature)
            probabilities = exp_values / np.sum(exp_values)
        else:
            raise ValueError("Invalid strategy. Use 'greedy' or 'softmax'.")

        return probabilities

    def select_action(self, state: np.ndarray, strategy: str = "greedy", temperature: float = 1.0) -> List[int]:
        """
        Select an action based on the strategy.

        :param state: The current state.
        :param strategy: Strategy to calculate probabilities ("greedy" or "softmax").
        :param temperature: Temperature parameter for softmax.
        :return: A selected action.
        """
        probabilities = self.get_action_probabilities(state, strategy, temperature)
        flat_action = np.random.choice(len(probabilities), p=probabilities)
        action = self._flat_index_to_action(flat_action)
        return action

    def _flat_index_to_action(self, index: int) -> List[int]:
        """Convert a flat action index to a multi-dimensional action based on the action space."""
        sizes = self.action_space
        action = []
        for size in reversed(sizes):
            action.append(index % size)
            index //= size
        return action[::-1]

    def _action_to_flat_index(self, action: List[int]) -> int:
        """Convert a multi-dimensional action to a flat action index."""
        sizes = self.action_space
        index = 0
        for i, size in enumerate(sizes):
            index = index * size + action[i]
        return index

    def store_experience(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, done: bool):
        """
        Store an experience in the replay buffer.

        :param state: Current state.
        :param action: Taken action.
        :param reward: Received reward.
        :param next_state: Next state.
        :param done: Whether the episode has terminated.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def _train_from_replay_buffer(self, alpha: float, gamma: float):
        """
        Train the model using samples from the replay buffer with given learning rate and discount factor.
        After training, randomly discard half of the replay buffer.
        """
        if len(self.replay_buffer) < self.replay_buffer_size:
            return

        # Prepare dataset and dataloader
        experiences = list(self.replay_buffer)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor([self._action_to_flat_index(a) for a in actions], dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.SGD(self.model.parameters(), lr=alpha)

        losses = 0.0
        counter = 0
        for _ in range(self.train_epochs):
            for batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones in dataloader:
                q_values = self.model(batch_states)
                q_value = q_values[torch.arange(len(batch_actions)), batch_actions]

                with torch.no_grad():
                    next_q_values = self.model(batch_next_states)
                    max_next_q_value = torch.max(next_q_values, dim=1).values

                # Rescale rewards
                batch_rewards /= self.max_q_value_abs

                # Correctly handle the `done` flag
                target = batch_rewards + gamma * max_next_q_value * (1 - batch_dones)
                target[batch_dones.bool()] = batch_rewards[batch_dones.bool()]  # Override target for done states

                loss = self.criterion(q_value, target)
                losses += loss.item()
                counter += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Discard half of the replay buffer
        remaining_size = len(self.replay_buffer) // 2
        self.replay_buffer = deque(random.sample(self.replay_buffer, remaining_size), maxlen=self.replay_buffer.maxlen)
        # print("Loss:", losses / counter)

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, done: bool,
               alpha: float = 0.001, gamma: float = 0.99):
        """
        Store the experience and trigger training if sufficient data is available.

        :param state: Current state.
        :param action: Taken action.
        :param reward: Received reward.
        :param next_state: Next state.
        :param done: Whether the episode has terminated.
        """
        self.store_experience(state, action, reward, next_state, done)
        self._train_from_replay_buffer(alpha, gamma)

        # Update a Q-table entry using DQN-predicted Q-values
        state_key = tuple(state)
        action_key = tuple(action)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze(0).detach().cpu().numpy()

        # Rescale q values
        q_values *= self.max_q_value_abs

        if not hasattr(self, 'q_table'):
            from collections import defaultdict
            self.q_table = defaultdict(lambda: 0.0)  # Initialize Q-table

        self.q_table[(state_key, action_key)] = q_values[self._action_to_flat_index(action)]

    def clone(self):
        """
        Create a deep copy of the current agent.

        :return: A new DQNAgent instance with the same parameters and model state.
        """
        cloned_agent = DQNAgent(
            input_dims=self.input_dims,
            action_space=self.action_space,
            hidden_layers=self.hidden_layers,
            replay_buffer_size=self.replay_buffer.maxlen,
            batch_size=self.batch_size,
            train_epochs=self.train_epochs
        )
        cloned_agent.model.load_state_dict(self.model.state_dict())
        cloned_agent.replay_buffer = deque(self.replay_buffer, maxlen=self.replay_buffer.maxlen)
        cloned_agent.q_table = self.q_table.copy()
        return cloned_agent

    def save_model(self, file_path: str):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str):
        """Load the model from a file."""
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.to(self.device)

    def save_q_table(self, file_path: str = None) -> pd.DataFrame:
        """
        Save the Q-Table to a CSV file and/or return as a DataFrame.

        :param file_path: Path to save the file.
        :return: DataFrame representation of the Q-Table.
        """
        data = []
        for (state, action), q_value in self.q_table.items():
            row = {f"state_dim_{i}": state[i] for i in range(len(state))}
            row.update({f"action_dim_{j}": action[j] for j in range(len(action))})
            row.update({"q_value": q_value})
            data.append(row)
        df = pd.DataFrame(data)
        if file_path:
            df.to_csv(file_path, index=False)
            print(f"Q-Table saved to {file_path}.")
        return df
