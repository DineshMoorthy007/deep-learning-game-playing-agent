"""
Deep Q-Network (DQN) Agent Implementation
Encapsulates policy decision-making, epsilon-greedy exploration, experience
replay buffering, temporal difference target calculation, and batch gradient updates.
"""

from collections import deque
import os
import random
from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model import build_dqn_model


class DQNAgent:
    """
    DQN Agent capable of learning game strategies through interaction and experience replay.
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 9,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        memory_size: int = 50000,
        batch_size: int = 64,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Experience replay buffer
        self.memory: deque = deque(maxlen=memory_size)

        # Primary Deep Q-Network
        self.model: keras.Model = build_dqn_model(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_units=(64, 64),
            learning_rate=learning_rate,
        )

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Appends an environmental transition tuple to the replay buffer.
        """
        self.memory.append((state.astype(np.float32), action, reward, next_state.astype(np.float32), done))

    def act(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Selects an action using the epsilon-greedy policy with valid-action masking.

        Args:
            state (np.ndarray): 9-element array representing the board.
            valid_actions (List[int]): List of valid empty cell indices.
            training (bool): If True, applies epsilon exploration; if False, behaves greedily.

        Returns:
            int: Selected action index (0-8).
        """
        if not valid_actions:
            raise ValueError("Cannot select action: valid_actions list is empty.")

        # Exploration: choose random valid action
        if training and np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation: evaluate Q-values and pick argmax among valid actions
        q_values = self.get_q_values(state)

        # Valid-action masking: mask invalid actions with large negative value
        masked_q = {action: q_values[action] for action in valid_actions}
        best_action = max(masked_q, key=masked_q.get)  # type: ignore[arg-type]
        return best_action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Performs a fast forward pass through the neural network to retrieve Q-values for all 9 actions.

        Args:
            state (np.ndarray): 9-element board state.

        Returns:
            np.ndarray: Array of 9 scalar Q-values.
        """
        state_tensor = tf.convert_to_tensor(state.astype(np.float32).reshape(1, 9))
        q_preds = self.model(state_tensor, training=False).numpy()
        return q_preds[0]

    def replay(self, batch_size: Optional[int] = None) -> float:
        """
        Trains the neural network by sampling a mini-batch from experience replay memory.

        Returns:
            float: Training loss value from the batch update (or 0.0 if insufficient memory).
        """
        b_size = batch_size or self.batch_size
        if len(self.memory) < b_size:
            return 0.0

        minibatch = random.sample(self.memory, b_size)

        states = np.array([transition[0] for transition in minibatch], dtype=np.float32)
        actions = np.array([transition[1] for transition in minibatch], dtype=np.int32)
        rewards = np.array([transition[2] for transition in minibatch], dtype=np.float32)
        next_states = np.array([transition[3] for transition in minibatch], dtype=np.float32)
        dones = np.array([transition[4] for transition in minibatch], dtype=bool)

        # Fast forward pass for current and next-state Q-values
        states_tensor = tf.convert_to_tensor(states)
        next_states_tensor = tf.convert_to_tensor(next_states)

        current_q_targets = self.model(states_tensor, training=False).numpy()
        next_q_values = self.model(next_states_tensor, training=False).numpy()

        # Compute Bellman targets with valid-action masking for next state:
        # y = r if terminal, else y = r + gamma * max_{a' in valid(s')} Q(s', a')
        for i in range(b_size):
            if dones[i]:
                target = rewards[i]
            else:
                # Mask invalid actions in next state
                valid_next_actions = [idx for idx in range(9) if next_states[i, idx] == 0]
                if valid_next_actions:
                    max_next_q_val = np.max([next_q_values[i, act] for act in valid_next_actions])
                else:
                    max_next_q_val = 0.0
                target = rewards[i] + self.gamma * max_next_q_val

            # Bound Q-value targets between -1.0 and +1.0 for Tic-Tac-Toe
            target = np.clip(target, -1.0, 1.0)
            current_q_targets[i, actions[i]] = target

        # Perform single gradient descent update on the mini-batch
        loss = float(self.model.train_on_batch(states, current_q_targets))
        return loss

    def decay_epsilon(self) -> None:
        """
        Decays the exploration probability epsilon down to epsilon_min.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """
        Saves the trained neural network model to disk.

        Args:
            filepath (str): Target file path (.keras or .h5).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved successfully to: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Loads pre-trained neural network weights from disk.

        Args:
            filepath (str): Source model file path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{filepath}'. "
                "Please train the agent first using 'python -m src.train'."
            )
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded successfully from: {filepath}")
