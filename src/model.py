"""
Deep Q-Network (DQN) Neural Network Architecture
Defines the Multi-Layer Perceptron (MLP) mapping board state representations
to action Q-values using TensorFlow/Keras.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dqn_model(
    input_dim: int = 9,
    output_dim: int = 9,
    hidden_units: Tuple[int, int] = (64, 64),
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs and compiles the Deep Q-Network Multi-Layer Perceptron.

    Architecture:
        - Input Layer: input_dim neurons (9 board cells)
        - Dense Hidden Layer 1: hidden_units[0] neurons (64), ReLU activation
        - Dense Hidden Layer 2: hidden_units[1] neurons (64), ReLU activation
        - Dense Output Layer: output_dim neurons (9), Linear activation

    Args:
        input_dim (int): Number of state features (9 for Tic-Tac-Toe).
        output_dim (int): Number of discrete actions (9 for Tic-Tac-Toe).
        hidden_units (Tuple[int, int]): Neuron counts for hidden layers.
        learning_rate (float): Step size for Adam optimizer.

    Returns:
        keras.Model: Compiled Keras neural network model.
    """
    inputs = layers.Input(shape=(input_dim,), name="board_state_input")
    x = layers.Dense(hidden_units[0], activation="relu", name="dense_hidden_1")(inputs)
    x = layers.Dense(hidden_units[1], activation="relu", name="dense_hidden_2")(x)
    outputs = layers.Dense(output_dim, activation="linear", name="q_value_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="tic_tac_toe_dqn")

    # Huber loss provides quadratic behavior for small errors and linear for large,
    # preventing gradient explosion in reinforcement learning TD targets.
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.Huber()

    model.compile(optimizer=optimizer, loss=loss_fn)
    return model
