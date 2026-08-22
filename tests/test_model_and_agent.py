import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf

from src.agent import DQNAgent
from src.environment import TicTacToeEnvironment
from src.model import build_dqn_model


def test_dqn_model_architecture():
    model = build_dqn_model(input_dim=9, output_dim=9, hidden_units=(64, 64), learning_rate=0.001)
    assert len(model.layers) == 4
    assert model.input_shape == (None, 9)
    assert model.output_shape == (None, 9)

    # Test single-sample prediction
    dummy_input = np.zeros((1, 9), dtype=np.float32)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 9)

    # Test batch prediction
    batch_input = np.zeros((32, 9), dtype=np.float32)
    batch_output = model.predict(batch_input, verbose=0)
    assert batch_output.shape == (32, 9)
    print("test_dqn_model_architecture: PASSED")


def test_agent_act_and_masking():
    agent = DQNAgent(state_dim=9, action_dim=9, epsilon=0.0)  # Pure exploitation
    state = np.zeros(9, dtype=np.int8)

    # Only cell 3 and 7 valid
    valid_actions = [3, 7]
    action = agent.act(state, valid_actions, training=False)
    assert action in valid_actions

    # Test with epsilon=1.0 (pure exploration)
    agent.epsilon = 1.0
    for _ in range(20):
        action = agent.act(state, valid_actions, training=True)
        assert action in valid_actions
    print("test_agent_act_and_masking: PASSED")


def test_agent_replay_training_step():
    agent = DQNAgent(state_dim=9, action_dim=9, batch_size=4)
    env = TicTacToeEnvironment()

    # Fill replay buffer with dummy experiences
    for i in range(10):
        s = np.random.choice([-1, 0, 1], size=9).astype(np.float32)
        a = int(np.random.randint(0, 9))
        r = float(np.random.choice([-1.0, 0.0, 0.2, 1.0]))
        s_next = np.random.choice([-1, 0, 1], size=9).astype(np.float32)
        done = bool(np.random.choice([True, False]))
        agent.remember(s, a, r, s_next, done)

    assert len(agent.memory) == 10
    loss = agent.replay(batch_size=4)
    assert isinstance(loss, float)
    assert loss >= 0.0
    print("test_agent_replay_training_step: PASSED")


def test_agent_save_and_load(tmp_path: str = "models/test_model.keras"):
    agent = DQNAgent(state_dim=9, action_dim=9)
    agent.save(tmp_path)
    assert os.path.exists(tmp_path)

    new_agent = DQNAgent(state_dim=9, action_dim=9)
    new_agent.load(tmp_path)

    # Compare predictions
    dummy_input = np.ones(9, dtype=np.int8)
    q1 = agent.get_q_values(dummy_input)
    q2 = new_agent.get_q_values(dummy_input)
    assert np.allclose(q1, q2, atol=1e-5)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("test_agent_save_and_load: PASSED")


if __name__ == "__main__":
    test_dqn_model_architecture()
    test_agent_act_and_masking()
    test_agent_replay_training_step()
    test_agent_save_and_load()
    print("\nALL MODEL AND AGENT UNIT TESTS PASSED SUCCESSFULLY!")
