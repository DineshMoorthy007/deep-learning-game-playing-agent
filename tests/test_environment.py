import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.environment import TicTacToeEnvironment


def test_environment_initialization():
    env = TicTacToeEnvironment()
    state = env.get_state()
    assert state.shape == (9,)
    assert np.all(state == 0)
    assert len(env.get_valid_actions()) == 9
    assert env.check_winner() is None
    assert not env.is_terminal()
    print("test_environment_initialization: PASSED")


def test_valid_and_invalid_moves():
    env = TicTacToeEnvironment()
    assert env.is_valid_action(0)
    assert env.is_valid_action(8)
    assert not env.is_valid_action(-1)
    assert not env.is_valid_action(9)

    env.step(4, player=1)
    assert not env.is_valid_action(4)
    assert 4 not in env.get_valid_actions()
    assert len(env.get_valid_actions()) == 8
    print("test_valid_and_invalid_moves: PASSED")


def test_horizontal_win():
    env = TicTacToeEnvironment()
    # Row 0: 0, 1, 2
    env.step(0, player=1)
    env.step(3, player=-1)
    env.step(1, player=1)
    env.step(4, player=-1)
    state, reward, done, info = env.step(2, player=1)

    assert done is True
    assert reward == 1.0
    assert info["winner"] == 1
    assert env.check_winner() == 1
    print("test_horizontal_win: PASSED")


def test_vertical_win():
    env = TicTacToeEnvironment()
    # Col 1: 1, 4, 7 for opponent
    env.step(0, player=1)
    env.step(1, player=-1)
    env.step(2, player=1)
    env.step(4, player=-1)
    env.step(5, player=1)
    state, reward, done, info = env.step(7, player=-1)

    assert done is True
    assert reward == -1.0
    assert info["winner"] == -1
    assert env.check_winner() == -1
    print("test_vertical_win: PASSED")


def test_diagonal_win():
    env = TicTacToeEnvironment()
    # Diagonal: 0, 4, 8
    env.step(0, player=1)
    env.step(1, player=-1)
    env.step(4, player=1)
    env.step(2, player=-1)
    state, reward, done, info = env.step(8, player=1)

    assert done is True
    assert reward == 1.0
    assert info["winner"] == 1
    print("test_diagonal_win: PASSED")


def test_draw():
    env = TicTacToeEnvironment()
    # X O X
    # X X O
    # O X O
    moves = [
        (0, 1), (1, -1), (2, 1),
        (4, -1), (3, 1), (5, -1),
        (7, 1), (6, -1), (8, 1)  # wait, let's verify no win
    ]
    # Let's use standard draw sequence:
    # 0(X), 1(O), 2(X)
    # 4(O), 3(X), 5(O)
    # 7(X), 6(O), 8(X)
    env.reset()
    env.step(0, 1)   # X . .
    env.step(1, -1)  # X O .
    env.step(2, 1)   # X O X
    env.step(4, -1)  # X O X / . O .
    env.step(3, 1)   # X O X / X O .
    env.step(5, -1)  # X O X / X O O
    env.step(7, 1)   # X O X / X O O / . X .
    env.step(6, -1)  # X O X / X O O / O X .
    state, reward, done, info = env.step(8, 1)  # X O X / X O O / O X X

    assert done is True
    assert info["winner"] == 0
    assert reward == 0.2
    assert env.check_winner() == 0
    print("test_draw: PASSED")


def test_reset():
    env = TicTacToeEnvironment()
    env.step(0, 1)
    env.step(1, -1)
    assert len(env.get_valid_actions()) == 7
    env.reset()
    assert len(env.get_valid_actions()) == 9
    assert np.all(env.get_state() == 0)
    print("test_reset: PASSED")


if __name__ == "__main__":
    test_environment_initialization()
    test_valid_and_invalid_moves()
    test_horizontal_win()
    test_vertical_win()
    test_diagonal_win()
    test_draw()
    test_reset()
    print("\nALL ENVIRONMENT UNIT TESTS PASSED SUCCESSFULLY!")
