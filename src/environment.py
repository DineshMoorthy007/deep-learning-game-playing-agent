"""
Tic-Tac-Toe Environment for Reinforcement Learning
Defines the 3x3 board dynamics, transition rules, valid-action detection,
terminal-state evaluations, and scalar reward assignments.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class TicTacToeEnvironment:
    """
    Tic-Tac-Toe environment simulating a discrete 3x3 grid game.

    Board State Encoding:
         1 : AI / Current Agent mark (X or O)
        -1 : Opponent mark
         0 : Empty cell
    """

    WIN_COMBINATIONS = [
        # Rows
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        # Columns
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        # Diagonals
        (0, 4, 8),
        (2, 4, 6),
    ]

    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros(9, dtype=np.int8)
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets the board to an initial empty state.

        Returns:
            np.ndarray: 9-element array of zeros representing the clean board.
        """
        self.board = np.zeros(9, dtype=np.int8)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Returns a copy of the current 9-element board state array.
        """
        return np.copy(self.board)

    def get_valid_actions(self) -> List[int]:
        """
        Returns a list of indices (0-8) corresponding to empty board cells.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def is_valid_action(self, action: int) -> bool:
        """
        Checks if the specified action is within range and the cell is empty.

        Args:
            action: Cell index between 0 and 8.

        Returns:
            bool: True if action is legal, False otherwise.
        """
        return 0 <= action < 9 and self.board[action] == 0

    def step(self, action: int, player: int = 1) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes an action for the specified player.

        Args:
            action: Board cell index (0-8).
            player: 1 for AI agent, -1 for Opponent.

        Returns:
            Tuple of (next_state, reward, done, info):
                - next_state (np.ndarray): 9-element board array.
                - reward (float): Scalar feedback (+1.0 win, -1.0 loss, +0.2 draw, 0.0 intermediate).
                - done (bool): True if terminal state reached, False otherwise.
                - info (dict): Detailed transition metadata (e.g. winner identifier).
        """
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action {action} attempted on board: {self.board.tolist()}")

        self.board[action] = player
        winner = self.check_winner()

        if winner is not None:
            if winner == 1:
                # AI Win
                return self.get_state(), 1.0, True, {"winner": 1, "status": "win"}
            elif winner == -1:
                # Opponent Win
                return self.get_state(), -1.0, True, {"winner": -1, "status": "loss"}
            elif winner == 0:
                # Draw / Stalemate
                return self.get_state(), 0.2, True, {"winner": 0, "status": "draw"}

        # Game continues
        return self.get_state(), 0.0, False, {"winner": None, "status": "ongoing"}

    def check_winner(self) -> Optional[int]:
        """
        Evaluates the board for winning configurations or stalemates.

        Returns:
            1 if Player 1 (AI) wins,
            -1 if Player -1 (Opponent) wins,
            0 if the game is a Draw (board full with no winner),
            None if the game is ongoing.
        """
        for c1, c2, c3 in self.WIN_COMBINATIONS:
            line_sum = self.board[c1] + self.board[c2] + self.board[c3]
            if line_sum == 3:
                return 1
            if line_sum == -3:
                return -1

        if len(self.get_valid_actions()) == 0:
            return 0  # Draw

        return None

    def is_terminal(self) -> bool:
        """
        Returns True if the game has ended (win, loss, or draw), False otherwise.
        """
        return self.check_winner() is not None

    def render_cli(self) -> str:
        """
        Generates a human-readable ASCII representation of the board.
        """
        symbols = {1: "X", -1: "O", 0: " "}
        cells = [symbols[val] for val in self.board]
        grid = (
            f"\n {cells[0]} | {cells[1]} | {cells[2]} \n"
            f"---+---+---\n"
            f" {cells[3]} | {cells[4]} | {cells[5]} \n"
            f"---+---+---\n"
            f" {cells[6]} | {cells[7]} | {cells[8]} \n"
        )
        return grid
