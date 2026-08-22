"""
Training Pipeline for Deep Q-Network (DQN) Game Playing Agent
Executes multi-episode reinforcement learning against random and strategic
heuristic opponents, logs convergence statistics, and saves the trained model.
"""

import argparse
import os
import random
from typing import Any, Dict, List, Optional
import numpy as np

from src.agent import DQNAgent
from src.environment import TicTacToeEnvironment
from src.utils import plot_training_results, set_seed


class RandomOpponent:
    """Baseline opponent that selects uniformly from available legal moves."""

    @staticmethod
    def get_action(env: TicTacToeEnvironment) -> int:
        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)


class StrategicOpponent:
    """
    Heuristic rule-based opponent that:
      1. Takes an immediate winning move if available.
      2. Blocks an immediate opponent win if possible.
      3. Prefers center cell (4) or corners.
      4. Otherwise selects a random valid move.
    """

    @staticmethod
    def get_action(env: TicTacToeEnvironment) -> int:
        valid_actions = env.get_valid_actions()
        board = env.board

        # 1. Check for immediate winning move for opponent (-1)
        for action in valid_actions:
            board[action] = -1
            if env.check_winner() == -1:
                board[action] = 0
                return action
            board[action] = 0

        # 2. Check for immediate winning move for AI (1) to block
        for action in valid_actions:
            board[action] = 1
            if env.check_winner() == 1:
                board[action] = 0
                return action
            board[action] = 0

        # 3. Prefer center if open
        if 4 in valid_actions:
            return 4

        # 4. Prefer corners
        corners = [a for a in [0, 2, 6, 8] if a in valid_actions]
        if corners:
            return random.choice(corners)

        # 5. Default random valid move
        return random.choice(valid_actions)


def train_dqn(
    episodes: int = 10000,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    memory_size: int = 50000,
    save_path: str = "models/tic_tac_toe_dqn.keras",
    results_dir: str = "results",
    log_interval: int = 500,
    seed: int = 42,
) -> DQNAgent:
    """
    Executes the complete DQN training routine.

    Args:
        episodes: Total number of game episodes to simulate.
        batch_size: Mini-batch size sampled from replay memory.
        learning_rate: Learning rate for Adam optimizer.
        gamma: Discount factor for future rewards.
        epsilon_start: Initial exploration rate.
        epsilon_min: Lower bound floor for exploration.
        epsilon_decay: Multiplicative decay factor per episode.
        memory_size: Maximum capacity of circular replay buffer.
        save_path: Destination path for trained neural network model.
        results_dir: Directory where training charts will be saved.
        log_interval: Episode frequency for console progress output.
        seed: Random seed for reproducibility.

    Returns:
        DQNAgent: Fully trained agent instance.
    """
    set_seed(seed)
    print("=" * 70)
    print("STARTING DEEP Q-NETWORK (DQN) TRAINING PIPELINE")
    print("=" * 70)
    print(f"Target Episodes     : {episodes:,}")
    print(f"Replay Batch Size   : {batch_size}")
    print(f"Learning Rate       : {learning_rate}")
    print(f"Discount Factor (gamma) : {gamma}")
    print(f"Epsilon Schedule    : {epsilon_start} -> {epsilon_min} (decay: {epsilon_decay})")
    print(f"Replay Buffer Size  : {memory_size:,}")
    print(f"Checkpoint Target   : {save_path}")
    print("-" * 70)

    env = TicTacToeEnvironment()
    agent = DQNAgent(
        state_dim=9,
        action_dim=9,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
    )

    random_opp = RandomOpponent()
    strategic_opp = StrategicOpponent()

    # Metric tracking structures
    recent_outcomes: List[str] = []
    losses_history: List[float] = []
    win_rates: List[float] = []
    draw_rates: List[float] = []
    loss_rates: List[float] = []
    epsilons: List[float] = []
    rewards_history: List[float] = []
    checkpoint_episodes: List[int] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_losses: List[float] = []

        # Train against mixed opponent distribution: 50% Random, 50% Strategic
        opponent = strategic_opp if random.random() < 0.5 else random_opp

        # Randomize turn order: 50% AI first, 50% Opponent first
        ai_turn = True
        if random.random() < 0.5:
            # Opponent makes initial move
            opp_act = opponent.get_action(env)
            state, _, done, _ = env.step(opp_act, player=-1)

        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # 1. AI selects and executes action
            action = agent.act(state, valid_actions, training=True)
            next_state, reward, done, info = env.step(action, player=1)
            episode_reward += reward

            if done:
                # Terminal state after AI move (AI win or Draw)
                agent.remember(state, action, reward, next_state, True)
                loss = agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                outcome = info.get("status", "unknown")
                recent_outcomes.append(outcome)
                break

            # 2. Opponent counter-move
            opp_act = opponent.get_action(env)
            opp_next_state, opp_reward, opp_done, opp_info = env.step(opp_act, player=-1)

            if opp_done:
                # Terminal state after opponent move (Opponent win or Draw)
                final_ai_reward = -1.0 if opp_info.get("winner") == -1 else 0.2
                episode_reward += final_ai_reward
                agent.remember(state, action, final_ai_reward, opp_next_state, True)
                loss = agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                outcome = opp_info.get("status", "unknown")
                recent_outcomes.append(outcome)
                done = True
            else:
                # Non-terminal intermediate transition
                agent.remember(state, action, 0.0, opp_next_state, False)
                loss = agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                state = opp_next_state

        # Exploration decay
        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if episode_losses:
            losses_history.append(float(np.mean(episode_losses)))

        # Periodic logging and window evaluation
        if episode % log_interval == 0:
            window = recent_outcomes[-log_interval:]
            wins = window.count("win")
            draws = window.count("draw")
            losses = window.count("loss")
            total = len(window)

            win_rate = (wins / total) * 100 if total > 0 else 0.0
            draw_rate = (draws / total) * 100 if total > 0 else 0.0
            loss_rate = (losses / total) * 100 if total > 0 else 0.0
            avg_loss = float(np.mean(losses_history[-log_interval:])) if losses_history else 0.0

            checkpoint_episodes.append(episode)
            win_rates.append(win_rate)
            draw_rates.append(draw_rate)
            loss_rates.append(loss_rate)
            epsilons.append(agent.epsilon)

            print(
                f"Episode {episode:6d}/{episodes} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Win: {win_rate:5.1f}% | "
                f"Draw: {draw_rate:5.1f}% | "
                f"Loss: {loss_rate:5.1f}% | "
                f"Avg Loss: {avg_loss:.5f}",
                flush=True,
            )

    print("-" * 70)
    print("Training finished. Saving model and generating analytical charts...")
    agent.save(save_path)

    # Plot metrics
    history_data: Dict[str, List[Any]] = {
        "episodes": checkpoint_episodes,
        "win_rates": win_rates,
        "draw_rates": draw_rates,
        "loss_rates": loss_rates,
        "epsilons": epsilons,
        "losses": losses_history,
        "rewards": rewards_history,
    }
    plot_training_results(history_data, save_dir=results_dir)
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Deep Q-Network Agent on Tic-Tac-Toe.")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (default: 0.95)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate (default: 1.0)")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum exploration rate (default: 0.05)")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Epsilon decay factor (default: 0.9995)")
    parser.add_argument("--memory-size", type=int, default=50000, help="Replay buffer capacity (default: 50000)")
    parser.add_argument("--save-path", type=str, default="models/tic_tac_toe_dqn.keras", help="Model checkpoint path")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save metric plots")
    parser.add_argument("--log-interval", type=int, default=500, help="Logging episode interval (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()
    train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        save_path=args.save_path,
        results_dir=args.results_dir,
        log_interval=args.log_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
