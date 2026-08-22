"""
Utility Functions for Deep Q-Learning Project
Includes random seed initialization, plotting functions for metrics,
results logging, and terminal table formatting.
"""

import csv
import os
import random
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds for Python random, NumPy, and TensorFlow.

    Args:
        seed (int): Random seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_training_results(history: Dict[str, List[Any]], save_dir: str = "results") -> None:
    """
    Generates and saves professional academic metric plots to disk.

    Plots generated:
        1. training_loss.png
        2. win_rate.png
        3. epsilon_decay.png
        4. reward_history.png

    Args:
        history (dict): Dictionary of recorded metric lists across training episodes.
        save_dir (str): Destination folder path.
    """
    os.makedirs(save_dir, exist_ok=True)
    episodes = history.get("episodes", list(range(1, len(history.get("win_rates", [])) + 1)))

    # Set clean plotting style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    })

    # 1. Training Loss
    if "losses" in history and history["losses"]:
        plt.figure(figsize=(8, 5))
        loss_steps = list(range(1, len(history["losses"]) + 1))
        plt.plot(loss_steps, history["losses"], color="#1f77b4", linewidth=1.5, label="Huber Loss")
        
        # Plot smoothed moving average if sufficient points
        if len(history["losses"]) >= 50:
            window = 50
            smooth_loss = np.convolve(history["losses"], np.ones(window) / window, mode="valid")
            plt.plot(range(window, len(history["losses"]) + 1), smooth_loss, color="#d62728", linewidth=2.0, label=f"Moving Avg ({window})")
        
        plt.title("DQN Training Loss Over Training Steps")
        plt.xlabel("Replay Update Step")
        plt.ylabel("Loss (Huber)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300)
        plt.close()

    # 2. Win / Draw / Loss Rates Progression
    if "win_rates" in history and history["win_rates"]:
        plt.figure(figsize=(8, 5))
        plt.plot(episodes, history["win_rates"], color="#2ca02c", linewidth=2.0, label="Win Rate (%)")
        plt.plot(episodes, history.get("draw_rates", []), color="#ff7f0e", linewidth=1.8, label="Draw Rate (%)")
        plt.plot(episodes, history.get("loss_rates", []), color="#d62728", linewidth=1.8, label="Loss Rate (%)")
        plt.title("Agent Performance Progression Over Training Episodes")
        plt.xlabel("Training Episode")
        plt.ylabel("Percentage (%)")
        plt.ylim(-2, 102)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="center right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "win_rate.png"), dpi=300)
        plt.close()

    # 3. Epsilon Decay Progression
    if "epsilons" in history and history["epsilons"]:
        plt.figure(figsize=(8, 5))
        plt.plot(episodes, history["epsilons"], color="#9467bd", linewidth=2.0, label="Exploration Rate (epsilon)")
        plt.title("Epsilon Decay Schedule (Exploration vs. Exploitation)")
        plt.xlabel("Training Episode")
        plt.ylabel("Epsilon")
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "epsilon_decay.png"), dpi=300)
        plt.close()

    # 4. Episode Rewards History
    if "rewards" in history and history["rewards"]:
        plt.figure(figsize=(8, 5))
        all_episodes = list(range(1, len(history["rewards"]) + 1))
        plt.plot(all_episodes, history["rewards"], color="#17becf", alpha=0.3, label="Raw Episode Reward")
        if len(history["rewards"]) >= 50:
            window = 50
            smooth_rewards = np.convolve(history["rewards"], np.ones(window) / window, mode="valid")
            plt.plot(all_episodes[window - 1:], smooth_rewards, color="#005580", linewidth=2.0, label=f"Rolling Average ({window})")
        plt.title("Cumulative Reward per Training Episode")
        plt.xlabel("Training Episode")
        plt.ylabel("Reward")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "reward_history.png"), dpi=300)
        plt.close()

    print(f"All metric visualization plots successfully saved to: {os.path.abspath(save_dir)}")


def save_evaluation_csv(results: List[Dict[str, Any]], filepath: str = "results/evaluation_results.csv") -> None:
    """
    Exports evaluation benchmark metrics to a structured CSV file.

    Args:
        results (list of dict): Evaluation metric records.
        filepath (str): Destination CSV filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Evaluation benchmark saved to: {os.path.abspath(filepath)}")


def format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Constructs a clean ASCII table for command-line presentation.
    """
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row))
        for row in rows
    ]
    return f"\n{header_line}\n{separator}\n" + "\n".join(row_lines) + "\n"
