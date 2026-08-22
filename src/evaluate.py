"""
Model Evaluation and Benchmarking Suite
Evaluates trained DQN agents against baseline opponents (Random and Strategic Heuristic),
compares pre-training vs. post-training performance, and saves structured CSV reports.
"""

import argparse
import os
import random
from typing import Any, Dict, List
import numpy as np

from src.agent import DQNAgent
from src.environment import TicTacToeEnvironment
from src.train import RandomOpponent, StrategicOpponent
from src.utils import format_table, save_evaluation_csv, set_seed


def evaluate_agent_against_opponent(
    agent: DQNAgent,
    opponent_name: str,
    opponent_instance: Any,
    num_games: int = 1000,
    agent_label: str = "Trained DQN",
) -> Dict[str, Any]:
    """
    Simulates games between an agent and a specified opponent in pure exploitation mode.

    Args:
        agent (DQNAgent): Agent being evaluated.
        opponent_name (str): Human-readable opponent identifier.
        opponent_instance (Any): Opponent decision logic class.
        num_games (int): Number of evaluation games to run.
        agent_label (str): Descriptive label for the evaluated agent.

    Returns:
        dict: Performance summary metrics.
    """
    env = TicTacToeEnvironment()
    wins = 0
    draws = 0
    losses = 0

    for game_idx in range(num_games):
        state = env.reset()

        # Alternate who plays first to guarantee fair evaluation
        if game_idx % 2 == 1:
            # Opponent starts
            opp_act = opponent_instance.get_action(env)
            state, _, done, _ = env.step(opp_act, player=-1)

        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # AI move (pure greedy policy, no exploration noise)
            action = agent.act(state, valid_actions, training=False)
            next_state, reward, done, info = env.step(action, player=1)

            if done:
                winner = info.get("winner")
                if winner == 1:
                    wins += 1
                elif winner == 0:
                    draws += 1
                else:
                    losses += 1
                break

            # Opponent move
            opp_act = opponent_instance.get_action(env)
            opp_next_state, _, opp_done, opp_info = env.step(opp_act, player=-1)

            if opp_done:
                winner = opp_info.get("winner")
                if winner == -1:
                    losses += 1
                elif winner == 0:
                    draws += 1
                else:
                    wins += 1
                done = True
            else:
                state = opp_next_state

    win_rate = (wins / num_games) * 100.0
    draw_rate = (draws / num_games) * 100.0
    loss_rate = (losses / num_games) * 100.0

    return {
        "Agent": agent_label,
        "Opponent": opponent_name,
        "Total Games": num_games,
        "Wins": wins,
        "Draws": draws,
        "Losses": losses,
        "Win Rate (%)": f"{win_rate:.2f}",
        "Draw Rate (%)": f"{draw_rate:.2f}",
        "Loss Rate (%)": f"{loss_rate:.2f}",
    }


def run_evaluation_suite(
    model_path: str = "models/tic_tac_toe_dqn.keras",
    num_games: int = 1000,
    output_csv: str = "results/evaluation_results.csv",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Executes a comprehensive evaluation comparing Untrained Agent vs. Trained DQN Agent.

    Args:
        model_path (str): Checkpoint file path for the trained model.
        num_games (int): Number of games per benchmark match.
        output_csv (str): File destination for results CSV.
        seed (int): Reproducibility seed.

    Returns:
        list of dict: Complete benchmark table records.
    """
    set_seed(seed)

    print("=" * 75)
    print("DEEP Q-NETWORK (DQN) BENCHMARK EVALUATION")
    print("=" * 75)
    print(f"Loading trained model checkpoint: '{model_path}'...")

    trained_agent = DQNAgent()
    trained_agent.load(model_path)

    # Instantiate untrained agent for baseline comparison
    untrained_agent = DQNAgent(epsilon=1.0)  # Purely random moves

    random_opp = RandomOpponent()
    strategic_opp = StrategicOpponent()

    evaluation_records: List[Dict[str, Any]] = []

    print(f"\n[1/4] Benchmarking Untrained Baseline vs. Random Opponent ({num_games:,} games)...")
    res_untrained_rand = evaluate_agent_against_opponent(
        untrained_agent, "Random Opponent", random_opp, num_games, agent_label="Untrained Baseline"
    )
    evaluation_records.append(res_untrained_rand)

    print(f"[2/4] Benchmarking Untrained Baseline vs. Strategic Opponent ({num_games:,} games)...")
    res_untrained_strat = evaluate_agent_against_opponent(
        untrained_agent, "Strategic Opponent", strategic_opp, num_games, agent_label="Untrained Baseline"
    )
    evaluation_records.append(res_untrained_strat)

    print(f"[3/4] Benchmarking Trained DQN Agent vs. Random Opponent ({num_games:,} games)...")
    res_trained_rand = evaluate_agent_against_opponent(
        trained_agent, "Random Opponent", random_opp, num_games, agent_label="Trained DQN Agent"
    )
    evaluation_records.append(res_trained_rand)

    print(f"[4/4] Benchmarking Trained DQN Agent vs. Strategic Opponent ({num_games:,} games)...")
    res_trained_strat = evaluate_agent_against_opponent(
        trained_agent, "Strategic Opponent", strategic_opp, num_games, agent_label="Trained DQN Agent"
    )
    evaluation_records.append(res_trained_strat)

    # Format table output
    headers = [
        "Agent",
        "Opponent",
        "Total Games",
        "Wins",
        "Draws",
        "Losses",
        "Win Rate (%)",
        "Draw Rate (%)",
        "Loss Rate (%)",
    ]
    rows = [[record[h] for h in headers] for record in evaluation_records]

    print("\n" + "=" * 75)
    print("EVALUATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(format_table(headers, rows))

    save_evaluation_csv(evaluation_records, filepath=output_csv)
    return evaluation_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Trained DQN Agent on Tic-Tac-Toe.")
    parser.add_argument("--model-path", type=str, default="models/tic_tac_toe_dqn.keras", help="Model file path")
    parser.add_argument("--games", type=int, default=1000, help="Number of evaluation games per opponent (default: 1000)")
    parser.add_argument("--output-csv", type=str, default="results/evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")

    args = parser.parse_args()
    run_evaluation_suite(
        model_path=args.model_path,
        num_games=args.games,
        output_csv=args.output_csv,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
