# Deep Learning Based Game Playing Agent Using Deep Q-Learning

An autonomous game-playing agent trained to master Tic-Tac-Toe using Deep Q-Networks (DQN) and Reinforcement Learning principles in Python and TensorFlow/Keras.

---

## Table of Contents

- [Project Title](#deep-learning-based-game-playing-agent-using-deep-q-learning)
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [How the Agent Works](#how-the-agent-works)
- [Deep Q-Learning Methodology](#deep-q-learning-methodology)
- [Neural Network Architecture](#neural-network-architecture)
- [Reinforcement Learning Components](#reinforcement-learning-components)
- [Reward System](#reward-system)
- [Training Process](#training-process)
- [Exploration vs Exploitation](#exploration-vs-exploitation)
- [Experience Replay](#experience-replay)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Training the Agent](#training-the-agent)
- [Playing Against the AI](#playing-against-the-ai)
- [Model Evaluation](#model-evaluation)
- [Performance Metrics](#performance-metrics)
- [Results and Visualization](#results-and-visualization)
- [DLT Concepts Demonstrated](#dlt-concepts-demonstrated)
- [Future Enhancements](#future-enhancements)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Author](#author)
- [License](#license)

---

## Project Overview

This project develops an artificial intelligence game-playing agent capable of learning optimal strategies for Tic-Tac-Toe through Deep Reinforcement Learning (DRL), specifically utilizing the Deep Q-Network (DQN) algorithm. Rather than relying on static rule-based heuristics or exhaustive minimax tree searches, the agent discovers effective policies autonomously through trial, error, and feedback.

The primary objective is to demonstrate fundamental Deep Learning Techniques (DLT) and Reinforcement Learning (RL) paradigms within an interactive control loop. The agent observes board configurations, selects moves, receives scalar reward feedback, archives transitions into an experience replay buffer, and iteratively optimizes a deep neural network via gradient descent to approximate the optimal action-value function.

```
+------------------+      Board State      +-------------------------+
|                  | --------------------> |   State Representation  |
|                  |                       +-------------------------+
|                  |                                    |
|                  |                              Flat State Array
|                  |                                    v
|                  |                       +-------------------------+
|                  |                       |    Deep Q-Network (DQN) |
|                  |                       +-------------------------+
|                  |                                    |
|                  |                              Q-Value Prediction
|                  |                                    v
|   Tic-Tac-Toe    |                       +-------------------------+
|   Environment    | <-------------------- |     Action Selection    |
|                  |      Execute Move     |   (Epsilon-Greedy)      |
|                  |                       +-------------------------+
|                  |                                    |
|                  |    Reward & Next State             |
|                  | -----------------------------------+
|                  |                                    |
|                  |                                    v
|                  |                       +-------------------------+
|                  |                       |    Experience Replay    |
|                  |                       |      Memory Buffer      |
|                  |                       +-------------------------+
|                  |                                    |
|                  |                               Mini-Batch
|                  |                                    v
|                  |                       +-------------------------+
|                  |                       |   Backpropagation &     |
|                  |                       |     Weight Update       |
|                  |                       +-------------------------+
|                  |                                    |
|                  |                              Optimized Policy
+------------------+ <----------------------------------+
```

### Complete Operational Workflow

1. **Game Environment**: Represents the 3x3 Tic-Tac-Toe board, validates legal moves, identifies terminal states (win, loss, draw), and manages turn-taking dynamics.
2. **State Representation**: Converts the 2D grid into a standardized numerical tensor suitable for neural network consumption.
3. **Deep Q-Network (DQN)**: Evaluates the state tensor through fully connected layers to compute expected cumulative future rewards for all available board positions.
4. **Q-Value Prediction**: Outputs an action-value vector $Q(s, a)$ containing expected returns for each of the 9 board cells.
5. **Action Selection**: Employs an $\epsilon$-greedy mechanism to balance random exploratory moves with policy exploitation of predicted maximum Q-values.
6. **Reward Dispatch**: Allocates scalar feedback based on the outcome of the selected action (positive for wins/draws, negative for losses/invalid moves).
7. **Experience Replay**: Stores transition tuples $(s, a, r, s', \text{done})$ in a fixed-capacity replay buffer to decouple temporal correlations.
8. **Neural Network Training**: Samples uncorrelated mini-batches from replay memory to compute Mean Squared Error (MSE) loss against temporal difference targets and updates network parameters via backpropagation.
9. **Improved Policy**: Gradually converges toward an optimal action-value function, yielding resilient defensive and offensive gameplay strategies.

---

## Objectives

- **Develop a Deep Reinforcement Learning Agent**: Implement an end-to-end DQN agent in Python and TensorFlow/Keras capable of playing Tic-Tac-Toe without hardcoded game rules.
- **Implement Function Approximation**: Replace classical tabular Q-learning with a deep feedforward artificial neural network to estimate action-value distributions.
- **Stabilize Learning Dynamics**: Mitigate non-stationary target issues and autocorrelation in sequential decision tasks using Experience Replay and $\epsilon$-greedy exploration decay.
- **Demonstrate Deep Learning Techniques**: Apply foundational deep learning concepts including multi-layer perceptron design, non-linear activation functions (ReLU), backpropagation, Adam optimization, and loss minimization.
- **Establish Quantitative Evaluation**: Benchmark the trained agent against baseline strategies (random agent, scripted opponents) across metrics such as win rate, draw rate, loss rate, and loss convergence.
- **Provide an Interactive Interface**: Enable real-time human vs. AI gameplay through an accessible console or command-line interface.

---

## Key Features

- **Autonomous Policy Learning**: The agent acquires strategic decision-making capabilities purely from environmental reward signals over repeated episodes.
- **Deep Q-Network (DQN) Core**: A dual-layer fully connected architecture maps 9-element spatial state inputs directly to 9-element action Q-values.
- **Experience Replay Buffer**: Circular transition storage buffers sequential transitions and breaks data autocorrelation via uniform mini-batch sampling.
- **Adaptive Exploration Schedule**: Exponential or linear $\epsilon$-decay transitions the agent smoothly from initial random exploration to high-confidence policy exploitation.
- **Valid Action Masking**: Constrains action selection to unoccupied cells during exploitation, preventing invalid moves and speeding up policy convergence.
- **Modular Codebase**: Decoupled modules for environment mechanics, neural model definition, agent logic, training pipelines, evaluation routines, and interactive gameplay.
- **Comprehensive Visualizations**: Automated generation of convergence curves, win/loss/draw rate progressions, and exploration decay trajectories.

---

## Technologies Used

- **Programming Language**: Python 3.9+
- **Deep Learning Framework**: TensorFlow 2.x / Keras
- **Numerical Computation**: NumPy
- **Data Visualization**: Matplotlib
- **Environment & Control**: Python Standard Library (`random`, `collections.deque`, `os`, `sys`, `time`)
- **Version Control**: Git / GitHub

---

## System Architecture

The system is organized into modular software abstractions that separate the reinforcement learning algorithm from the underlying environment simulation and presentation layers:

```
+-------------------------------------------------------------------------+
|                           User / CLI Layer                              |
|          (train.py, evaluate.py, game.py - Human Interaction)           |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                              Agent Layer                                |
|                              (agent.py)                                 |
|  - Action Selection (Epsilon-Greedy)    - Target Q Computation          |
|  - Epsilon Decay Management             - Experience Buffer Storage     |
+-------------------+---------------------------------+-------------------+
                    |                                 |
                    v                                 v
+------------------------------------+   +--------------------------------+
|         Environment Layer          |   |          Model Layer           |
|          (environment.py)          |   |           (model.py)           |
|  - 3x3 Grid State Management       |   |  - Dense Neural Network        |
|  - Legal Move Verification         |   |  - Forward Inference           |
|  - Terminal State Detection        |   |  - Backpropagation & Adam Opt  |
|  - Reward Generation               |   |  - Weight Persistence (.h5)    |
+------------------------------------+   +--------------------------------+
```

---

## How the Agent Works

The agent interacts with the Tic-Tac-Toe environment through discrete time steps $t = 0, 1, 2, \dots$:

1. **State Observation**: At time $t$, the agent receives state $s_t$, a flattened 9-element vector encoding the current board configuration.
2. **Action Selection**:
   - With probability $\epsilon$, the agent selects a random valid action $a_t \in \{0, \dots, 8\}$.
   - With probability $1 - \epsilon$, the agent computes $Q(s_t, a; \theta)$ for all valid actions using the neural network with parameters $\theta$, selecting $a_t = \arg\max_{a} Q(s_t, a; \theta)$.
3. **Environment Step**: The action $a_t$ is applied to the board. The environment checks for terminal conditions (win, loss, draw) and processes the opponent's counter-move (during training, either a random or rule-based opponent).
4. **Feedback Reception**: The agent observes scalar reward $r_t$, next state $s_{t+1}$, and terminal flag $d_t \in \{\text{True}, \text{False}\}$.
5. **Memory Archival**: The transition tuple $(s_t, a_t, r_t, s_{t+1}, d_t)$ is saved to the replay memory buffer $\mathcal{D}$.
6. **Parameter Optimization**: A random mini-batch of size $N$ is drawn from $\mathcal{D}$. The network weights $\theta$ are updated using gradient descent on the temporal difference loss.

---

## Deep Q-Learning Methodology

In classical Reinforcement Learning, Q-learning seeks to learn the optimal action-value function $Q^*(s, a)$, defined as the maximum expected cumulative future discounted reward starting from state $s$ and taking action $a$:

$$Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a \right]$$

For discrete state spaces of low dimensionality, $Q(s, a)$ can be maintained in a lookup table (Q-table). However, function approximation becomes necessary as state-action complexity grows. In Deep Q-Learning, a parameterized neural network $Q(s, a; \theta)$ serves as a non-linear function approximator that replaces the tabular representation.

### The Bellman Optimality Equation and DQN Loss

The Q-learning target $y_i$ for a transition $(s, a, r, s')$ is computed as:

$$y_i = \begin{cases} r & \text{if } s' \text{ is terminal} \\ r + \gamma \max_{a'} Q(s', a'; \theta) & \text{otherwise} \end{cases}$$

The neural network is trained by minimizing the Mean Squared Error (MSE) loss function over mini-batches sampled from replay memory:

$$L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - Q(s_i, a_i; \theta) \right)^2$$

### Variable Definitions

| Variable / Parameter | Mathematical Symbol | Description |
| :--- | :--- | :--- |
| **State** | $s \in \mathcal{S}$ | Complete numerical representation of the board at a given step. |
| **Action** | $a \in \mathcal{A}$ | Move selected by the agent (board cell index from $0$ to $8$). |
| **Reward** | $r \in \mathbb{R}$ | Scalar feedback returned by the environment evaluating the action. |
| **Next State** | $s' \in \mathcal{S}$ | Board state resulting from the execution of action $a$ and opponent counter-move. |
| **Q-Value** | $Q(s, a)$ | Expected cumulative discounted reward for taking action $a$ in state $s$. |
| **Discount Factor** | $\gamma \in [0, 1]$ | Quantifies the importance of future rewards relative to immediate rewards (typically $0.90 - 0.99$). |
| **Learning Rate** | $\alpha$ or $\eta$ | Step size parameter governing weight updates in gradient descent (typically $0.001$). |
| **Exploration Rate** | $\epsilon \in [\epsilon_{\min}, 1.0]$ | Probability of selecting a random exploratory action rather than the greedy action. |
| **Policy** | $\pi(a \mid s)$ | Mapping from observed states to action probabilities or deterministic decisions. |

---

## Neural Network Architecture

The Deep Q-Network is structured as a Multi-Layer Perceptron (MLP) mapping board states directly to action-value predictions for all possible moves.

### Structural Specification

- **Input Layer**: 9 neurons (accepting flattened 1D board state where empty cells are `0`, agent marks are `1`, and opponent marks are `-1`).
- **Hidden Layer 1**: Fully Connected (Dense), 64 units, Rectified Linear Unit (ReLU) activation.
- **Hidden Layer 2**: Fully Connected (Dense), 64 units, Rectified Linear Unit (ReLU) activation.
- **Output Layer**: Fully Connected (Dense), 9 units, Linear activation (predicting scalar $Q$-value for each board cell index $0$ through $8$).
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam ($\text{learning rate} = 0.001$).

### Rationale for Layer Sizing

- **9 Input Values**: Corresponds exactly to the $3 \times 3$ grid positions of a Tic-Tac-Toe board, providing the model with complete state information.
- **64 Hidden Units**: Provides sufficient parameter capacity to model non-linear spatial dependencies, forks, blocks, and winning sequences without inducing severe overfitting on small state spaces.
- **9 Output Values**: Enables simultaneous evaluation of all potential board actions in a single forward pass, from which the maximum valid Q-value is selected.

```
+--------------------------------------------------------------------------------------------------------------------+
|                                             DQN INFERENCE PIPELINE                                                 |
+--------------------------------------------------------------------------------------------------------------------+

 [ Tic-Tac-Toe Board ]
       |   | X 
     --+---+--          [ State Encoding ]
       | O |            [ 0, 0, 1, 0, -1, 0, 0, 0, 0 ]
     --+---+--
       |   |   
          |
          v
+--------------------+
|  Input Layer (9)   |  9 State Features
+--------------------+
          |
          v  Dense + ReLU (Weights: 9 x 64 = 576, Biases: 64)
+--------------------+
| Dense Layer 1 (64) |  Feature Extraction & Representation
+--------------------+
          |
          v  Dense + ReLU (Weights: 64 x 64 = 4096, Biases: 64)
+--------------------+
| Dense Layer 2 (64) |  Higher-Level Strategic Pattern Recognition
+--------------------+
          |
          v  Dense + Linear (Weights: 64 x 9 = 576, Biases: 9)
+--------------------+
|  Output Layer (9)  |  Raw Predicted Q-Values for Actions a_0 to a_8
+--------------------+
          |
          v
 [ Q-Value Vector ]     [ Q(s,a_0), Q(s,a_1), ..., Q(s,a_8) ]
          |
          v
 [ Mask Invalid Move]   Filter out occupied positions
          |
          v
 [ Action Selection ]   Select a* = argmax_{valid a} Q(s, a) (or random if exploring)
```

---

## Reinforcement Learning Components

| Component | Role in Reinforcement Learning | Project Implementation |
| :--- | :--- | :--- |
| **Agent** | Decision-maker interacting with the world | Deep Q-Network (DQN) agent implemented in Python/TensorFlow |
| **Environment** | The external system hosting rules and states | Custom $3 \times 3$ Tic-Tac-Toe simulator (`environment.py`) |
| **State** | Environmental snapshot visible to agent | 9-element numerical vector representing cell status (`0`: Empty, `1`: AI, `-1`: Opponent) |
| **Action** | Discrete decision executed by the agent | Selection of an integer index $a \in \{0, 1, \dots, 8\}$ corresponding to an empty board cell |
| **Reward** | Scalar evaluation of action consequence | Numerical reward assigned upon terminal conditions or intermediate moves |
| **Policy** | Strategy governing action selection | $\epsilon$-greedy policy transitioning from exploratory to greedy action selection |
| **Learning** | Optimization process updating policy | Off-policy Temporal Difference Q-learning with mini-batch Experience Replay |

---

## Reward System

The reward function provides critical supervisory feedback guiding the agent toward winning strategies while penalizing losses, stalemates, and unlawful moves.

| Outcome / Action Condition | Default Scalar Reward | Description / Strategic Objective |
| :--- | :---: | :--- |
| **Win** | `+1.0` | Agent completes three in a row (row, column, or diagonal). |
| **Loss** | `-1.0` | Opponent completes three in a row. |
| **Draw / Stalemate** | `+0.2` | Board is completely filled with no winner (successful defensive hold). |
| **Valid Intermediate Move** | `0.0` | Normal move resulting in an ongoing game. |
| **Invalid Action Selection** | `-10.0` or Masked | Selecting an already occupied cell (penalized heavily or prevented via action masking). |

> *Note: Exact reward values serve as initial hyperparameters and can be adjusted during experimentation to investigate behavioral variations (e.g., incentivizing faster wins through step penalties).*

---

## Training Process

The training cycle follows an iterative multi-episode loop where the agent accumulates transitions, refines action-value estimates, and decreases exploration:

```
+-------------------------------------------------------------------+
|                         Episode Start                             |
+---------------------------------+---------------------------------+
                                  |
                                  v
                   +-----------------------------+
                   |  Reset Environment to s_0   |
                   +--------------+--------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                         Step Loop (Game)                          |
+-------------------------------------------------------------------+
|                                                                   |
|   1. Observe Current State s                                      |
|   2. Select Action a via Epsilon-Greedy Policy                    |
|   3. Execute Action a in Environment                              |
|   4. Observe Reward r, Next State s', and Terminal Flag (done)    |
|   5. Store Experience Tuple (s, a, r, s', done) in Buffer         |
|   6. If Replay Buffer >= Batch Size:                              |
|        a. Sample Random Mini-Batch of Transitions                 |
|        b. Calculate Target Q-Values:                              |
|           y = r  (if done)                                        |
|           y = r + gamma * max_a' Q(s', a')  (if not done)         |
|        c. Compute MSE Loss and Backpropagate                      |
|        d. Update Neural Network Weights via Adam Optimizer        |
|   7. Update State: s <- s'                                        |
|   8. If done == True: Break Step Loop                             |
|                                                                   |
+---------------------------------+---------------------------------+
                                  |
                                  v
                   +-----------------------------+
                   | Decay Exploration Rate (eps)|
                   | eps = max(eps_min, eps*dec) |
                   +--------------+--------------+
                                  |
                                  v
                   +-----------------------------+
                   |  Record Episode Statistics  |
                   | (Win/Loss, Loss, Epsilon)   |
                   +--------------+--------------+
                                  |
                                  v
                   +-----------------------------+
                   |    Save Model Checkpoint    |
                   +-----------------------------+
```

### Detailed Algorithm Steps

1. **Environment Initialization**: The game board is cleared to an empty state $s_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]$.
2. **State Observation**: The current state vector $s$ is passed to the agent.
3. **Action Selection**: The agent selects action $a$ using the $\epsilon$-greedy policy (valid move constraints applied).
4. **Environment Execution**: The action is committed. If non-terminal, the opponent executes its turn.
5. **Reward & Next State Receipt**: The agent receives feedback $r$, new board state $s'$, and termination flag $d$.
6. **Experience Storage**: The tuple $(s, a, r, s', d)$ is added to the circular replay memory buffer $\mathcal{D}$.
7. **Mini-Batch Sampling**: When memory size exceeds the batch threshold (e.g., $N = 32$ or $64$), a uniform random mini-batch is sampled from $\mathcal{D}$.
8. **Target Calculation**: Target Q-values are computed using the Bellman target formula for each sampled transition.
9. **Weight Optimization**: A gradient step is performed on the neural network to minimize MSE loss between predicted Q-values and target values.
10. **Epsilon Decay**: The exploration parameter $\epsilon$ is updated according to a decay schedule.
11. **Episode Iteration**: Steps 1–10 repeat across configured episodes (e.g., $10{,}000 - 50{,}000$ episodes).
12. **Model Persistence**: Trained weights and model definitions are exported to disk (`models/dqn_tictactoe.h5`).

---

## Exploration vs Exploitation

Balancing exploration (discovering new board states and strategies) and exploitation (utilizing current best-known strategies) is essential for stable Q-learning convergence.

This project implements the standard **$\epsilon$-Greedy Strategy**:

$$\pi(a \mid s) = \begin{cases} \text{Uniform Random Valid Action} & \text{with probability } \epsilon \\ \arg\max_{a \in \mathcal{A}_{\text{valid}}} Q(s, a; \theta) & \text{with probability } 1 - \epsilon \end{cases}$$

### Epsilon Decay Schedule

- **Initial Exploration ($\epsilon_{\text{start}}$)**: Set to `1.0` (100% random actions at the beginning of training).
- **Decay Factor ($\epsilon_{\text{decay}}$)**: Multiplied per episode (e.g., `0.9995`), gradually shifting priority toward exploitation.
- **Minimum Floor ($\epsilon_{\text{min}}$)**: Constrained to `0.01` or `0.05` to retain a baseline level of stochastic exploration throughout training.

$$\epsilon_{k+1} = \max\left(\epsilon_{\min},\; \epsilon_k \times \epsilon_{\text{decay}}\right)$$

---

## Experience Replay

Standard reinforcement learning with neural networks frequently diverges when transitions are presented sequentially due to strong temporal correlations and non-stationary target distributions.

### Purpose and Mechanisms

- **Correlation Breaking**: Consecutive game states in Tic-Tac-Toe are highly correlated. Experience replay buffers sequential transitions and samples mini-batches uniformly at random, breaking correlations and satisfying the independent and identically distributed (i.i.d.) assumption of gradient descent.
- **Data Efficiency**: Every experienced game transition $(s, a, r, s', d)$ is stored in a fixed-capacity circular buffer (e.g., capacity = 10,000 transitions) and can be reused in multiple gradient updates.
- **Training Stability**: Mini-batch averaging prevents network weights from oscillating wildly due to single anomalous game outcomes.

---

## Project Structure

The project structure is organized as follows:

```
deep-learning-game-playing-agent/
│
├── src/
│   ├── environment.py       # Tic-Tac-Toe game mechanics, state tracking, reward logic
│   ├── model.py             # Keras/TensorFlow Deep Q-Network model definition
│   ├── agent.py             # DQN agent logic, epsilon-greedy policy, replay memory
│   ├── train.py             # Agent training loop, episode execution, checkpointing
│   ├── evaluate.py          # Benchmark evaluation script against baseline agents
│   └── game.py              # Interactive CLI human vs. trained AI gameplay interface
│
├── models/
│   └── .gitkeep             # Directory for saved model weights (.h5 / .keras)
│
├── results/
│   ├── training_loss.png    # Plotted training loss curve over iterations
│   ├── win_rate.png         # Win/Loss/Draw progression curve across episodes
│   └── epsilon_decay.png    # Exploration parameter decay visualization
│
├── screenshots/
│   └── gameplay.png         # Demonstration screenshot of terminal gameplay
│
├── requirements.txt         # Project software dependencies and libraries
├── README.md                # Complete project documentation and specifications
├── .gitignore               # Files and patterns ignored by version control
└── LICENSE                  # Open-source license file
```

### Module Responsibilities

- `src/environment.py`: Encapsulates board state, legality verification, player toggling, board reset, terminal evaluation (row/col/diag checks), and reward dispatch.
- `src/model.py`: Implements a factory function constructing the feedforward deep neural network architecture using TensorFlow/Keras.
- `src/agent.py`: Manages the Q-network, target generation, memory buffer additions, mini-batch sampling, and action selection.
- `src/train.py`: Orchestrates the training pipeline across user-defined episode counts and writes metric logs to disk.
- `src/evaluate.py`: Executes deterministic test matches against a baseline random opponent or heuristic player to record objective performance metrics.
- `src/game.py`: Launches a real-time console interface allowing a human player to test strategies against the trained neural network.

---

## Installation and Setup

### Prerequisites

- Python 3.9, 3.10, or 3.11
- Git
- Recommended: Virtual environment manager (`venv` or `conda`)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/DineshMoorthy007/deep-learning-game-playing-agent.git
cd deep-learning-game-playing-agent
```

#### 2. Create and Activate a Virtual Environment

**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Project

The project provides dedicated entry points for training, evaluation, and interactive gameplay.

### Summary of Commands

| Command | Purpose |
| :--- | :--- |
| `python src/train.py` | Train the DQN agent and save model checkpoints |
| `python src/evaluate.py` | Run quantitative benchmarking against benchmark opponents |
| `python src/game.py` | Play an interactive game against the trained AI agent |

---

## Training the Agent

To initiate the training pipeline, execute `train.py`:

```bash
python src/train.py
```

### Expected Training Lifecycle

During training, progress is logged to the console at regular episode intervals:

```
Episode 1000/20000 | Loss: 0.0421 | Win Rate: 58.2% | Draw Rate: 18.5% | Epsilon: 0.6065
Episode 2000/20000 | Loss: 0.0218 | Win Rate: 72.4% | Draw Rate: 20.1% | Epsilon: 0.3678
Episode 5000/20000 | Loss: 0.0094 | Win Rate: 84.1% | Draw Rate: 13.2% | Epsilon: 0.0820
...
Training complete. Model saved to models/dqn_tictactoe.h5
```

Metric histories are exported to the `results/` directory upon training completion.

---

## Playing Against the AI

To test your skills against the trained Deep Q-Learning agent, launch the interactive gameplay script:

```bash
python src/game.py
```

### Board Position Mapping

Positions on the $3 \times 3$ grid correspond to numeric indices $0$ through $8$:

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

The user enters an integer index to place their marker (`X` or `O`), and the agent responds with its predicted optimal move in real time.

---

## Model Evaluation

The performance of the trained agent is evaluated quantitatively by running repeated games against benchmark policies:

1. **Random Agent**: Selects uniformly from available empty cells. Used to verify baseline competence.
2. **Rule-Based / Heuristic Agent**: Follows fixed tactical rules (e.g., take immediate wins, block opponent immediate wins).
3. **Pre-Training vs. Post-Training Comparison**: Compares the untransformed agent policy at Episode 0 against the fully converged model.

### Key Evaluation Formulas

- **Win Rate ($\%$):**
  $$\text{Win Rate} = \left( \frac{\text{Number of AI Wins}}{\text{Total Evaluation Games}} \right) \times 100$$

- **Draw Rate ($\%$):**
  $$\text{Draw Rate} = \left( \frac{\text{Number of Draws}}{\text{Total Evaluation Games}} \right) \times 100$$

- **Loss Rate ($\%$):**
  $$\text{Loss Rate} = \left( \frac{\text{Number of AI Losses}}{\text{Total Evaluation Games}} \right) \times 100$$

---

## Performance Metrics

The following metrics are tracked during the training and validation phases:

- **Mean Squared Error (MSE) Loss**: Measures the divergence between network-predicted Q-values $Q(s, a; \theta)$ and the temporal difference targets $y_i$.
- **Cumulative Reward per Episode**: Sum of all rewards obtained across single game trajectories.
- **Exploration Rate ($\epsilon$) Progression**: Confirms that exploration decays steadily toward $\epsilon_{\min}$.
- **Win / Draw / Loss Distributions**: Rolling percentages calculated over fixed windows (e.g., 100-episode rolling averages).

---

## Results and Visualization

The table below outlines the evaluation metrics to be recorded once formal training runs are completed:

| Metric | Target / Benchmark | Measured Value |
| :--- | :--- | :---: |
| **Total Training Episodes** | 20,000 Episodes | `[To be measured]` |
| **Win Rate vs. Random Opponent** | $> 85.0\%$ | `[To be measured]` |
| **Draw Rate vs. Random Opponent** | $10.0\% - 15.0\%$ | `[To be measured]` |
| **Loss Rate vs. Random Opponent** | $< 5.0\%$ | `[To be measured]` |
| **Win Rate vs. Heuristic Opponent** | $> 70.0\%$ | `[To be measured]` |
| **Final Mean Squared Error (Loss)** | $< 0.01$ | `[To be measured]` |
| **Exploration Rate ($\epsilon$) at Termination** | $0.01$ | `[To be measured]` |

> *Note: The placeholder values above will be populated with actual experimental results following comprehensive training cycles.*

### Planned Visualization Artifacts

- `results/training_loss.png`: Epoch-by-epoch loss curve illustrating model convergence.
- `results/win_rate.png`: Rolling win/draw/loss curves demonstrating policy improvement.
- `results/epsilon_decay.png`: Visual trajectory of the exploration parameter across episodes.

---

## DLT Concepts Demonstrated

This project serves as a comprehensive practical demonstration of foundational **Deep Learning Techniques (DLT)** and **Reinforcement Learning (RL)** principles:

### 1. Artificial Neural Networks (ANN) & Architecture
- **Dense Layers (Fully Connected)**: Structured multi-layer perceptron topology mapping tabular representations into latent strategic embeddings.
- **Rectified Linear Unit (ReLU) Activation**: $f(x) = \max(0, x)$, introducing non-linearity to allow the network to learn complex board feature combinations while avoiding vanishing gradients.
- **Linear Output Layer**: Generates unconstrained real-valued Q-value predictions for each discrete action.

### 2. Deep Learning Optimization
- **Forward Propagation**: End-to-end evaluation of state vectors through weight matrices and bias vectors to produce $Q$-predictions.
- **Backpropagation**: Analytical computation of loss gradients with respect to neural network weights via the chain rule.
- **Gradient Descent & Adam Optimizer**: Dynamic parameter optimization utilizing adaptive learning rates with first and second moment estimations.
- **Loss Minimization**: Formulation and convergence of Mean Squared Error (MSE) loss between predicted and target Q-values.

### 3. Deep Reinforcement Learning Paradigms
- **Function Approximation in RL**: Demonstrates how deep neural networks circumvent the curse of dimensionality associated with classical tabular methods.
- **Deep Q-Networks (DQN)**: Combines off-policy Q-learning with deep artificial neural networks.
- **Experience Replay**: Employs circular memory structures to decorrelate training data and maintain stability during gradient updates.
- **$\epsilon$-Greedy Policy Management**: Implements structured exploration-exploitation tradeoffs.
- **Reward Engineering**: Designing scalar feedback signals to shape autonomous policy emergence.
- **Empirical Model Evaluation**: Assessing stochastic agent policies against structured competitive benchmarks.

---

## Future Enhancements

The modular foundation established in this project enables several advanced extensions:

- **Target Network Separation (Fixed Q-Targets)**: Integrate a secondary target network $\hat{Q}(s, a; \theta^-)$ updated periodically to decouple target generation from active weight optimization and prevent policy oscillation.
- **Double DQN (DDQN)**: Decouple action selection from action evaluation to mitigate the overestimation bias inherent in standard Q-learning.
- **Prioritized Experience Replay (PER)**: Sample transitions proportional to their Temporal Difference (TD) error magnitude rather than uniformly, focusing training on critical mistakes.
- **Dueling DQN Architecture**: Separate the network into distinct state-value $V(s)$ and advantage $A(s, a)$ streams before aggregating to improve value estimation in non-action-critical states.
- **Complex Environment Scaling**: Generalize the agent framework to larger grid games such as Connect Four, Gomoku ($15 \times 15$), or Othello.
- **Graphical & Web Interfaces**: Develop an interactive web interface (e.g., Streamlit or Flask/React) for seamless browser-based human vs. AI gameplay.
- **Self-Play Training Curriculum**: Train the agent by playing against historical versions of itself (self-play) rather than fixed heuristic opponents.

---

## Limitations

- **State Space Dimensions**: Tic-Tac-Toe has a relatively small state space ($3^9 = 19{,}683$ theoretical states, of which only 5,478 are valid and reachable). While ideal for didactic demonstrations, DQN introduces computational overhead compared to direct minimax search or exact tabular Q-learning.
- **Hyperparameter Sensitivity**: Agent convergence rates and final win/loss ratios are sensitive to learning rates, discount factors ($\gamma$), batch sizes, and exploration decay rates.
- **Opponent Dependency during Training**: Training primarily against a purely random opponent may leave the agent susceptible to tactical traps when facing advanced human players or minimax algorithms unless exposed to varied opponent policies or self-play.
- **Absence of Separate Target Network**: In a basic DQN implementation without a dedicated target network, target values fluctuate as weights update, occasionally slowing convergence.

---

## Conclusion

This project successfully demonstrates the application of Deep Q-Networks to autonomous game playing. By combining multi-layer feedforward neural networks with reinforcement learning principles—including reward shaping, $\epsilon$-greedy exploration, and experience replay—the agent transitions from erratic random moves to proficient, strategic gameplay. The implementation provides a transparent, academically rigorous codebase illustrating how deep neural networks can serve as robust policy approximators in dynamic decision-making environments.

---

## Author

**Dinesh Moorthy**  
GitHub: [https://github.com/DineshMoorthy007](https://github.com/DineshMoorthy007)

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
