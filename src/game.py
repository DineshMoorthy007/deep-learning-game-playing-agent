"""
Interactive Gameplay Interface (Pygame GUI & CLI Mode)
Provides a high-end, modern dark-mode graphical user interface with real-time
Deep Q-Network Q-value visualization, match statistics, and interactive controls,
along with an automated CLI fallback mode.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# Suppress Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from src.agent import DQNAgent
from src.environment import TicTacToeEnvironment


# =============================================================================
# CLI INTERACTIVE GAMEPLAY (FALLBACK / HEADLESS)
# =============================================================================

def run_cli_game(model_path: str = "models/tic_tac_toe_dqn.keras") -> None:
    """
    Runs an interactive terminal-based Tic-Tac-Toe match against the trained DQN AI.
    """
    print("=" * 60)
    print("TIC-TAC-TOE: HUMAN VS. DEEP Q-NETWORK AI (CLI MODE)")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"ERROR: Model checkpoint '{model_path}' not found!")
        print("Please train the agent first using: python -m src.train")
        return

    agent = DQNAgent()
    agent.load(model_path)

    human_wins = 0
    ai_wins = 0
    draws = 0

    while True:
        env = TicTacToeEnvironment()
        print("\nChoose your side:")
        print("  1. Play as X (You go first)")
        print("  2. Play as O (AI goes first)")
        print("  Q. Quit to terminal")
        choice = input("Enter choice (1/2/Q): ").strip().upper()

        if choice == "Q":
            print("Thanks for playing!")
            break

        human_player = 1 if choice == "1" else -1
        ai_player = -human_player

        print(f"\nYou are playing as {'X' if human_player == 1 else 'O'}.")
        print("Board cell indices:")
        print(" 0 | 1 | 2 \n---+---+---\n 3 | 4 | 5 \n---+---+---\n 6 | 7 | 8 \n")

        current_player = 1  # 1 always moves first (X)
        done = False

        while not done:
            print(env.render_cli())

            if current_player == human_player:
                # Human move
                valid_actions = env.get_valid_actions()
                action = None
                while action is None:
                    try:
                        user_input = input(f"Enter move {valid_actions}: ").strip()
                        val = int(user_input)
                        if val in valid_actions:
                            action = val
                        else:
                            print(f"Invalid move. Choose from {valid_actions}.")
                    except ValueError:
                        print("Please enter a valid integer cell index (0-8).")

                _, _, done, info = env.step(action, player=human_player)
            else:
                # AI move
                print("AI is calculating optimal Q-values...")
                state = env.get_state()
                # If AI is player -1, state must be perspective-aligned
                ai_state = state if ai_player == 1 else -state
                valid_actions = env.get_valid_actions()
                
                # Show AI Q-values for educational insight
                q_vals = agent.get_q_values(ai_state)
                print("AI Predicted Q-Values for valid moves:")
                for va in valid_actions:
                    print(f"  Cell {va}: Q = {q_vals[va]:.4f}")

                action = agent.act(ai_state, valid_actions, training=False)
                print(f"AI chooses cell: {action}")
                _, _, done, info = env.step(action, player=ai_player)

            if done:
                print(env.render_cli())
                winner = info.get("winner")
                if winner == human_player:
                    print("CONGRATULATIONS! You won the game!")
                    human_wins += 1
                elif winner == ai_player:
                    print("GAME OVER! Deep Q-Network AI wins!")
                    ai_wins += 1
                else:
                    print("GAME OVER! It's a draw!")
                    draws += 1

                print(f"Scoreboard -> Human: {human_wins} | AI: {ai_wins} | Draws: {draws}")
                break

            current_player = -current_player


# =============================================================================
# MODERN PYGAME GRAPHICAL USER INTERFACE
# =============================================================================

def run_gui_game(model_path: str = "models/tic_tac_toe_dqn.keras") -> None:
    """
    Launches an advanced, dark-mode Pygame GUI with live Q-value visualization.
    """
    try:
        import pygame
    except ImportError:
        print("Pygame is not installed. Falling back to CLI mode.")
        run_cli_game(model_path)
        return

    # Check for model checkpoint
    if not os.path.exists(model_path):
        print(f"\n[!] Model checkpoint '{model_path}' not found.")
        print("Please train the model first by running: python -m src.train\n")
        return

    agent = DQNAgent()
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    pygame.init()
    pygame.font.init()

    # Window Dimensions & Layout
    WINDOW_WIDTH = 960
    WINDOW_HEIGHT = 640
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Deep Learning Game Playing Agent | Deep Q-Network Tic-Tac-Toe")
    clock = pygame.time.Clock()

    # Modern Dark Aesthetic Palette
    COLOR_BG = (24, 24, 37)          # Sleek Charcoal / Deep Slate
    COLOR_CARD = (33, 33, 52)        # Elevated Surface Card
    COLOR_BORDER = (49, 50, 68)      # Border Outline
    COLOR_CELL = (40, 41, 64)        # Board Tile Base
    COLOR_CELL_HOVER = (55, 57, 86)  # Tile Hover State
    COLOR_TEXT = (205, 214, 244)     # Primary Off-white Text
    COLOR_TEXT_DIM = (147, 153, 178) # Dim Subtext
    COLOR_CYAN = (0, 229, 255)       # Player X Neon Cyan
    COLOR_CORAL = (255, 82, 82)      # AI O Coral Red
    COLOR_GOLD = (255, 209, 102)     # Win Highlight Gold
    COLOR_GREEN = (46, 204, 113)     # Positive Q Bar Green
    COLOR_RED = (231, 76, 60)        # Negative Q Bar Red
    COLOR_BTN = (60, 63, 94)         # Button Base
    COLOR_BTN_HOVER = (80, 84, 122)  # Button Hover

    # Typography
    font_large = pygame.font.SysPygameFont if hasattr(pygame.font, "SysPygameFont") else pygame.font.SysFont
    try:
        font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
        font_board = pygame.font.SysFont("Arial", 54, bold=True)
        font_hud = pygame.font.SysFont("Segoe UI", 16, bold=True)
        font_sub = pygame.font.SysFont("Segoe UI", 14)
        font_qval = pygame.font.SysFont("Consolas", 13, bold=True)
    except Exception:
        font_title = pygame.font.Font(None, 24)
        font_board = pygame.font.Font(None, 64)
        font_hud = pygame.font.Font(None, 18)
        font_sub = pygame.font.Font(None, 16)
        font_qval = pygame.font.Font(None, 14)

    # Game State Variables
    env = TicTacToeEnvironment()
    human_symbol = 1  # 1 = X (First), -1 = O (Second)
    ai_symbol = -human_symbol
    human_score = 0
    ai_score = 0
    draw_score = 0
    game_over = False
    game_result_msg = ""
    winning_line: Optional[Tuple[int, int, int]] = None
    ai_thinking_delay = 0

    # Board tile geometry (Left Panel)
    BOARD_ORIGIN_X = 50
    BOARD_ORIGIN_Y = 110
    TILE_SIZE = 130
    TILE_GAP = 12

    # Cached AI Q-values for real-time visualization
    latest_q_values = np.zeros(9, dtype=np.float32)

    def compute_ai_q_values() -> None:
        nonlocal latest_q_values
        state = env.get_state()
        ai_state = state if ai_symbol == 1 else -state
        latest_q_values = agent.get_q_values(ai_state)

    def reset_game() -> None:
        nonlocal game_over, game_result_msg, winning_line, ai_thinking_delay
        env.reset()
        game_over = False
        game_result_msg = ""
        winning_line = None
        compute_ai_q_values()

        # If AI is playing as X (symbol = 1), trigger initial AI move
        if ai_symbol == 1:
            ai_thinking_delay = 15  # Frames delay for smooth feel

    compute_ai_q_values()

    # Main Interactive Loop
    running = True
    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        # -----------------------------------------------------------------
        # EVENT HANDLING
        # -----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Check Button: New Game (Reset)
                if 540 <= mx <= 690 and 550 <= my <= 595:
                    reset_game()

                # Check Button: Toggle Side (Play as X or O)
                elif 710 <= mx <= 890 and 550 <= my <= 595:
                    human_symbol = -human_symbol
                    ai_symbol = -human_symbol
                    reset_game()

                # Check Board Tile Clicks (Human Move)
                elif not game_over and ai_thinking_delay == 0:
                    for i in range(9):
                        row, col = divmod(i, 3)
                        tx = BOARD_ORIGIN_X + col * (TILE_SIZE + TILE_GAP)
                        ty = BOARD_ORIGIN_Y + row * (TILE_SIZE + TILE_GAP)
                        tile_rect = pygame.Rect(tx, ty, TILE_SIZE, TILE_SIZE)

                        if tile_rect.collidepoint(mx, my) and env.is_valid_action(i):
                            # Execute human move
                            _, _, done, info = env.step(i, player=human_symbol)
                            compute_ai_q_values()

                            if done:
                                game_over = True
                                winner = info.get("winner")
                                if winner == human_symbol:
                                    game_result_msg = "VICTORY! You Won!"
                                    human_score += 1
                                elif winner == 0:
                                    game_result_msg = "DRAW! Well Played!"
                                    draw_score += 1
                                # Detect winning combo for highlight
                                for combo in TicTacToeEnvironment.WIN_COMBINATIONS:
                                    if abs(sum(env.board[c] for c in combo)) == 3:
                                        winning_line = combo
                                        break
                            else:
                                # Schedule AI response
                                ai_thinking_delay = 12

        # -----------------------------------------------------------------
        # AI AUTOMATIC MOVE STEP
        # -----------------------------------------------------------------
        if not game_over and ai_thinking_delay > 0:
            ai_thinking_delay -= 1
            if ai_thinking_delay == 0:
                valid_actions = env.get_valid_actions()
                if valid_actions:
                    state = env.get_state()
                    ai_state = state if ai_symbol == 1 else -state
                    best_action = agent.act(ai_state, valid_actions, training=False)
                    _, _, done, info = env.step(best_action, player=ai_symbol)
                    compute_ai_q_values()

                    if done:
                        game_over = True
                        winner = info.get("winner")
                        if winner == ai_symbol:
                            game_result_msg = "AI WINS! DQN Optimal Play"
                            ai_score += 1
                        elif winner == 0:
                            game_result_msg = "DRAW! Stalemate Reached"
                            draw_score += 1
                        for combo in TicTacToeEnvironment.WIN_COMBINATIONS:
                            if abs(sum(env.board[c] for c in combo)) == 3:
                                winning_line = combo
                                break

        # -----------------------------------------------------------------
        # RENDERING PIPELINE
        # -----------------------------------------------------------------
        screen.fill(COLOR_BG)

        # Header Title
        title_surf = font_title.render("DEEP LEARNING GAME PLAYING AGENT (DQN)", True, COLOR_TEXT)
        sub_surf = font_sub.render("Deep Q-Network Autonomous Policy Demonstration", True, COLOR_TEXT_DIM)
        screen.blit(title_surf, (50, 30))
        screen.blit(sub_surf, (50, 60))

        # 1. Left Panel: 3x3 Interactive Board
        for i in range(9):
            row, col = divmod(i, 3)
            tx = BOARD_ORIGIN_X + col * (TILE_SIZE + TILE_GAP)
            ty = BOARD_ORIGIN_Y + row * (TILE_SIZE + TILE_GAP)
            tile_rect = pygame.Rect(tx, ty, TILE_SIZE, TILE_SIZE)

            # Hover highlight
            is_hover = tile_rect.collidepoint(mouse_pos) and env.is_valid_action(i) and not game_over
            base_color = COLOR_CELL_HOVER if is_hover else COLOR_CELL
            if winning_line and i in winning_line:
                base_color = (60, 55, 30)  # Subtle gold tint

            pygame.draw.rect(screen, base_color, tile_rect, border_radius=12)
            pygame.draw.rect(screen, COLOR_BORDER, tile_rect, width=2, border_radius=12)

            # Render Mark (X or O)
            val = env.board[i]
            if val == 1:
                # Mark 'X'
                x_color = COLOR_GOLD if (winning_line and i in winning_line) else COLOR_CYAN
                txt = font_board.render("X", True, x_color)
                screen.blit(txt, txt.get_rect(center=tile_rect.center))
            elif val == -1:
                # Mark 'O'
                o_color = COLOR_GOLD if (winning_line and i in winning_line) else COLOR_CORAL
                txt = font_board.render("O", True, o_color)
                screen.blit(txt, txt.get_rect(center=tile_rect.center))

        # Status Message Banner under the board
        status_box = pygame.Rect(50, 545, 414, 50)
        pygame.draw.rect(screen, COLOR_CARD, status_box, border_radius=10)
        pygame.draw.rect(screen, COLOR_BORDER, status_box, width=1, border_radius=10)

        if game_over:
            status_text = game_result_msg
            status_color = COLOR_GOLD if "VICTORY" in status_text or "DRAW" in status_text else COLOR_CORAL
        elif ai_thinking_delay > 0:
            status_text = "AI Thinking & Evaluating Q-Values..."
            status_color = COLOR_CORAL
        else:
            turn_sym = "X" if (human_symbol == 1 and env.board.sum() == 0) or (env.board.sum() == (1 if human_symbol == -1 else 0)) else "O"
            status_text = f"Your Turn ({'X' if human_symbol == 1 else 'O'}) - Select a cell"
            status_color = COLOR_CYAN

        status_render = font_hud.render(status_text, True, status_color)
        screen.blit(status_render, status_render.get_rect(center=status_box.center))

        # 2. Right Panel: Real-Time Deep Learning Insights & Scoreboard
        panel_rect = pygame.Rect(510, 110, 400, 415)
        pygame.draw.rect(screen, COLOR_CARD, panel_rect, border_radius=14)
        pygame.draw.rect(screen, COLOR_BORDER, panel_rect, width=1, border_radius=14)

        # Scoreboard Section
        score_title = font_hud.render("MATCH SCOREBOARD", True, COLOR_TEXT)
        screen.blit(score_title, (530, 125))

        score_details = (
            f"You ({'X' if human_symbol == 1 else 'O'}): {human_score}   |   "
            f"AI ({'O' if human_symbol == 1 else 'X'}): {ai_score}   |   "
            f"Draws: {draw_score}"
        )
        score_render = font_sub.render(score_details, True, COLOR_TEXT_DIM)
        screen.blit(score_render, (530, 150))

        pygame.draw.line(screen, COLOR_BORDER, (530, 180), (890, 180), width=1)

        # Real-time Q-Value Visualization
        q_title = font_hud.render("REAL-TIME DQN Q-VALUE OUTPUTS", True, COLOR_TEXT)
        screen.blit(q_title, (530, 195))
        q_expl = font_sub.render("Expected cumulative reward Q(s, a) predicted per cell:", True, COLOR_TEXT_DIM)
        screen.blit(q_expl, (530, 218))

        # Visual Q-Value Bar Chart (9 Cells)
        valid_acts = env.get_valid_actions()
        max_q_act = max(valid_acts, key=lambda a: latest_q_values[a]) if valid_acts else -1

        for i in range(9):
            row_y = 250 + i * 28
            q_val = latest_q_values[i]
            is_valid = i in valid_acts
            is_best = (i == max_q_act) and is_valid

            # Cell label
            label_col = COLOR_TEXT if is_valid else (90, 95, 120)
            cell_lbl = font_qval.render(f"Cell {i}:", True, label_col)
            screen.blit(cell_lbl, (530, row_y))

            # Numerical Q-value text
            q_str = f"{q_val:+6.3f}" if is_valid else "OCCUPIED"
            q_txt_col = COLOR_CYAN if is_best else (COLOR_TEXT if is_valid else (90, 95, 120))
            q_render = font_qval.render(q_str, True, q_txt_col)
            screen.blit(q_render, (600, row_y))

            # Bar representation
            bar_bg = pygame.Rect(690, row_y + 4, 180, 12)
            pygame.draw.rect(screen, (25, 26, 40), bar_bg, border_radius=4)

            if is_valid:
                # Normalize Q roughly between -1.0 and +1.0 for bar width
                norm_q = max(-1.0, min(1.0, float(q_val)))
                bar_len = int(abs(norm_q) * 85)
                mid_x = 690 + 90
                bar_color = COLOR_GREEN if norm_q >= 0 else COLOR_RED
                if is_best:
                    bar_color = COLOR_GOLD

                if norm_q >= 0:
                    fill_rect = pygame.Rect(mid_x, row_y + 4, bar_len, 12)
                else:
                    fill_rect = pygame.Rect(mid_x - bar_len, row_y + 4, bar_len, 12)

                pygame.draw.rect(screen, bar_color, fill_rect, border_radius=3)

        # 3. Interactive Buttons (Bottom Right)
        # New Game Button
        btn_new_rect = pygame.Rect(540, 550, 150, 45)
        is_btn_new_hover = btn_new_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, COLOR_BTN_HOVER if is_btn_new_hover else COLOR_BTN, btn_new_rect, border_radius=8)
        new_txt = font_hud.render("New Game", True, COLOR_TEXT)
        screen.blit(new_txt, new_txt.get_rect(center=btn_new_rect.center))

        # Side Switch Button
        btn_side_rect = pygame.Rect(710, 550, 180, 45)
        is_btn_side_hover = btn_side_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, COLOR_BTN_HOVER if is_btn_side_hover else COLOR_BTN, btn_side_rect, border_radius=8)
        side_txt = font_hud.render(f"Play as {'O' if human_symbol == 1 else 'X'}", True, COLOR_TEXT)
        screen.blit(side_txt, side_txt.get_rect(center=btn_side_rect.center))

        pygame.display.flip()

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe against Deep Q-Network AI.")
    parser.add_argument("--model-path", type=str, default="models/tic_tac_toe_dqn.keras", help="Model weights path")
    parser.add_argument("--cli", action="store_true", help="Launch interactive command-line interface instead of GUI")
    args = parser.parse_args()

    if args.cli:
        run_cli_game(model_path=args.model_path)
    else:
        run_gui_game(model_path=args.model_path)


if __name__ == "__main__":
    main()
