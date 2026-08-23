"""Interactive viewer for AlphaZero self-play sim logs.

Renders each move of a saved simulation:
  - the Connect 4 board (yellow = player 1, red = player -1)
  - the MCTS policy as a per-column bar chart
  - the value target and whose turn it is, in the corner

Navigate with the LEFT / RIGHT arrow keys.

Usage:
    python viz/visualize.py                 # latest sim_logs/sim_<n>.json
    python viz/visualize.py sim_logs/sim_3.json
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROWS, COLS = 6, 7
BOARD_BLUE = "#1f4fd8"
YELLOW = "#f4d03f"
RED = "#e74c3c"
EMPTY = "#0b2a8a"  # slot hole, darker than the board


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_log(log_dir):
    """Return the sim_<n>.json with the highest n, or None."""
    if not os.path.isdir(log_dir):
        return None
    numbered = [
        (int(m.group(1)), f)
        for f in os.listdir(log_dir)
        if (m := re.fullmatch(r"sim_(\d+)\.json", f))
    ]
    if not numbered:
        return None
    return os.path.join(log_dir, max(numbered)[1])


def load_frames(path):
    with open(path) as f:
        return json.load(f)["frames"]


def player_to_move(board):
    """Player 1 (yellow) moves first; parity of placed pieces gives the turn."""
    placed = sum(1 for row in board for c in row if c != 0)
    return 1 if placed % 2 == 0 else -1


class Viewer:
    def __init__(self, frames, title):
        self.frames = frames
        self.idx = 0
        self.title = title

        self.fig, (self.ax_board, self.ax_policy) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.2, 1]}
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    def on_key(self, event):
        if event.key == "right":
            self.idx = min(self.idx + 1, len(self.frames) - 1)
        elif event.key == "left":
            self.idx = max(self.idx - 1, 0)
        else:
            return
        self.draw()

    def draw_board(self, board):
        ax = self.ax_board
        ax.clear()
        # blue board backing
        ax.add_patch(Rectangle((-0.5, -0.5), COLS, ROWS, color=BOARD_BLUE, zorder=0))
        for r in range(ROWS):
            for c in range(COLS):
                # tensor row 0 is the top row; flip so it draws upright
                y = ROWS - 1 - r
                val = board[r][c]
                color = YELLOW if val == 1 else RED if val == -1 else EMPTY
                ax.add_patch(Circle((c, y), 0.38, color=color, zorder=1))
        ax.set_xlim(-0.5, COLS - 0.5)
        ax.set_ylim(-0.5, ROWS - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(COLS))
        ax.set_yticks([])
        ax.set_xlabel("column")

    def draw_policy(self, policy, to_move):
        ax = self.ax_policy
        ax.clear()
        bar_color = YELLOW if to_move == 1 else RED
        ax.bar(range(COLS), policy, color=bar_color, edgecolor="black")
        ax.set_xticks(range(COLS))
        ax.set_ylim(0, 1)
        ax.set_xlabel("column")
        ax.set_ylabel("MCTS policy (visit share)")
        for c, p in enumerate(policy):
            if p > 0:
                ax.text(c, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)

    def draw(self):
        frame = self.frames[self.idx]
        board = frame["board"]
        to_move = player_to_move(board)
        who = "Yellow (1)" if to_move == 1 else "Red (-1)"

        self.draw_board(board)
        self.draw_policy(frame["policy"], to_move)

        value = frame["value"]
        value_str = "n/a" if value is None else f"{value:+.3f}"
        # value in the corner, tinted by who is to move
        self.ax_board.text(
            0.02,
            0.98,
            f"value (to move): {value_str}",
            transform=self.ax_board.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(
                boxstyle="round",
                facecolor=YELLOW if to_move == 1 else RED,
                alpha=0.85,
            ),
        )

        self.fig.suptitle(
            f"{self.title}   |   move {self.idx + 1}/{len(self.frames)}   |   "
            f"to move: {who}   |   ←/→ to navigate",
            fontsize=12,
        )
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="sim log json (defaults to the latest in sim_logs/)",
    )
    args = parser.parse_args()

    log_dir = os.path.join(project_root(), "sim_logs")
    path = args.path or latest_log(log_dir)
    if path is None:
        raise SystemExit(f"No sim_<n>.json found in {log_dir}. Run a simulation first.")

    frames = load_frames(path)
    if not frames:
        raise SystemExit(f"{path} has no frames.")

    print(f"Loaded {len(frames)} frames from {path}")
    # keep a strong reference: matplotlib holds key-press callbacks weakly, so
    # if the Viewer is garbage-collected the arrow keys silently stop working.
    viewer = Viewer(frames, title=os.path.basename(path))
    plt.show()
    return viewer


if __name__ == "__main__":
    main()
