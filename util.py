import os
import re

import torch
from math import sqrt
from torch import tensor

from env.connect4 import Connect4

C_PUCT = 1.0


def puct_eval(action_val, prior, visit_count, parent_visit_count):
    return action_val + C_PUCT * prior * ((parent_visit_count - 1) ** 0.5) / (
        1 + visit_count
    )


def sample_move_from_visits(child_visits: dict, temperature):
    # child_visits: {action: visit_count} over valid actions
    actions = list(child_visits)
    counts = torch.tensor([child_visits[a] for a in actions], dtype=torch.float)

    # greedy
    if temperature == 0:
        return actions[counts.argmax().item()]

    # sample
    weights = counts ** (1 / temperature)
    idx = torch.multinomial(weights, 1).item()
    return actions[idx]


def get_decoded_board_state(encoded_state, current_player, other_player):
    if encoded_state.shape != (2, 6, 7):
        raise "Invalid shape"
    return encoded_state[0] * current_player + encoded_state[1] * other_player


def re_normalize(policy, mask):
    # 0 the probs for illegal moves
    policy[~mask] = 0.0
    # normalize
    s = policy.sum()
    if s > 0:
        policy = policy / policy.sum()
    else:
        # if the network policy makes all legal action
        # probs 0.0 then fallback to uniform distribution
        policy = mask.float() / mask.sum()
    return policy


def get_mock_predictor_val():
    return tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1.0


def next_log_path(log_dir="sim_logs", ext="json"):
    os.makedirs(log_dir, exist_ok=True)
    # find the highest existing sim_<n>.<ext> and go one up
    nums = [
        int(m.group(1))
        for f in os.listdir(log_dir)
        if (m := re.fullmatch(rf"sim_(\d+)\.{ext}", f))
    ]
    next_num = max(nums, default=0) + 1
    return os.path.join(log_dir, f"sim_{next_num}.{ext}")


def buffer_to_frames(buf):
    """Turn a self-play buffer into JSON-friendly per-move frames.

    Each buffer entry is (state, encoded_state, policy[, value]); we keep the
    decoded board, the MCTS policy, and the value target (from the perspective
    of the player to move at that frame).
    """
    frames = []
    for t in buf:
        board = t[0]
        # state may be a torch tensor or already a list
        board = board.tolist() if hasattr(board, "tolist") else board
        frames.append(
            {
                "board": board,
                "policy": list(t[2]),
                "value": t[3] if len(t) > 3 else None,
            }
        )
    return frames
