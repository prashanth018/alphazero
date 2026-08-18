import torch
from math import sqrt

from env.connect4 import Connect4

C_PUCT = 1.0


def puct_eval(action_val, prior, visit_count, parent_visit_count):
    return action_val + C_PUCT * prior * (parent_visit_count**0.5) / (1 + visit_count)


def get_decoded_board_state(encoded_state, current_player, other_player):
    if encoded_state.shape != (2, 6, 7):
        raise "Invalid shape"
    return encoded_state[0] * current_player + encoded_state[1] * other_player
