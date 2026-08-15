import torch

from env.connect4 import Connect4


def encode_board_state(game: Connect4, current_player):
    state = game.get_state()
    return torch.stack(
        [
            (state == current_player).float(),
            (state == game.other_player(current_player)).float(),
        ]
    )
