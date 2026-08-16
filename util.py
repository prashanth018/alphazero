import torch

from env.connect4 import Connect4


def encode_board_state(game: Connect4):
    state = game.get_state()
    current_player = game.get_current_player()
    return torch.stack(
        [
            (state == current_player).float(),
            (state == game.other_player(current_player)).float(),
        ]
    )
