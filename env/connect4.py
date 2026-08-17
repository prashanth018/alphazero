import torch
import torch.nn.functional as F

from env.constants import WIN_KERNELS


class Connect4:
    def __init__(self):
        self.players = [1, -1]
        self.turn = 0
        self.shape = (6, 7)
        self.board = torch.zeros(self.shape[0], self.shape[1], dtype=torch.int8)
        self.actions = [0, 1, 2, 3, 4, 5, 6]

    def reset(self):
        self.board = torch.zeros(self.shape[0], self.shape[1], dtype=torch.int8)
        return self.board

    def get_state(self):
        return self.board

    # def get_valid_actions(self):
    #     actions = []
    #     for action in self.actions:
    #         if self._get_valid_row(action=action) >= 0:
    #             actions.append(action)
    #     return actions

    def get_current_player(self):
        return self.players[self.turn]

    def get_valid_actions(self):
        # zeros per col, vector of (7,)
        actions = (self.board == 0).float().sum(dim=0)
        # return valid actions
        return torch.nonzero(actions > 0, as_tuple=True)[0].tolist()

    def is_board_full(self):
        if self.get_valid_actions() == []:
            return True
        return False

    def get_shape(self):
        self.shape

    def step(self, action):
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action: {action}")
        legal_row = self._get_valid_row(action)
        if legal_row >= 0:
            self.board[legal_row, action] = self.players[self.turn]
            self.turn = (self.turn + 1) % 2
        win, player = self._check_win()
        is_board_full = self.is_board_full()
        # win state
        if win:
            return self.board, True, player
        # draw
        if is_board_full:
            return self.board, True, 0
        return self.board, False, 0

    def _get_col_top(self, col):
        return (self.board[:, col] == 0).sum()

    def _get_valid_row(self, action):
        legal_row = self._get_col_top(action)
        return legal_row - 1

    def _check_win(self):
        for player in self.players:
            if self._player_has_won(player):
                return (True, player)
        return (False, None)

    def other_player(self, current_player):
        if current_player == 1:
            return -1
        return 1

    def _player_has_won(self, player):
        board_conf = (self.board == player).float().unsqueeze(0)
        return (F.conv2d(board_conf, WIN_KERNELS, padding=3) == 4).any()

    def is_done(self):
        win, _ = self._check_win()
        is_board_full = self.is_board_full()
        if win or is_board_full:
            return True
        return False

    def get_encoded_board_state(self):
        state = self.get_state()
        current_player = self.get_current_player()
        return torch.stack(
            [
                (state == current_player).float(),
                (state == self.other_player(current_player)).float(),
            ]
        )
