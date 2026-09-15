from env.connect4 import Connect4
from mcts import MCTS
from model import AlphaZeroNet
from replay_buffer import ReplayBuffer


class Trainer:
    # number of games
    rollouts = 100
    buff_cap = 100000
    sims = 10

    def __init__(self):
        self.game = Connect4()
        self.model = AlphaZeroNet()
        self.mcts = MCTS(game=self.game, net=self.model, rollouts=self.rollouts)
        self.buffer = ReplayBuffer(self.buff_cap)

    def data_gen(self):
        self.mcts.train()
        for _ in range(self.sims):
            buf = self.mcts.simulation()
            self.buffer.push(buf)

    def train(self):
        pass
