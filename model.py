import torch
import torch.nn as nn


class AlphaZeroNet(nn.Module):
    STATE_CHANNELS = 2
    ACTION_DIM = 7
    STATE_VALUE_DIM = 1

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(self.STATE_CHANNELS, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            # self.net(),
            nn.Flatten(),
            nn.Linear(2688, self.ACTION_DIM),
        )

        self.value_head = nn.Sequential(
            # self.net(),
            nn.Flatten(),
            nn.Linear(2688, self.STATE_VALUE_DIM),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)
        return (policy_logits, state_value)


if __name__ == "__main__":
    net = AlphaZeroNet()
    p, v = net(torch.zeros((1, 2, 6, 7)))
    print(p, v)
    print(p.shape, v.shape)
