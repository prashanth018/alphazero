import torch
from torch.nn import (
    Conv2d,
    Sequential,
    Module,
    ReLU,
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Flatten,
    Linear,
)


class AlphaZeroNet(Module):
    STATE_CHANNELS = 2
    ACTION_DIM = 7
    STATE_VALUE_DIM = 1

    def __init__(self):
        super().__init__()
        self.net = Sequential(
            # (in, out, h, w) = (2, 8, 6, 7)
            Conv2d(self.STATE_CHANNELS, 8, kernel_size=3, padding=1),
            # AdaptiveAvgPool2d((6, 7)),
            BatchNorm2d(8),
            ReLU(),
            # (in, out, h, w) = (8, 32, 6, 7)
            Conv2d(8, 32, kernel_size=3, padding=1),
            # AdaptiveAvgPool2d((6, 7)),
            BatchNorm2d(32),
            ReLU(),
            # (in, out, h, w) = (32, 128, 6, 7)
            Conv2d(32, 64, kernel_size=3, padding=1),
            # AdaptiveAvgPool2d((4, 4)),
            # (in, out, h, w) = (128, 6, 7)
            BatchNorm2d(64),
            ReLU(),
        )

        self.policy_head = Sequential(
            Flatten(),
            Linear(2688, self.ACTION_DIM),
        )

        self.value_head = Sequential(
            Flatten(),
            Linear(2688, self.STATE_VALUE_DIM),
        )

    def forward(self, x):
        x = self.net(x)
        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)
        return (policy_logits, state_value)


def print_params(model):
    val = 0.0
    for p in model.parameters():
        val += p.numel()
    return val


if __name__ == "__main__":
    model = AlphaZeroNet()
    print(print_params(model))
    p, v = model(torch.zeros((1, 2, 6, 7)))
    # print(p, v)
    # print(p.shape, v.shape)
    # print(p[0][2].item())
