import torch

# import torch.nn.functional as F

horizontal = torch.tensor(
    [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32
)

vertical = torch.tensor(
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.float32
)

diagonal = torch.tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32
)

reverse_diagonal = torch.tensor(
    [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32
)

filters = torch.stack([horizontal, vertical, diagonal, reverse_diagonal])

WIN_KERNELS = filters.unsqueeze(1)
