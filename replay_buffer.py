import torch
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, policy_vec, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # Convert state and policy to numpy arrays before storing.
        self.buffer[self.position] = (
            np.array(state, dtype=np.float32),
            np.array(policy_vec, dtype=np.float32),
            float(reward),
        )
        self.position = (self.position + 1) % self.capacity

    def push(self, buf: list):
        assert len(buf) > 0
        assert len(buf[0]) == 3
        for b in buf:
            self.push(*b)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        batch = random.sample(self.buffer, batch_size)
        return self._to_tensor(batch)

    def get_first_n(self, n):
        batch = self.buffer[:n]
        return self._to_tensor(batch)

    def _to_tensor(self, batch):
        if not batch:
            return None, None, None

        states, policy_vecs, rewards = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(policy_vecs), dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
