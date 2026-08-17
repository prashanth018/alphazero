import torch
from math import sqrt

C_PUCT = 1.0


def puct_eval(action_val, prior, visit_count, parent_visit_count):
    return action_val + C_PUCT * prior * (parent_visit_count**0.5) / (1 + visit_count)
