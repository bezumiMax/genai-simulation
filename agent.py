import torch
import numpy as np
from decay_shedule import decay_schedule

class Agent():
    def __init__(self, grid_size, episodes, init_eps=1.0, min_eps=0.1, eps_decay_ratio=0.9, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5):
        n_states = grid_size ** 2
        n_actions = 4
        self.Q1 = np.zeros((n_states, n_actions), dtype=np.float64)
        self.Q2 = np.zeros((n_states, n_actions), dtype=np.float64)
        self.epsilons = decay_schedule(init_eps, min_eps, eps_decay_ratio, episodes)
        self.alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, episodes)
    
    def update(self, Q1, Q2):
        self.Q1 = Q1.copy()
        self.Q2 = Q2.copy()