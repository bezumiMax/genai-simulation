import torch
import numpy as np
from decay_shedule import decay_schedule

class Agent():
    def __init__(self, grid_size, episodes, init_eps=1.0, min_eps=0.1, 
                 eps_decay_ratio=0.9, init_alpha=0.5, min_alpha=0.01, 
                 alpha_decay_ratio=0.5, use_genetic=False):
        """
        Args:
            use_genetic: True если используем генетический алгоритм, False для Double Q-learning
        """
        n_states = grid_size ** 2
        n_actions = 4
        
        if use_genetic:
            # Для генетического алгоритма
            self.policy_weights = np.random.randn(n_states, n_actions) * 0.1
            self.best_fitness = -float('inf')
            self.best_policy_weights = None
            self.use_genetic = True
        else:
            # Для Double Q-learning
            self.Q1 = np.zeros((n_states, n_actions), dtype=np.float64)
            self.Q2 = np.zeros((n_states, n_actions), dtype=np.float64)
            self.epsilons = decay_schedule(init_eps, min_eps, eps_decay_ratio, episodes)
            self.alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, episodes)
            self.use_genetic = False
    
    def update(self, Q1=None, Q2=None, policy_weights=None, fitness=None):
        """
        Обновляет параметры агента.
        Для Double Q-learning: Q1, Q2
        Для генетического алгоритма: policy_weights, fitness
        """
        if self.use_genetic:
            if policy_weights is not None:
                self.policy_weights = policy_weights.copy()
            if fitness is not None and fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_policy_weights = policy_weights.copy()
        else:
            if Q1 is not None and Q2 is not None:
                self.Q1 = Q1.copy()
                self.Q2 = Q2.copy()
    
    def get_policy(self, state):
        """Возвращает вероятности действий для заданного состояния"""
        if self.use_genetic:
            # Softmax policy для генетического алгоритма
            logits = self.policy_weights[state] - np.max(self.policy_weights[state])
            action_probs = np.exp(logits) / np.sum(np.exp(logits))
            return action_probs
        else:
            # Epsilon-greedy policy для Q-learning
            return None
    
    def get_best_policy(self):
        """Возвращает лучшую найденную политику (для генетического алгоритма)"""
        if self.use_genetic and self.best_policy_weights is not None:
            return self.best_policy_weights
        elif self.use_genetic:
            return self.policy_weights
        else:
            return None