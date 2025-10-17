from __future__ import annotations
import numpy as np

class TabularQLearner:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995, seed: int | None = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, sp: int, done: bool):
        best_next = np.max(self.Q[sp])
        td_target = r + (0.0 if done else self.gamma * best_next)
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
