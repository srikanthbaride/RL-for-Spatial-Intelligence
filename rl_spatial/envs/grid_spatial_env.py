import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridSpatialEnv(gym.Env):
    """
    A toy spatial grid environment with Points-of-Interest (POIs). 
    The agent moves on a grid; rewards are given for discovering co-located POI pairs within a radius R.

    Observation: agent (row, col) encoded as a single discrete state in [0, grid_size^2).
    Action space: 4 moves (0=up,1=right,2=down,3=left).
    Reward: +1 when the agent's neighborhood contains at least one required co-location pair.
    Episode ends after max_steps.

    POIs: list of tuples (row, col, type_id) with type_id in {0..K-1}
    Co-location target pairs: list of tuples (a, b) meaning type a near type b within radius R.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size: int = 10, n_pois: int = 30, n_types: int = 3, radius: int = 2,
                 target_pairs=None, max_steps: int = 100, seed: int | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.n_pois = n_pois
        self.n_types = n_types
        self.radius = radius
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        if target_pairs is None:
            target_pairs = [(0, 1), (1, 2)]
        self.target_pairs = target_pairs

        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)

        self._pois = None
        self._agent = None
        self._t = 0

        self.reset(seed=seed)

    def _encode_obs(self, row: int, col: int) -> int:
        return row * self.grid_size + col

    def _neighbors_within_radius(self, row, col):
        r = self.radius
        rr_min = max(0, row - r)
        rr_max = min(self.grid_size - 1, row + r)
        cc_min = max(0, col - r)
        cc_max = min(self.grid_size - 1, col + r)
        return rr_min, rr_max, cc_min, cc_max

    def _has_colocation(self, row, col) -> bool:
        rr_min, rr_max, cc_min, cc_max = self._neighbors_within_radius(row, col)
        types_here = set()
        for (r, c, t) in self._pois:
            if rr_min <= r <= rr_max and cc_min <= c <= cc_max:
                types_here.add(t)
        for (a, b) in self.target_pairs:
            if a in types_here and b in types_here:
                return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        # Sample POIs uniformly without duplicates
        self._pois = set()
        while len(self._pois) < self.n_pois:
            r = int(self.rng.integers(0, self.grid_size))
            c = int(self.rng.integers(0, self.grid_size))
            t = int(self.rng.integers(0, self.n_types))
            self._pois.add((r, c, t))
        self._pois = list(self._pois)

        # Place agent randomly
        self._agent = (int(self.rng.integers(0, self.grid_size)), int(self.rng.integers(0, self.grid_size)))
        obs = self._encode_obs(*self._agent)
        info = {"pois": self._pois}
        return obs, info

    def step(self, action):
        self._t += 1
        r, c = self._agent
        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and c < self.grid_size - 1:
            c += 1
        elif action == 2 and r < self.grid_size - 1:
            r += 1
        elif action == 3 and c > 0:
            c -= 1
        self._agent = (r, c)

        reward = 1.0 if self._has_colocation(r, c) else 0.0
        terminated = self._t >= self.max_steps
        truncated = False
        obs = self._encode_obs(r, c)
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), fill_value='.', dtype=object)
        for (r, c, t) in self._pois:
            grid[r, c] = str(t)
        ar, ac = self._agent
        grid[ar, ac] = 'A'
        lines = [''.join(row) for row in grid]
        return "\n".join(lines)
