import numpy as np

class SoftmaxBanditAttacker:
    """
    Attacker treats each node as an arm in a softmax bandit.
    It never sees assets or the payoff matrix, only (node j, reward r).
    """
    def __init__(self, n_nodes: int, eta: float = 0.1, seed: int = 7):
        self.n = n_nodes
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)
        self.prefs = np.zeros(n_nodes, dtype=float)

    def _softmax(self):
        m = np.max(self.prefs)
        z = np.exp(self.prefs - m)
        s = z.sum()
        if s <= 0 or not np.isfinite(s):
            return np.ones(self.n, dtype=float) / self.n
        return z / s

    def sample_node(self) -> int:
        p = self._softmax()
        return int(self.rng.choice(self.n, p=p))

    def get_distribution(self) -> np.ndarray:
        return self._softmax()

    def update(self, j: int, reward: float):
        self.prefs[j] += self.eta * reward
