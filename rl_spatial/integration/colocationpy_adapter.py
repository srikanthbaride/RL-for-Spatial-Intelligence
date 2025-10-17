class ColocationAdapter:
    """Shim adapter for colocation pattern mining.
    Replace method bodies with real calls once `colocationpy` (or your implementation) is available.
    """
    def __init__(self):
        pass

    def mine_colocations(self, points, radius: float = 2.0, min_prev: float = 0.2):
        """points: list of (x, y, type_label). Returns list of discovered pairs like [('A','B'), ...]."""
        # Placeholder behavior: return a static example to keep API stable
        return [('A', 'B')]
