"""Non-learning key refresh policies used as baselines."""
from __future__ import annotations


class PeriodicPolicy:
    """Classical static scheme: refresh every `interval` steps."""

    def __init__(self, interval: int):
        self.interval = interval
        self.name = f"Periodic (T={interval})"

    def reset(self):
        self.k = 0

    def act(self, obs) -> int:
        self.k += 1
        if self.k >= self.interval:
            self.k = 0
            return 1
        return 0


class ThresholdPolicy:
    """Reactive heuristic: refresh when the estimated compromised fraction
    reaches `theta`."""

    def __init__(self, theta: float):
        self.theta = theta
        self.name = f"Threshold (θ={theta:.3f})"

    def reset(self):
        pass

    def act(self, obs) -> int:
        return int(obs[0] >= self.theta)
