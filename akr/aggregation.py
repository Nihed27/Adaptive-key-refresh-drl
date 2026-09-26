"""Base Station aggregation of watchdog reports.

`reports[j, s]` is the report (0/1) of watchdog `monitors[j, s]` about node j.
Each aggregator returns a suspicion score in [0, 1] per node; the mean of
those scores is the estimated compromised fraction given to the agent.

* MeanAggregator     : naive average -> directly polluted by liars.
* MajorityAggregator : majority vote per node (robust while liars < m/2).
* TrustAggregator    : Beta-reputation trust per watchdog. A watchdog that
  keeps disagreeing with the (trust-weighted) consensus loses weight, so
  false metrics from byzantine nodes are progressively filtered out.
"""
from __future__ import annotations

import numpy as np


class MeanAggregator:
    def __init__(self, n_nodes: int, n_watchdogs: int):
        self.n, self.m = n_nodes, n_watchdogs

    def reset(self):
        pass

    def aggregate(self, reports: np.ndarray, monitors: np.ndarray) -> np.ndarray:
        return reports.mean(axis=1)


class MajorityAggregator(MeanAggregator):
    def aggregate(self, reports, monitors):
        return (reports.sum(axis=1) > self.m / 2).astype(float)


class TrustAggregator(MeanAggregator):
    """Beta reputation system (Josang & Ismail, 2002) with forgetting factor."""

    def __init__(self, n_nodes, n_watchdogs, decay: float = 0.97, prior: float = 2.0,
                 agree_weight: float = 0.05):
        super().__init__(n_nodes, n_watchdogs)
        self.decay, self.prior, self.agree_weight = decay, prior, agree_weight
        self.reset()

    def reset(self):
        self.alpha = np.full(self.n, self.prior)  # evidence of honest behaviour
        self.beta = np.ones(self.n)                # evidence of dishonest behaviour

    @property
    def trust(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def aggregate(self, reports, monitors):
        w = self.trust[monitors] ** 2                        # (n, m) weights
        scores = (w * reports).sum(axis=1) / w.sum(axis=1)
        consensus = scores > 0.5
        agree = reports == consensus[:, None]
        # agreeing is cheap evidence, disagreeing is strong evidence (asymmetric)
        agree_cnt = np.bincount(monitors.ravel(), weights=agree.ravel(), minlength=self.n)
        total_cnt = np.bincount(monitors.ravel(), minlength=self.n)
        self.alpha = self.decay * self.alpha + (1 - self.decay) * self.prior + agree_cnt * self.agree_weight
        self.beta = self.decay * self.beta + (1 - self.decay) * 1.0 + (total_cnt - agree_cnt)
        return scores


def make_aggregator(name: str, n_nodes: int, n_watchdogs: int):
    return {
        "mean": MeanAggregator,
        "majority": MajorityAggregator,
        "trust": TrustAggregator,
    }[name](n_nodes, n_watchdogs)
