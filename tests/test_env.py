import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akr import EnvConfig, KeyRefreshEnv, PeriodicPolicy, ThresholdPolicy, make_env, run_episode  # noqa: E402
from akr.aggregation import MeanAggregator, TrustAggregator  # noqa: E402


def test_observation_space():
    env = make_env("byzantine-trust", seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert r <= 0


def test_link_compromise_formula():
    env = KeyRefreshEnv(EnvConfig(ring_size=50, key_pool=1000))
    assert env.link_compromise(0) == 0.0
    assert np.isclose(env.link_compromise(1), 0.05)
    assert np.isclose(env.link_compromise(10), 1 - 0.95 ** 10)


def test_refresh_heals_network():
    env = make_env("honest", lambda_calm=5.0)  # many captures
    env.reset(seed=1)
    for _ in range(5):
        env.step(0)
    assert env.compromised.sum() > 0
    env.cfg.lambda_calm = 0.0
    env.cfg.lambda_attack = 0.0
    _, _, _, _, info = env.step(1)
    assert info["refreshed"] and info["compromised"] == 0


def test_event_reporting_sends_fewer_messages():
    per = run_episode(make_env("byzantine-trust", reporting="periodic"), PeriodicPolicy(30), seed=3)
    evt = run_episode(make_env("byzantine-trust", reporting="event"), PeriodicPolicy(30), seed=3)
    assert evt["messages"] < 0.1 * per["messages"]


def test_trust_filters_consistent_liar():
    n, m = 20, 5
    rng = np.random.default_rng(0)
    monitors = np.array([rng.choice([k for k in range(n) if k != j], m, replace=False) for j in range(n)])
    liar = 0
    trust, mean = TrustAggregator(n, m), MeanAggregator(n, m)
    for _ in range(100):
        reports = np.zeros((n, m), dtype=bool)
        reports[monitors == liar] = True          # node 0 frames everybody it watches
        s_trust = trust.aggregate(reports, monitors)
        s_mean = mean.aggregate(reports, monitors)
    assert trust.trust[liar] < 0.2
    assert s_trust.mean() < 0.25 * s_mean.mean()


def test_threshold_policy():
    p = ThresholdPolicy(0.1)
    assert p.act(np.array([0.05])) == 0
    assert p.act(np.array([0.2])) == 1
