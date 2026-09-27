from .env import EnvConfig, KeyRefreshEnv, OBS_NAMES
from .baselines import PeriodicPolicy, ThresholdPolicy
from .dqn import DQNAgent, DQNConfig

SCENARIOS = {
    # honest monitoring, naive averaging
    "honest": dict(byzantine=False, aggregator="mean"),
    # compromised watchdogs lie, naive averaging
    "byzantine-mean": dict(byzantine=True, aggregator="mean"),
    # compromised watchdogs lie, trust-aware aggregation
    "byzantine-trust": dict(byzantine=True, aggregator="trust"),
}


def make_env(scenario: str, seed: int | None = None, **overrides) -> KeyRefreshEnv:
    cfg = EnvConfig(**{**SCENARIOS[scenario], **overrides})
    return KeyRefreshEnv(cfg, seed=seed)


def run_episode(env: KeyRefreshEnv, policy, seed: int, record: bool = False):
    """Roll out one episode with a policy exposing reset() and act(obs)."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    stats = dict(ret=0.0, sec=0.0, refreshes=0, energy=0.0, messages=0, lifetime=0, est_err=0.0)
    trace = []
    done = False
    while not done:
        a = policy.act(obs)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        stats["ret"] += r
        stats["sec"] += info["link_compromise"]
        stats["refreshes"] += int(info["refreshed"])
        stats["energy"] += info["energy"]
        stats["messages"] += info["messages"]
        stats["lifetime"] += int(info["alive"])
        if info["alive"]:
            stats["est_err"] += abs(info["est_frac"] - info["true_frac"])
        if record:
            trace.append(info)
    T = env.cfg.horizon
    stats["sec"] /= T
    stats["est_err"] /= max(stats["lifetime"], 1)
    return (stats, trace) if record else stats
