"""Simulated IoT / Wireless Sensor Network (WSN) where a Base Station decides
*when* to refresh the symmetric keys of the network (self-healing).

Model summary
-------------
* Key management: Random Key Predistribution (Eschenauer & Gligor, 2002).
  Each node stores `ring_size` keys drawn from a pool of `key_pool` keys.
  If the adversary has captured `c` nodes since the last refresh, a given
  link key is exposed with probability  L(c) = 1 - (1 - ring_size/key_pool)^c.
* Adversary: mobile adversary (as in POSH / self-healing UWSN literature).
  Node captures follow a Poisson process whose rate switches between a
  "calm" and an "attack" regime (Markov-modulated Poisson process).
  A key refresh makes every key the adversary learned obsolete -> c = 0.
* Monitoring: every node is watched by `n_watchdogs` random neighbours.
  Each watchdog keeps a belief "this neighbour is compromised" (with
  detection delay and false alarms) and reports it to the Base Station.
* Byzantine nodes: compromised watchdogs lie. They hide compromised peers
  (report 0) and frame honest nodes (report 1) to mislead the agent.
  In addition, `n_insiders` nodes run permanently malicious monitoring
  firmware (key refresh does not heal them). They hide compromised nodes
  and run intermittent false-alarm campaigns to trigger useless, energy
  draining refreshes.
* Reporting: `periodic` (every watchdog reports every step) or `event`
  (a watchdog only sends a message when its report changes). The Base
  Station keeps the last received value, so both give the same view,
  but event-driven reporting sends far fewer messages.
* Energy: normalised average battery (1.0 = full). A refresh costs
  `e_refresh`, each report message costs `e_msg / n_nodes`. When the
  battery reaches 0 the network is dead (security cost = 1 every step).

Action space: 0 = wait, 1 = refresh keys now.
Reward: r_t = -w_security * L(c_t) - w_energy * energy_t / e_refresh
(one refresh costs 1 "energy unit" in the reward).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .aggregation import make_aggregator


@dataclass
class EnvConfig:
    n_nodes: int = 100
    key_pool: int = 1000
    ring_size: int = 50
    n_watchdogs: int = 5
    horizon: int = 1000
    # adversary (Markov-modulated Poisson captures, nodes / step)
    lambda_calm: float = 0.02
    lambda_attack: float = 0.6
    p_calm_to_attack: float = 0.01
    p_attack_to_calm: float = 0.05
    # watchdog behaviour
    p_detect: float = 0.15       # per step, honest watchdog notices a compromised neighbour
    p_false_alarm: float = 0.002 # per step, honest watchdog wrongly suspects an honest neighbour
    p_fa_recover: float = 0.2    # per step, a false alarm is cleared
    # byzantine behaviour of compromised watchdogs
    byzantine: bool = True
    n_insiders: int = 15         # nodes whose monitoring firmware is permanently malicious
    p_frame: float = 0.6         # share of honest neighbours a liar frames
    p_campaign_on: float = 0.02  # insiders start a false-alarm campaign
    p_campaign_off: float = 0.05 # insiders stop the campaign
    # reporting / aggregation
    reporting: str = "event"     # "periodic" | "event"
    aggregator: str = "trust"    # "mean" | "majority" | "trust"
    # energy
    e_msg: float = 2e-5
    e_refresh: float = 1e-2
    # reward weights
    w_security: float = 1.0
    w_energy: float = 1.0

    def to_dict(self):
        return asdict(self)


OBS_NAMES = [
    "est_compromised",   # aggregated estimate of the compromised fraction
    "ema_fast",          # fast exponential moving average of the estimate
    "ema_slow",          # slow exponential moving average of the estimate
    "time_since_refresh",
    "battery",
    "time_remaining",
]


class KeyRefreshEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(OBS_NAMES),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._q = self.cfg.ring_size / self.cfg.key_pool
        self.rng = np.random.default_rng(seed)
        self.aggregator = make_aggregator(self.cfg.aggregator, self.cfg.n_nodes, self.cfg.n_watchdogs)

    # ------------------------------------------------------------------ utils
    def link_compromise(self, c: int) -> float:
        """Probability that a link key is known by the adversary (RKP model)."""
        return 1.0 - (1.0 - self._q) ** c

    def _assign_watchdogs(self):
        n, m = self.cfg.n_nodes, self.cfg.n_watchdogs
        mons = np.empty((n, m), dtype=np.int64)
        for j in range(n):
            cand = self.rng.choice(n - 1, size=m, replace=False)
            cand[cand >= j] += 1  # skip j itself
            mons[j] = cand
        return mons

    def _new_epoch(self):
        """State reset after a key refresh (new keying epoch)."""
        n, m = self.cfg.n_nodes, self.cfg.n_watchdogs
        self.compromised = np.zeros(n, dtype=bool)
        self.belief = np.zeros((n, m), dtype=bool)
        self.framed = self.rng.random((n, m)) < self.cfg.p_frame
        self.last_sent = np.zeros((n, m), dtype=bool)
        self.steps_since_refresh = 0

    def _obs(self):
        return np.array(
            [
                self.est,
                self.ema_fast,
                self.ema_slow,
                min(self.steps_since_refresh / 200.0, 1.0),
                max(self.battery, 0.0),
                1.0 - self.t / self.cfg.horizon,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.monitors = self._assign_watchdogs()
        self.insider = np.zeros(self.cfg.n_nodes, dtype=bool)
        if self.cfg.byzantine and self.cfg.n_insiders > 0:
            self.insider[self.rng.choice(self.cfg.n_nodes, self.cfg.n_insiders, replace=False)] = True
        self.campaign = False
        self.aggregator.reset()
        self._new_epoch()
        self.attack = False
        self.battery = 1.0
        self.alive = True
        self.t = 0
        self.est = self.ema_fast = self.ema_slow = 0.0
        return self._obs(), {}

    def step(self, action: int):
        cfg = self.cfg
        n = cfg.n_nodes
        energy = 0.0
        refreshed = False
        msgs = 0

        if self.alive and action == 1:
            energy += cfg.e_refresh
            refreshed = True
            self._new_epoch()

        # --- adversary: regime switch + node captures
        if self.attack:
            if self.rng.random() < cfg.p_attack_to_calm:
                self.attack = False
        elif self.rng.random() < cfg.p_calm_to_attack:
            self.attack = True
        lam = cfg.lambda_attack if self.attack else cfg.lambda_calm
        n_new = self.rng.poisson(lam)
        free = np.flatnonzero(~self.compromised)
        if n_new > 0 and free.size > 0:
            idx = self.rng.choice(free, size=min(n_new, free.size), replace=False)
            self.compromised[idx] = True

        if self.alive:
            # --- watchdog beliefs (honest view)
            tgt_comp = np.broadcast_to(self.compromised[:, None], self.belief.shape)
            u1 = self.rng.random(self.belief.shape)
            u2 = self.rng.random(self.belief.shape)
            detect = tgt_comp & (u1 < cfg.p_detect)
            false_alarm = ~tgt_comp & (u1 < cfg.p_false_alarm)
            recover = ~tgt_comp & (u2 < cfg.p_fa_recover)
            self.belief = (self.belief | detect | false_alarm) & ~recover

            # --- what is actually reported (byzantine watchdogs lie)
            reports = self.belief.copy()
            if cfg.byzantine:
                if self.campaign:
                    self.campaign = self.rng.random() >= cfg.p_campaign_off
                else:
                    self.campaign = self.rng.random() < cfg.p_campaign_on
                captured_liar = self.compromised[self.monitors]
                insider_liar = self.insider[self.monitors]
                liar = captured_liar | insider_liar
                reports[liar & tgt_comp] = False                     # hide accomplices
                framers = captured_liar | (insider_liar & self.campaign)
                reports[framers & ~tgt_comp & self.framed] = True    # frame honest nodes

            if cfg.reporting == "periodic":
                msgs = reports.size
            else:  # event-driven: only changes are transmitted
                msgs = int((reports != self.last_sent).sum())
            self.last_sent = reports
            energy += msgs * cfg.e_msg / n

            # --- base station aggregation
            scores = self.aggregator.aggregate(reports, self.monitors)
            self.est = float(scores.mean())
            self.ema_fast = 0.7 * self.ema_fast + 0.3 * self.est
            self.ema_slow = 0.95 * self.ema_slow + 0.05 * self.est

            self.battery -= energy
            if self.battery <= 0:
                self.alive = False

        c = int(self.compromised.sum())
        link_comp = self.link_compromise(c) if self.alive else 1.0
        reward = -cfg.w_security * link_comp - cfg.w_energy * energy / cfg.e_refresh

        self.t += 1
        self.steps_since_refresh += 1
        truncated = self.t >= cfg.horizon
        info = {
            "compromised": c,
            "true_frac": c / n,
            "est_frac": self.est,
            "link_compromise": link_comp,
            "energy": energy,
            "messages": msgs,
            "refreshed": refreshed,
            "alive": self.alive,
            "attack": self.attack,
            "campaign": bool(getattr(self, "campaign", False)),
        }
        return self._obs(), float(reward), False, truncated, info
