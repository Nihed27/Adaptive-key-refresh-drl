"""Double Deep Q-Network (Mnih et al., 2015; van Hasselt et al., 2016).

Q(s, a) is approximated by an MLP. Target used in the Bellman update:
    y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
Exploration: epsilon-greedy with linear decay.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (
            torch.as_tensor(s, dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64),
            torch.as_tensor(r, dtype=torch.float32),
            torch.as_tensor(s2, dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class DQNConfig:
    gamma: float = 0.98
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_every: int = 2
    target_tau: float = 0.01          # soft (Polyak) target update
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 100_000
    hidden: int = 128
    reward_scale: float = 0.1


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None, seed: int = 0):
        self.cfg = cfg or DQNConfig()
        torch.manual_seed(seed)
        random.seed(seed)
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions, self.cfg.hidden)
        self.q_target = QNetwork(obs_dim, n_actions, self.cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size)
        self.steps = 0
        self.name = "DQN (ours)"

    # -------------------------------------------------------------- acting
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(self.steps / c.eps_decay_steps, 1.0)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, obs, greedy: bool = True) -> int:
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def reset(self):
        pass

    # -------------------------------------------------------------- learning
    def observe(self, s, a, r, s2, done):
        self.buffer.push(s, a, r * self.cfg.reward_scale, s2, done)
        self.steps += 1
        if len(self.buffer) >= self.cfg.learning_starts and self.steps % self.cfg.train_every == 0:
            return self._update()
        return None

    def _update(self):
        c = self.cfg
        s, a, r, s2, d = self.buffer.sample(c.batch_size)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a2 = self.q(s2).argmax(dim=1, keepdim=True)               # Double DQN
            q_next = self.q_target(s2).gather(1, a2).squeeze(1)
            y = r + c.gamma * (1.0 - d) * q_next                       # Bellman target
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - c.target_tau).add_(c.target_tau * p)
        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location="cpu"))
        self.q_target.load_state_dict(self.q.state_dict())
