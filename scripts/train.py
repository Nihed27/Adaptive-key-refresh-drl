"""Train a DQN key-refresh agent on one scenario.

Usage:
    python scripts/train.py --scenario byzantine-trust --episodes 200
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akr import SCENARIOS, DQNAgent, DQNConfig, make_env, run_episode  # noqa: E402

VAL_SEEDS = range(7000, 7010)  # validation seeds (disjoint from tuning/test seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="byzantine-trust", choices=list(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(f"{args.out}/models", exist_ok=True)
    env = make_env(args.scenario)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, DQNConfig(), seed=args.seed)

    log_path = f"{args.out}/train_{args.scenario}.csv"
    model_path = f"{args.out}/models/dqn_{args.scenario}.pt"
    val_env = make_env(args.scenario)
    best_val = -np.inf
    t0 = time.time()
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "refreshes", "lifetime", "epsilon"])
        for ep in range(args.episodes):
            # training seeds are disjoint from the evaluation seeds (>= 10_000)
            obs, _ = env.reset(seed=args.seed * 100_000 + ep)
            done, ret, refreshes, lifetime = False, 0.0, 0, 0
            while not done:
                a = agent.act(obs, greedy=False)
                obs2, r, term, trunc, info = env.step(a)
                done = term or trunc
                # time-limit truncation is not a real terminal state -> bootstrap
                agent.observe(obs, a, r, obs2, float(term))
                obs = obs2
                ret += r
                refreshes += int(info["refreshed"])
                lifetime += int(info["alive"])
            writer.writerow([ep, round(ret, 3), refreshes, lifetime, round(agent.epsilon(), 3)])
            if ep % 10 == 9:
                # greedy validation -> keep the best checkpoint (DQN returns are noisy)
                val = np.mean([run_episode(val_env, agent, s)["ret"] for s in VAL_SEEDS])
                tag = ""
                if val > best_val:
                    best_val, tag = val, "  <- best, saved"
                    agent.save(model_path)
                print(f"[{args.scenario}] ep {ep:4d}  train return {ret:8.1f}  refreshes {refreshes:3d}  "
                      f"eps {agent.epsilon():.2f}  | greedy val return {val:8.1f}{tag}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    print(f"best validation return {best_val:.1f} -> {model_path}")


if __name__ == "__main__":
    main()
