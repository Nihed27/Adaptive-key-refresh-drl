"""Benchmark DQN against tuned baselines and produce tables + figures.

Protocol (fair comparison):
  * Baselines are tuned (grid search) on tuning seeds 5000-5019,
    separately for each scenario.
  * Every policy is then evaluated on the same unseen test seeds 10000-10049.

Usage:
    python scripts/evaluate.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akr import (SCENARIOS, DQNAgent, PeriodicPolicy, ThresholdPolicy,  # noqa: E402
                 make_env, run_episode)

OUT = "results"
FIG = f"{OUT}/figures"
TUNE_SEEDS = range(5000, 5020)
TEST_SEEDS = range(10000, 10050)
PERIODS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]

# validated categorical palette (fixed order) + text/surface tokens
C_DQN, C_THR, C_PER = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"honest": "Honest monitoring", "byzantine-mean": "Byzantine + mean agg.",
          "byzantine-trust": "Byzantine + trust agg."}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})


def evaluate(env, policy, seeds):
    runs = [run_episode(env, policy, s) for s in seeds]
    return {k: np.array([r[k] for r in runs]) for k in runs[0]}


def tune(env, make_policy, grid):
    best = max(grid, key=lambda g: evaluate(env, make_policy(g), TUNE_SEEDS)["ret"].mean())
    return make_policy(best)


def load_dqn(env, scenario):
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load(f"{OUT}/models/dqn_{scenario}.pt")
    return agent


def main():
    os.makedirs(FIG, exist_ok=True)
    table, results = [], {}
    for sc in SCENARIOS:
        env = make_env(sc)
        policies = [
            tune(env, PeriodicPolicy, PERIODS),
            tune(env, ThresholdPolicy, THRESHOLDS),
            load_dqn(env, sc),
        ]
        results[sc] = {}
        for p in policies:
            r = evaluate(env, p, TEST_SEEDS)
            results[sc][p.name] = r
            table.append([sc, p.name, r["ret"].mean(), 1.96 * r["ret"].std() / np.sqrt(len(r["ret"])),
                          r["sec"].mean(), r["refreshes"].mean(), r["energy"].mean(),
                          r["lifetime"].mean(), r["est_err"].mean()])
            print(f"{sc:16s} {p.name:22s} return {r['ret'].mean():8.1f}  sec {r['sec'].mean():.3f}  "
                  f"refresh {r['refreshes'].mean():5.1f}  life {r['lifetime'].mean():6.1f}  "
                  f"est_err {r['est_err'].mean():.4f}", flush=True)

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "policy", "return", "ci95", "avg_link_compromise", "refreshes",
                    "energy", "lifetime", "estimation_mae"])
        w.writerows(table)

    # ---------------------------------------------------------- reporting ablation
    abl = []
    for mode in ["periodic", "event"]:
        env = make_env("byzantine-trust", reporting=mode)
        r = evaluate(env, ThresholdPolicy(0.05), TEST_SEEDS[:20])
        abl.append((mode, r["messages"].mean(), r["energy"].mean(), r["ret"].mean()))

    write_markdown(table, abl)
    plot_bars(results)
    plot_timeline()
    plot_training()


def write_markdown(table, abl):
    lines = ["# Results (50 unseen test episodes, 1000 steps each)", "",
             "Total cost = −return (lower is better). ± = 95% confidence interval.", "",
             "| Scenario | Policy | Total cost ↓ | Avg. link compromise ↓ | Refreshes | Network lifetime | Estimation MAE |",
             "|---|---|---|---|---|---|---|"]
    for sc, name, ret, ci, sec, ref, en, life, err in table:
        lines.append(f"| {LABELS[sc]} | {name} | {-ret:.1f} ± {ci:.1f} | {sec:.3f} | {ref:.1f} | "
                     f"{life:.0f} | {err:.4f} |")
    lines += ["", "## Event-driven vs periodic reporting (Byzantine + trust, threshold policy)", "",
              "| Reporting | Messages / episode | Energy used | Total cost |", "|---|---|---|---|"]
    for mode, msgs, en, ret in abl:
        lines.append(f"| {mode} | {msgs:,.0f} | {en:.3f} | {-ret:.1f} |")
    red = 100 * (1 - abl[1][1] / abl[0][1])
    lines += ["", f"Event-driven reporting sends **{red:.1f}% fewer messages** for the same information."]
    open(f"{OUT}/results.md", "w").write("\n".join(lines) + "\n")


def plot_bars(results):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    scen = list(results)
    kinds = [("Periodic", C_PER), ("Threshold", C_THR), ("DQN", C_DQN)]
    width = 0.26
    for k, (kind, color) in enumerate(kinds):
        vals, errs = [], []
        for sc in scen:
            name = next(n for n in results[sc] if n.startswith(kind))
            r = -results[sc][name]["ret"]
            vals.append(r.mean())
            errs.append(1.96 * r.std() / np.sqrt(len(r)))
        x = np.arange(len(scen)) + (k - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=color, label=kind if kind != "DQN" else "DQN (ours)",
               yerr=errs, error_kw=dict(ecolor=MUTED, lw=1, capsize=3))
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels([LABELS[s] for s in scen])
    ax.set_ylabel("Total cost per episode (lower is better)")
    ax.set_title("Key refresh policies — security + energy cost", loc="left", color=TEXT)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)   # headroom for the legend
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/benchmark.png", dpi=160)
    plt.close(fig)


def plot_timeline(seed=10003, T=400):
    """What the Base Station sees during a false-alarm campaign, and what the DQN does."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, (agg, color, title) in zip(axes[:2], [("mean", C_THR, "Naive mean aggregation"),
                                                   ("trust", C_DQN, "Trust-aware aggregation")]):
        env = make_env("byzantine-trust", aggregator=agg)
        _, tr = run_episode(env, PeriodicPolicy(25), seed, record=True)
        t = np.arange(T)
        camp = np.array([i["campaign"] for i in tr[:T]])
        ax.fill_between(t, 0, 1, where=camp, transform=ax.get_xaxis_transform(),
                        color="#d9d7d0", alpha=0.8, lw=0, label="False-alarm campaign")
        ax.plot(t, [i["true_frac"] for i in tr[:T]], color=TEXT, lw=1.5, label="True compromised fraction")
        ax.plot(t, [i["est_frac"] for i in tr[:T]], color=color, lw=2, label="Base Station estimate")
        ax.set_title(title, loc="left", color=TEXT)
        ax.set_ylabel("Fraction of nodes")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(loc="upper left", ncol=3, fontsize=8)

    env = make_env("byzantine-trust")
    agent = load_dqn(env, "byzantine-trust")
    _, tr = run_episode(env, agent, seed, record=True)
    t = np.arange(T)
    ax = axes[2]
    att = np.array([i["attack"] for i in tr[:T]])
    ax.fill_between(t, 0, 1, where=att, transform=ax.get_xaxis_transform(),
                    color="#f6d9cc", alpha=0.8, lw=0, label="Attack regime")
    ax.plot(t, [i["link_compromise"] for i in tr[:T]], color=TEXT, lw=1.5, label="Link compromise L(c)")
    ref = [k for k in range(T) if tr[k]["refreshed"]]
    ax.vlines(ref, 0, 0.08, color=C_DQN, lw=2, label="DQN key refresh")
    ax.set_title("DQN decisions (Byzantine + trust)", loc="left", color=TEXT)
    ax.set_ylabel("Exposed link keys")
    ax.set_xlabel("Time step")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/timeline.png", dpi=160)
    plt.close(fig)


def plot_training():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for sc, color in zip(SCENARIOS, [C_PER, C_THR, C_DQN]):
        path = f"{OUT}/train_{sc}.csv"
        if not os.path.exists(path):
            continue
        d = np.loadtxt(path, delimiter=",", skiprows=1)
        cost = -d[:, 1]
        smooth = np.convolve(cost, np.ones(10) / 10, mode="valid")
        ax.plot(d[9:, 0], smooth, color=color, lw=2, label=LABELS[sc])
    ax.set_yscale("log")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total cost (10-episode average, log)")
    ax.set_title("DQN training curves", loc="left", color=TEXT)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/training.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
