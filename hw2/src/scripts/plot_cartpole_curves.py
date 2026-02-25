# Save as: hw2/src/scripts/plot_cartpole_curves.py
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def short_label(run_name: str) -> str:
    m = re.match(r"^CartPole-v0_(.+?)_sd\d+_\d{8}_\d{6}$", run_name)
    return m.group(1) if m else run_name


def make_plot(runs, title, out_file):
    plt.figure(figsize=(8, 5))
    for run in sorted(runs):
        df = pd.read_csv(run / "log.csv")
        x = df["Train_EnvstepsSoFar"]     # x-axis required by prompt
        y = df["Eval_AverageReturn"]      # average return curve
        plt.plot(x, y, linewidth=2, label=short_label(run.name))

    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel("Average Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


if __name__ == "__main__":
    base = Path("hw2/exp")
    out_dir = base / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = [
        p for p in base.iterdir()
        if p.is_dir() and (p / "log.csv").exists() and p.name.startswith("CartPole-v0_")
    ]

    small_batch_runs = [p for p in all_runs if "_cartpole_lb" not in p.name and "_cartpole" in p.name]
    large_batch_runs = [p for p in all_runs if "_cartpole_lb" in p.name]

    make_plot(
        small_batch_runs,
        "CartPole Small Batch: Learning Curves",
        out_dir / "cartpole_small_batch_learning_curves.png",
    )
    make_plot(
        large_batch_runs,
        "CartPole Large Batch (lb): Learning Curves",
        out_dir / "cartpole_large_batch_learning_curves.png",
    )
