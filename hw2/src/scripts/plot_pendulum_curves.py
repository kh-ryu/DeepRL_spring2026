from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def short_label(run_name: str) -> str:
    match = re.match(r"^InvertedPendulum-v4_(.+?)_sd\d+_\d{8}_\d{6}$", run_name)
    return match.group(1) if match else run_name


def collect_runs(exp_dir: Path) -> list[Path]:
    runs = []
    for path in sorted(exp_dir.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("InvertedPendulum-v4_pendulum"):
            continue
        if not (path / "log.csv").exists():
            continue
        runs.append(path)
    return runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = repo_root / "exp"
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pendulum_learning_curves.png"

    runs = collect_runs(exp_dir)
    if not runs:
        raise FileNotFoundError("No InvertedPendulum-v4 runs with exp_name starting 'pendulum' found.")

    plt.figure(figsize=(8, 5))
    plotted = 0

    for run in runs:
        df = pd.read_csv(run / "log.csv")
        if "Train_EnvstepsSoFar" not in df.columns or "Eval_AverageReturn" not in df.columns:
            continue

        plt.plot(
            df["Train_EnvstepsSoFar"],
            df["Eval_AverageReturn"],
            linewidth=2,
            label=short_label(run.name),
        )
        plotted += 1

    if plotted == 0:
        plt.close()
        raise ValueError("No valid runs had both Train_EnvstepsSoFar and Eval_AverageReturn.")

    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel("Eval_AverageReturn")
    plt.title("InvertedPendulum-v4: Eval Return vs Environment Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()

    print(f"Saved plot: {out_file}")


if __name__ == "__main__":
    main()
