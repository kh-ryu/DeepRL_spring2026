from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def short_label(run_name: str) -> str:
    match = re.match(r"^HalfCheetah-v4_(.+?)_sd\d+_\d{8}_\d{6}$", run_name)
    return match.group(1) if match else run_name


def get_halfcheetah_runs(exp_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in exp_dir.iterdir()
        if p.is_dir() and p.name.startswith("HalfCheetah-v4_") and (p / "log.csv").exists()
    )


def plot_curves(
    runs: list[Path], y_col: str, title: str, y_label: str, out_file: Path
) -> None:
    plt.figure(figsize=(8, 5))
    plotted = 0

    for run in runs:
        df = pd.read_csv(run / "log.csv")
        if "Train_EnvstepsSoFar" not in df.columns or y_col not in df.columns:
            continue

        plt.plot(
            df["Train_EnvstepsSoFar"],
            df[y_col],
            linewidth=2,
            label=short_label(run.name),
        )
        plotted += 1

    if plotted == 0:
        plt.close()
        raise ValueError(f"No runs had required columns for y_col={y_col}")

    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = repo_root / "exp"
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = get_halfcheetah_runs(exp_dir)

    baseline_runs = []
    for run in runs:
        df = pd.read_csv(run / "log.csv", nrows=1)
        if "Baseline Loss" in df.columns:
            baseline_runs.append(run)

    plot_curves(
        baseline_runs,
        y_col="Baseline Loss",
        title="HalfCheetah: Baseline Loss vs Environment Steps",
        y_label="Baseline Loss",
        out_file=out_dir / "halfcheetah_baseline_loss_curve.png",
    )
    plot_curves(
        runs,
        y_col="Eval_AverageReturn",
        title="HalfCheetah: Eval Return vs Environment Steps",
        y_label="Eval Average Return",
        out_file=out_dir / "halfcheetah_eval_return_curve.png",
    )

    print("Saved:")
    print(out_dir / "halfcheetah_baseline_loss_curve.png")
    print(out_dir / "halfcheetah_eval_return_curve.png")


if __name__ == "__main__":
    main()
