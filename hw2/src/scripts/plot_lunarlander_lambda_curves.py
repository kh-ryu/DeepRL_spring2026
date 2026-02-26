from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_LAMBDAS = ["0", "0.95", "0.98", "0.99", "1"]


def extract_lambda(run_name: str) -> str | None:
    match = re.search(r"lambda([0-9.]+)_sd", run_name)
    return match.group(1) if match else None


def collect_lunarlander_runs(exp_dir: Path) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for path in sorted(exp_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("LunarLander-v2_"):
            continue
        if not (path / "log.csv").exists():
            continue
        lam = extract_lambda(path.name)
        if lam is None:
            continue
        runs.append((lam, path))
    return runs


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exp_dir = repo_root / "exp"
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "lunar_lander_lambda_learning_curves.png"

    runs = collect_lunarlander_runs(exp_dir)
    if not runs:
        raise FileNotFoundError("No LunarLander-v2 runs with log.csv found in exp/.")

    plt.figure(figsize=(8, 5))
    summary: list[tuple[str, float, float, float, int, int]] = []

    for lam, run_dir in runs:
        df = pd.read_csv(run_dir / "log.csv")
        if "Train_EnvstepsSoFar" not in df.columns or "Eval_AverageReturn" not in df.columns:
            continue

        x = df["Train_EnvstepsSoFar"]
        y = df["Eval_AverageReturn"]

        plt.plot(x, y, linewidth=2, marker="o", markersize=2, label=f"lambda={lam}")
        summary.append(
            (lam, float(y.iloc[0]), float(y.iloc[-1]), float(y.max()), int(x.iloc[-1]), len(df))
        )

    if not summary:
        plt.close()
        raise ValueError(
            "No valid LunarLander-v2 logs had both Train_EnvstepsSoFar and Eval_AverageReturn."
        )

    plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
    plt.ylabel("Eval Average Return")
    plt.title("LunarLander-v2: Learning Curves by GAE Lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()

    observed = {lam for lam, _ in runs}
    missing = [lam for lam in EXPECTED_LAMBDAS if lam not in observed]

    print(f"Saved plot: {out_file}")
    print("lambda, first_eval, final_eval, best_eval, final_envsteps, num_points")
    for row in sorted(summary, key=lambda r: float(r[0])):
        print(", ".join(map(str, row)))
    if missing:
        print("Missing lambdas:", ", ".join(missing))


if __name__ == "__main__":
    main()
