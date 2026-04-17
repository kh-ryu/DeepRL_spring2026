import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ALGO_BY_GROUP = {
    "q1": "SAC+BC",
    "q2": "IQL",
    "q3": "FQL",
}

TARGET_ENVS_BY_GROUP = {
    "q1": [
        "cube-single-play-singletask-task1-v0",
        "antsoccer-arena-navigate-singletask-task1-v0",
    ],
    "q2": [
        "cube-single-play-singletask-task1-v0",
        "antsoccer-arena-navigate-singletask-task1-v0",
    ],
    "q3": [
        "cube-single-play-singletask-task1-v0",
        "antsoccer-arena-navigate-singletask-task1-v0",
    ],
}

ENV_LABELS = {
    "cube-single-play-singletask-task1-v0": "cube-single",
    "antsoccer-arena-navigate-singletask-task1-v0": "antsoccer",
    "antmaze-medium-navigate-singletask-task1-v0": "antmaze-medium",
}


@dataclass
class Run:
    group: str
    algorithm: str
    env_name: str
    env_label: str
    alpha: float
    expectile: float | None
    run_dir: Path
    steps: list[int]
    success_rates: list[float]

    @property
    def final_success(self) -> float:
        return self.success_rates[-1]

    @property
    def peak_success(self) -> float:
        return max(self.success_rates)

    @property
    def peak_step(self) -> int:
        peak = self.peak_success
        return self.steps[self.success_rates.index(peak)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report plots from experiment logs.")
    parser.add_argument("--exp_dir", type=Path, default=Path("exp"))
    parser.add_argument("--out_dir", type=Path, default=Path("report_plots"))
    return parser.parse_args()


def load_eval_csv(path: Path) -> tuple[list[int], list[float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty eval.csv: {path}")
    steps = [int(row["step"]) for row in rows]
    success_rates = [float(row["eval/success_rate"]) for row in rows]
    return steps, success_rates


def load_runs(exp_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for group, algorithm in ALGO_BY_GROUP.items():
        group_dir = exp_dir / group
        if not group_dir.exists():
            continue
        for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            flags_path = run_dir / "flags.json"
            eval_path = run_dir / "eval.csv"
            if not flags_path.exists() or not eval_path.exists():
                continue

            with flags_path.open() as f:
                flags = json.load(f)

            agent_kwargs = flags.get("agent_kwargs", {})
            env_name = flags["env_name"]
            alpha = float(agent_kwargs["alpha"])
            expectile = agent_kwargs.get("expectile")
            steps, success_rates = load_eval_csv(eval_path)

            runs.append(
                Run(
                    group=group,
                    algorithm=algorithm,
                    env_name=env_name,
                    env_label=ENV_LABELS.get(env_name, env_name),
                    alpha=alpha,
                    expectile=float(expectile) if expectile is not None else None,
                    run_dir=run_dir,
                    steps=steps,
                    success_rates=success_rates,
                )
            )
    return runs


def select_best_run(runs: list[Run]) -> Run:
    if not runs:
        raise ValueError("select_best_run requires at least one run")
    return sorted(
        runs,
        key=lambda run: (
            -run.final_success,
            -run.peak_success,
            run.peak_step,
            run.alpha,
        ),
    )[0]


def configure_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def save_best_runs_figure(group: str, algorithm: str, runs: list[Run], out_dir: Path) -> list[str]:
    notes: list[str] = []
    target_envs = TARGET_ENVS_BY_GROUP[group]
    available_envs = {run.env_name for run in runs}
    missing_envs = [env for env in target_envs if env not in available_envs]

    best_by_env: dict[str, Run] = {}
    for env_name in target_envs:
        env_runs = [run for run in runs if run.env_name == env_name]
        if env_runs:
            best_by_env[env_name] = select_best_run(env_runs)

    if not best_by_env:
        available = ", ".join(sorted({run.env_label for run in runs}))
        notes.append(f"{algorithm}: no target runs found; available environments: {available}.")
        return notes

    fig, axes = plt.subplots(1, len(target_envs), figsize=(6 * len(target_envs), 4.5), squeeze=False)
    for ax, env_name in zip(axes[0], target_envs):
        if env_name not in best_by_env:
            ax.axis("off")
            ax.text(0.5, 0.5, "Missing runs", ha="center", va="center", fontsize=12)
            ax.set_title(ENV_LABELS.get(env_name, env_name))
            continue

        run = best_by_env[env_name]
        ax.plot(run.steps, run.success_rates, marker="o", linewidth=2, label=f"alpha={run.alpha:g}")
        configure_axis(ax)
        ax.set_title(
            f"{run.env_label}\nalpha={run.alpha:g}, final={run.final_success:.2f}, peak={run.peak_success:.2f}"
        )
        ax.legend()
        notes.append(
            f"{algorithm} best on {run.env_label}: alpha={run.alpha:g}, "
            f"final={run.final_success:.2f}, peak={run.peak_success:.2f}, run={run.run_dir.name}"
        )

    fig.suptitle(f"{algorithm} best-agent training curves", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f"{group}_{algorithm.lower().replace('+', '').replace(' ', '_')}_best_runs.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for env_name in missing_envs:
        notes.append(f"{algorithm} missing required run for {ENV_LABELS.get(env_name, env_name)}.")
    return notes


def save_alpha_sweep_figure(group: str, algorithm: str, runs: list[Run], out_dir: Path) -> list[str]:
    notes: list[str] = []
    cube_env = "cube-single-play-singletask-task1-v0"
    cube_runs = sorted([run for run in runs if run.env_name == cube_env], key=lambda run: run.alpha)
    if len(cube_runs) < 3:
        notes.append(f"{algorithm}: fewer than 3 cube-single runs; skipping alpha sweep.")
        return notes

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in cube_runs:
        ax.plot(
            run.steps,
            run.success_rates,
            marker="o",
            linewidth=2,
            label=f"alpha={run.alpha:g}",
        )
    configure_axis(ax)
    ax.set_title(f"{algorithm} alpha sweep on cube-single")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{group}_{algorithm.lower().replace('+', '').replace(' ', '_')}_cube_alpha_sweep.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    alphas = ", ".join(f"{run.alpha:g}" for run in cube_runs)
    notes.append(f"{algorithm} cube-single alpha sweep includes alpha={alphas}.")
    return notes


def write_summary(summary_lines: list[str], out_dir: Path) -> None:
    summary_path = out_dir / "summary.md"
    with summary_path.open("w") as f:
        f.write("# Report Plot Summary\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.exp_dir)
    summary_lines: list[str] = []

    for group, algorithm in ALGO_BY_GROUP.items():
        group_runs = [run for run in runs if run.group == group]
        if not group_runs:
            summary_lines.append(f"{algorithm}: no runs found in {args.exp_dir / group}.")
            continue
        summary_lines.extend(save_best_runs_figure(group, algorithm, group_runs, args.out_dir))
        if group in {"q1", "q2"}:
            summary_lines.extend(save_alpha_sweep_figure(group, algorithm, group_runs, args.out_dir))

    write_summary(summary_lines, args.out_dir)
    print(f"Wrote plots and summary to {args.out_dir}")
    for line in summary_lines:
        print(f"- {line}")


if __name__ == "__main__":
    main()
