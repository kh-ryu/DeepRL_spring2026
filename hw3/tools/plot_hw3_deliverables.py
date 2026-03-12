from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "exp"


def _to_float(value: str | None) -> float:
    if value in ("", None):
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _run_prefix(run_dir: Path) -> str:
    return run_dir.name.partition("_sd")[0]


def latest_run(prefix: str, exp_dir: Path = EXP_DIR) -> Path:
    candidates = [
        path for path in exp_dir.iterdir()
        if path.is_dir() and _run_prefix(path) == prefix
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run found for prefix '{prefix}' in {exp_dir}. "
            "Run the matching subsection script first."
        )
    return sorted(candidates, key=lambda path: path.name)[-1]


def load_csv(run_dir: Path) -> list[dict[str, float]]:
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log.csv in {run_dir}")

    with log_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _to_float(value) for key, value in row.items()} for row in reader]


def series(run_dir: Path, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = load_csv(run_dir)
    x_vals = []
    y_vals = []
    for row in rows:
        x_val = row.get(x_key, np.nan)
        y_val = row.get(y_key, np.nan)
        if np.isnan(x_val) or np.isnan(y_val):
            continue
        x_vals.append(x_val)
        y_vals.append(y_val)
    return np.asarray(x_vals), np.asarray(y_vals)


def _setup_axis(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def plot_section_2_4(output_dir: Path) -> Path:
    run_dir = latest_run("CartPole-v1_dqn")
    steps, returns = series(run_dir, "step", "Eval_AverageReturn")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, returns, label="CartPole DQN", linewidth=2)
    _setup_axis(ax, "Environment Steps", "Eval Return", "Section 2.4: CartPole-v1")
    ax.legend()
    return _save(fig, output_dir, "sec_2_4_cartpole_eval_return.png")


def plot_section_2_5_lunarlander(output_dir: Path) -> Path:
    run_dir = latest_run("LunarLander-v2_dqn")
    train_steps, train_returns = series(run_dir, "step", "Train_EpisodeReturn")
    eval_steps, eval_returns = series(run_dir, "step", "Eval_AverageReturn")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_steps, train_returns, label="Train Return", alpha=0.8)
    ax.plot(eval_steps, eval_returns, label="Eval Return", linewidth=2)
    _setup_axis(ax, "Environment Steps", "Return", "Section 2.5: LunarLander-v2")
    ax.legend()
    return _save(fig, output_dir, "sec_2_5_lunarlander_train_eval_return.png")


def plot_section_2_5_mspacman(output_dir: Path) -> Path:
    run_dir = latest_run("MsPacman_dqn")
    train_steps, train_returns = series(run_dir, "step", "Train_EpisodeReturn")
    eval_steps, eval_returns = series(run_dir, "step", "Eval_AverageReturn")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_steps, train_returns, label="Train Return", alpha=0.8)
    ax.plot(eval_steps, eval_returns, label="Eval Return", linewidth=2)
    _setup_axis(ax, "Environment Steps", "Return", "Section 2.5: MsPacman")
    ax.legend()
    return _save(fig, output_dir, "sec_2_5_mspacman_train_eval_return.png")


def plot_section_2_6(output_dir: Path) -> Path:
    prefixes = {
        "lr=1e-3": "LunarLander-v2_dqn",
        "lr=2.5e-4": "LunarLander-v2_dqn_lr_0p00025",
        "lr=5e-4": "LunarLander-v2_dqn_lr_0p0005",
        "lr=2e-3": "LunarLander-v2_dqn_lr_0p002",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, prefix in prefixes.items():
        run_dir = latest_run(prefix)
        steps, returns = series(run_dir, "step", "Eval_AverageReturn")
        ax.plot(steps, returns, label=label)

    _setup_axis(
        ax,
        "Environment Steps",
        "Eval Return",
        "Section 2.6: LunarLander-v2 Learning-Rate Sweep",
    )
    ax.legend()
    return _save(fig, output_dir, "sec_2_6_lunarlander_lr_sweep.png")


def plot_section_3_4(output_dir: Path) -> Path:
    run_dir = latest_run("HalfCheetah-v4_sac")
    steps, returns = series(run_dir, "step", "Eval_AverageReturn")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, returns, label="HalfCheetah SAC", linewidth=2)
    _setup_axis(ax, "Environment Steps", "Eval Return", "Section 3.4: HalfCheetah-v4")
    ax.legend()
    return _save(fig, output_dir, "sec_3_4_halfcheetah_eval_return.png")


def plot_section_3_5(output_dir: Path) -> Path:
    fixed_run = latest_run("HalfCheetah-v4_sac")
    autotune_run = latest_run("HalfCheetah-v4_sac_autotune")

    fixed_steps, fixed_returns = series(fixed_run, "step", "Eval_AverageReturn")
    auto_steps, auto_returns = series(autotune_run, "step", "Eval_AverageReturn")
    temp_steps, temps = series(autotune_run, "step", "temperature")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fixed_steps, fixed_returns, label="Fixed temperature", linewidth=2)
    axes[0].plot(auto_steps, auto_returns, label="Auto-tuned temperature", linewidth=2)
    _setup_axis(
        axes[0],
        "Environment Steps",
        "Eval Return",
        "Section 3.5: HalfCheetah-v4 Performance",
    )
    axes[0].legend()

    axes[1].plot(temp_steps, temps, label="Temperature", linewidth=2)
    _setup_axis(
        axes[1],
        "Environment Steps",
        "Temperature",
        "Section 3.5: Auto-tuned Temperature",
    )
    axes[1].legend()
    return _save(fig, output_dir, "sec_3_5_halfcheetah_autotune.png")


def plot_section_3_6(output_dir: Path) -> Path:
    singleq_run = latest_run("Hopper-v4_sac_singleq")
    clipq_run = latest_run("Hopper-v4_sac_clipq")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, run_dir in [
        ("Single-Q", singleq_run),
        ("Clipped Double-Q", clipq_run),
    ]:
        steps, returns = series(run_dir, "step", "Eval_AverageReturn")
        axes[0].plot(steps, returns, label=label, linewidth=2)

        q_steps, q_values = series(run_dir, "step", "q_values")
        axes[1].plot(q_steps, q_values, label=label, linewidth=2)

    _setup_axis(
        axes[0],
        "Environment Steps",
        "Eval Return",
        "Section 3.6: Hopper-v4 Eval Return",
    )
    axes[0].legend()

    _setup_axis(
        axes[1],
        "Environment Steps",
        "Q Values",
        "Section 3.6: Hopper-v4 Q Values",
    )
    axes[1].legend()
    return _save(fig, output_dir, "sec_3_6_hopper_singleq_vs_clipq.png")


def generate_all_figures(output_dir: Path | None = None) -> dict[str, Path]:
    if output_dir is None:
        output_dir = ROOT / "figures"

    return {
        "2.4": plot_section_2_4(output_dir),
        "2.5_lunarlander": plot_section_2_5_lunarlander(output_dir),
        "2.5_mspacman": plot_section_2_5_mspacman(output_dir),
        "2.6": plot_section_2_6(output_dir),
        "3.4": plot_section_3_4(output_dir),
        "3.5": plot_section_3_5(output_dir),
        "3.6": plot_section_3_6(output_dir),
    }


def notebook_summary() -> list[str]:
    return [
        "Section 2.4: CartPole eval return vs environment steps.",
        "Section 2.5: LunarLander and MsPacman train return and eval return on the same axes.",
        "Section 2.6: LunarLander learning-rate sweep with four settings.",
        "Section 3.4: HalfCheetah eval return vs environment steps.",
        "Section 3.5: HalfCheetah fixed vs auto-tuned return, plus auto-tuned temperature.",
        "Section 3.6: Hopper single-Q vs clipped double-Q for eval return and q values.",
    ]


__all__ = [
    "generate_all_figures",
    "latest_run",
    "load_csv",
    "notebook_summary",
    "plot_section_2_4",
    "plot_section_2_5_lunarlander",
    "plot_section_2_5_mspacman",
    "plot_section_2_6",
    "plot_section_3_4",
    "plot_section_3_5",
    "plot_section_3_6",
    "series",
]
