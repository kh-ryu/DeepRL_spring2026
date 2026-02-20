"""Generate training curves (loss and reward vs steps) from a local experiment directory."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_wandb_history(wandb_file: Path) -> tuple[list[dict], list[dict]]:
    """Parse a local .wandb binary file and return train and eval history rows."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2

    ds = DataStore()
    ds.open_for_scan(str(wandb_file))

    train_rows, eval_rows = [], []
    while True:
        try:
            data = ds.scan_record()
        except AssertionError:
            # Checksum mismatch — file is truncated or corrupt (e.g. killed run); stop here
            print("Warning: corrupt record encountered, stopping early.")
            break
        if data is None:
            break
        pb = wandb_internal_pb2.Record()
        try:
            pb.ParseFromString(data[1])
        except Exception:
            continue
        if pb.WhichOneof("record_type") != "history":
            continue

        row = {}
        for item in pb.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                pass

        if "train/loss" in row:
            train_rows.append({"step": row.get("step", row.get("_step")), "loss": row["train/loss"]})
        if "eval/mean_reward" in row:
            eval_rows.append({"step": row.get("step", row.get("_step")), "reward": row["eval/mean_reward"]})

    return train_rows, eval_rows


def find_wandb_file(exp_dir: Path) -> Path:
    matches = list(exp_dir.glob("wandb/run-*.wandb"))
    if not matches:
        raise FileNotFoundError(f"No .wandb file found in {exp_dir}/wandb/")
    return matches[0]


def plot_curves(exp_dir: Path, output_path: Path) -> None:
    wandb_file = find_wandb_file(exp_dir)
    print(f"Reading: {wandb_file}")

    train_rows, eval_rows = parse_wandb_history(wandb_file)

    if not train_rows:
        raise ValueError("No train/loss data found in the .wandb file.")
    if not eval_rows:
        print("Warning: no eval/mean_reward data found — the run may have been killed before the first eval.")

    train_df = pd.DataFrame(train_rows).dropna().sort_values("step")
    eval_df = pd.DataFrame(eval_rows).dropna().sort_values("step") if eval_rows else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve with smoothing
    ax1.plot(train_df["step"], train_df["loss"], color="steelblue", linewidth=0.8, alpha=0.4, label="raw")
    smoothed = train_df["loss"].rolling(window=20, min_periods=1).mean()
    ax1.plot(train_df["step"], smoothed, color="steelblue", linewidth=1.8, label="smoothed (w=20)")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward curve
    if eval_df is not None:
        ax2.plot(eval_df["step"], eval_df["reward"], color="darkorange", marker="o", linewidth=1.8, markersize=6)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Evaluation Mean Reward")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"MSE Policy — {exp_dir.name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a local experiment directory.")
    parser.add_argument(
        "exp_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the experiment directory (e.g. exp/seed_42_20260210_120612). "
             "If omitted, the most recently modified run in exp/ is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_curves.png"),
        help="Output file path (default: training_curves.png)",
    )
    args = parser.parse_args()

    if args.exp_dir is None:
        exp_root = Path("exp")
        runs = sorted(exp_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise FileNotFoundError("No runs found in exp/")
        args.exp_dir = runs[0]
        print(f"Auto-selected most recent run: {args.exp_dir}")

    plot_curves(args.exp_dir, args.output)


if __name__ == "__main__":
    main()
