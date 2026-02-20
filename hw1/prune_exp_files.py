"""Keep only 10 evenly distributed checkpoints and their corresponding videos; delete the rest."""

import argparse
import re
from pathlib import Path


def parse_step(filename: str) -> int | None:
    m = re.search(r"_(\d+)[_.]", filename)
    return int(m.group(1)) if m else None


def select_evenly_distributed(steps: list[int], n: int) -> set[int]:
    """Pick n evenly distributed steps from a sorted list, always including first and last."""
    steps = sorted(set(steps))
    if len(steps) <= n:
        return set(steps)
    indices = [round(i * (len(steps) - 1) / (n - 1)) for i in range(n)]
    return {steps[i] for i in indices}


def prune(exp_dir: Path, keep: int = 10, dry_run: bool = False) -> None:
    checkpoint_dir = exp_dir / "wandb/files/checkpoints"
    video_dir = exp_dir / "wandb/files/media/videos/eval"

    # Collect all steps from checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pkl"))
    all_steps = sorted({parse_step(f.name) for f in checkpoints if parse_step(f.name) is not None})

    if not all_steps:
        print("No checkpoints found.")
        return

    keep_steps = select_evenly_distributed(all_steps, keep)
    print(f"Total steps: {len(all_steps)}, keeping {len(keep_steps)}: {sorted(keep_steps)}")

    # Prune checkpoints
    deleted_ckpt = 0
    for f in checkpoints:
        step = parse_step(f.name)
        if step not in keep_steps:
            print(f"  delete checkpoint: {f.name}")
            if not dry_run:
                f.unlink()
            deleted_ckpt += 1

    # Prune videos — match any rollout_ep* file whose step is not in keep_steps
    videos = list(video_dir.glob("rollout_ep*.mp4"))
    deleted_vid = 0
    for f in videos:
        step = parse_step(f.name)
        if step not in keep_steps:
            print(f"  delete video:      {f.name}")
            if not dry_run:
                f.unlink()
            deleted_vid += 1

    print(f"\n{'[dry-run] would delete' if dry_run else 'Deleted'} "
          f"{deleted_ckpt} checkpoints and {deleted_vid} videos.")
    print(f"Kept {len(checkpoints) - deleted_ckpt} checkpoints "
          f"and {len(videos) - deleted_vid} videos.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune checkpoints and videos to N evenly spaced steps.")
    parser.add_argument("exp_dir", type=Path, help="Path to the experiment directory (e.g. exp/flow)")
    parser.add_argument("--keep", type=int, default=10, help="Number of steps to keep (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    prune(args.exp_dir, keep=args.keep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
