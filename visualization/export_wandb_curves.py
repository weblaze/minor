"""Export real loss curves from wandb run history (replaces the old simulated plots).

Usage:
  python visualization/export_wandb_curves.py --entity <your-wandb-username> [--group 02_latent_diffusion_clap]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent / "loss_curves"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", type=str, required=True, help="wandb entity (username/team)")
    parser.add_argument("--project", type=str, default="abstraction")
    parser.add_argument("--group", type=str, default=None, help="only runs from this approach group")
    parser.add_argument("--metric", type=str, default="loss")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    filters = {"group": args.group} if args.group else {}
    runs = api.runs(f"{args.entity}/{args.project}", filters=filters)
    runs = [r for r in runs if r.state in ("finished", "running")]
    if not runs:
        print("No runs found — check --entity/--group and that training logged to wandb.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plotted = 0
    for run in runs:
        history = run.history(keys=["epoch", args.metric])
        if history.empty or args.metric not in history:
            continue
        plt.plot(history["epoch"], history[args.metric], label=run.name, alpha=0.85)
        plotted += 1

    if not plotted:
        print(f"No runs contained metric '{args.metric}'.")
        return

    plt.xlabel("epoch")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} — {args.group or 'all runs'}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    suffix = args.group or "all"
    out_path = OUTPUT_DIR / f"{args.metric}_{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plotted} run curves to {out_path}")


if __name__ == "__main__":
    main()
