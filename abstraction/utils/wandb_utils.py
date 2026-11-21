from datetime import datetime

import wandb


def init_wandb(config, approach, component):
    """Start a wandb run grouped by approach with a unique, readable name.

    Example: approach="03_byol_a_encoder", component="train"
    -> run "03_byol_a_encoder/train/0611-1432" in group "03_byol_a_encoder".
    """
    stamp = datetime.now().strftime("%m%d-%H%M")
    return wandb.init(
        project="abstraction",
        group=approach,
        job_type=component,
        name=f"{approach}/{component}/{stamp}",
        config=config,
        anonymous="allow",
    )
