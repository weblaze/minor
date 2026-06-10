from datetime import datetime

import wandb


def init_wandb(config, approach, component):
    """Start a wandb run grouped by approach with a unique, readable name.

    Example: approach="02_latent_diffusion_clap", component="train"
    -> run "02_latent_diffusion_clap/train/0611-1432" in group "02_latent_diffusion_clap".
    """
    stamp = datetime.now().strftime("%m%d-%H%M")
    return wandb.init(
        project="abstraction",
        group=approach,
        job_type=component,
        name=f"{approach}/{component}/{stamp}",
        config=config,
    )
