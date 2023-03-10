from datetime import timedelta
import logging
from pathlib import Path
import sys
from typing import List, Union

from lightning_lite.accelerators.cuda import num_cuda_devices
from pytorch_lightning.strategies import DDPStrategy

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
from calvin_agent.utils.utils import get_git_commit_hash, get_last_checkpoint, print_system_env_info
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

import hulc
import hulc.models.hulc as models_m
from hulc.utils.utils import initialize_pretrained_weights

logger = logging.getLogger(__name__)

import torch

import numpy as np

@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.

    Args:
        cfg: hydra config
    """
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed, workers=True)  # type: ignore
    device_id = 0
    data_module = hydra.utils.instantiate(cfg.datamodule, training_repo_root=Path(hulc.__file__).parents[1])
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.train_dataloader()["lang"]
    #dataset = dataloader.dataset.datasets["lang"]
    #device = torch.device(f"cuda:{device_id}")

    #import pdb;pdb.set_trace()

    checkpoint = Path('/iliad/u/manasis/hulc/checkpoints/HULC_ABCD_D/saved_models/HULC_ABCD_D.ckpt')
    model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(checkpoint.as_posix())

    hulc_latent_observations = np.empty((1013110, 32, 1024))
    hulc_actions = np.empty((1013110, 32, 7))
    hulc_language = np.empty((1013110, 32, 4096))

    for i, batch in enumerate(dataloader):
        perceptual_emb = model.perceptual_encoder(
                batch["rgb_obs"], batch["depth_obs"], batch["robot_obs"]
            )
        seq_len = perceptual_emb.shape[1]
        latent_goal = model.language_goal(batch["lang"])
        
        # ------------Plan Proposal------------ #
        pp_state = model.plan_proposal(perceptual_emb[:, 0], latent_goal)
        pp_dist = model.dist.get_dist(pp_state)
        sampled_plan_pp = model.dist.sample_latent_plan(pp_dist)
        sampled_plan_pp = sampled_plan_pp.unsqueeze(1).expand(-1, seq_len, -1)

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = model.plan_recognition(perceptual_emb)
        pr_dist = model.dist.get_dist(pr_state)
        sampled_plan_pr = pr_dist.rsample()  # sample from recognition net
        if model.dist.dist == "discrete":
            sampled_plan_pr = torch.flatten(sampled_plan_pr, start_dim=-2, end_dim=-1)
        sampled_plan_pr = sampled_plan_pr.unsqueeze(1).expand(-1, seq_len, -1)
        
        import pdb;pdb.set_trace()
    pass


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiate all training callbacks.

    Args:
        callbacks_cfg: DictConfig with all callback params

    Returns:
        List of instantiated callbacks.
    """
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule) -> Logger:
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
        # train_logger.watch(model)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger


def modify_argv_hydra() -> None:
    """
    To make hydra work with pytorch-lightning and ddp, we modify sys.argv for the child processes spawned with ddp.
    This is only used when NOT using slurm.
    """
    cwd = Path.cwd().as_posix()
    cwd = f'"{cwd}"'
    sys.argv = sys.argv[:1]
    sys.argv.extend(
        [
            f"hydra.run.dir={cwd}",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    overrides = OmegaConf.load(".hydra/overrides.yaml")
    for o in overrides:
        if "hydra/sweeper" in o:  # type: ignore
            continue

        if "hydra/launcher" in o:  # type: ignore
            continue

        sys.argv.append(o)  # type: ignore


def is_multi_gpu_training(devices: Union[int, str, ListConfig]) -> bool:
    """
    Check if training on multiple GPUs.
    See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#devices

     Args:
        devices: int, str or ListConfig specifying devices

    Returns:
        True if multi-gpu training (ddp), False otherwise.
    """
    num_gpu_available = num_cuda_devices()
    if isinstance(devices, int):
        return devices > 1 or (devices == -1 and num_gpu_available > 1)
    elif isinstance(devices, str) and devices == "auto":
        return num_gpu_available > 1
    elif isinstance(devices, str):
        return len(devices) > 1
    elif isinstance(devices, ListConfig):
        return len(devices) > 1
    else:
        raise ValueError


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
