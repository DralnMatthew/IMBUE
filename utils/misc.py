import torch
from torch import optim
from torch.optim import lr_scheduler
import os
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_dict(model, dir, name="dtnp.pth"):
    file_path = os.path.join(dir, 'model')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, name)
    torch.save(model.state_dict(), file_path)
    return file_path


def _resolve_model_file(dir, name):
    """Resolve either an explicit checkpoint path or output_dir/model/name."""
    name = os.fspath(name)
    candidates = []

    # Prefer explicit paths.
    if os.path.isabs(name) or os.path.dirname(name):
        candidates.append(name)

    candidates.append(os.path.join(dir, 'model', name))

    # Check bare filenames after output_dir/model.
    if not os.path.isabs(name):
        candidates.append(name)

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Checkpoint file not found. Tried: " +
        ", ".join(unique_candidates)
    )


def load_state_dict(model, dir, name="dtnp.pth"):
    file_path = _resolve_model_file(dir, name)
    model.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage, weights_only=True))
    return model

def load_ckpt_state_dict(model, dir, name):
    ckpt_path = _resolve_model_file(dir, name)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    model.load_state_dict(ckpt["model"])

    return model

def save_checkpoint(cfg, model, optimizer, scheduler, epoch, with_epoch=False):
    """ Save checkpoint

    Args:
        with_epoch (bool, optional): whether to add epoch as suffix. e.g. ckptname_1000.tar
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        # random generator
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
    }

    if with_epoch:
        checkpoint_name = f"{cfg.checkpoint_name.split('.')[0]}_{epoch}.tar"
    else:
        checkpoint_name = cfg.checkpoint_name

    checkpoint_path = os.path.join(cfg.output_dir, checkpoint_name)
    torch.save(ckpt, checkpoint_path)


def load_checkpoint(cfg, model, optimizer, scheduler, ckpt_path=None, check_layerwise=True):
    """Restore model and RNG state; return (epoch, optimizer, scheduler)."""
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.get_default_device(), weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt["epoch"]

    optimizer = ckpt["optimizer"]
    scheduler = ckpt["scheduler"]
    # Restore RNG states
    if not isinstance(ckpt['rng_state'], torch.ByteTensor):
        ckpt['rng_state'] = torch.ByteTensor(ckpt['rng_state'].cpu())
    torch.set_rng_state(ckpt['rng_state'])

    if torch.cuda.is_available() and ckpt['cuda_rng_state'] is not None:
        if not isinstance(ckpt['cuda_rng_state'], torch.ByteTensor):
            ckpt['cuda_rng_state'] = torch.ByteTensor(ckpt['cuda_rng_state'].cpu())
        torch.cuda.set_rng_state(ckpt['cuda_rng_state'])

    np.random.set_state(ckpt['numpy_rng_state'])
    random.setstate(ckpt['random_rng_state'])

    return epoch, optimizer, scheduler


def set_layerwise_lr(cfg, model, epoch=0):
    """Set layerwise learning rates."""
    if epoch < cfg.burning_epoch:
        optimizer = getattr(optim, cfg.optimizer)(
            model.parameters(),
            lr=cfg.lr,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch
        )    
    else:
        shared_params, predictor_params = [], []
        for name, param in model.named_parameters():
            if 'predictor' not in name:
                shared_params.append(param)
            else:
                predictor_params.append(param)

        optimizer = getattr(optim, cfg.optimizer)(
            [
                {"params": shared_params, "lr": cfg.lr / cfg.lr_scale_predictor},
                {"params": predictor_params, "lr": cfg.lr},
                # {"params": shared_params, "lr": cfg.lr / 5},
                # {"params": predictor_params, "lr": cfg.lr},
            ],
            lr=cfg.lr,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch - epoch
        )
    return optimizer, scheduler
