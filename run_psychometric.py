import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import os
import numpy as np
import time
import shutil

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from model import BaseTransformer
from utils.eval import eval_boed
from utils import create_logger, set_seed, save_state_dict, compute_ll, load_checkpoint, save_checkpoint
from utils.misc import set_layerwise_lr


def train(cfg, logger, model, experiment, batch_size: int, T: int, max_epoch: int, verbose: int = 10, eval_interval: int = 10000):
    """Train the design policy and posterior model."""
    optimizer, scheduler = set_layerwise_lr(cfg, model)

    coeff_gat = cfg.coeff_gat

    # Load from checkpoint if specified
    if cfg.load_checkpoint is True:
        start_epoch, optimizer, scheduler = load_checkpoint(cfg, model, optimizer, scheduler, cfg.load_path)
    else:
        start_epoch = 0

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    losses = []

    training_times = []

    for epoch in range(start_epoch, max_epoch):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        # Data
        if cfg.use_trajectories:
            n_source_tasks_sample = np.random.randint(cfg.n_source_tasks_range[0], cfg.n_source_tasks_range[1] + 1)
            batch = experiment.sample_correlated_batch(
                batch_size=batch_size, 
                n_source_tasks=n_source_tasks_sample,
                T=cfg.T,
                prob_good_trajectory=cfg.task.prob_good_trajectory
            )
            source_trajectories = batch.source_trajectories

        else:
            batch = experiment.sample_batch(batch_size)
            source_trajectories = None
        
        is_gating_active = (cfg.task.prior_injection or (cfg.use_trajectories and n_source_tasks_sample > 0)) and cfg.gating

        outs = model.forward(batch, source_trajectories)
        target_ll = compute_ll(batch.target_theta, 
                               outs.posterior_out.mixture_means,
                               outs.posterior_out.mixture_stds,
                               outs.posterior_out.mixture_weights)
        current_nll = - target_ll.mean(dim=-1) # [B]                      

        log_probs = []
        rewards = []
        all_nlls = [current_nll]
        gating_logits = []

        if is_gating_active and hasattr(outs, 'gating_logits') and outs.gating_logits is not None:
            gating_logits.append(outs.gating_logits)

        for t in range(T):
            design_out = outs.design_out
            log_probs.append(design_out.log_prob) # [T, B]

            batch = experiment.update_batch_with_query(batch, design_out.idx)

            next_outs = model.forward(batch, source_trajectories)

            if is_gating_active and hasattr(next_outs, 'gating_logits') and next_outs.gating_logits is not None:
                gating_logits.append(next_outs.gating_logits)

            next_target_ll = compute_ll(batch.target_theta,
                                        next_outs.posterior_out.mixture_means,
                                        next_outs.posterior_out.mixture_stds,
                                        next_outs.posterior_out.mixture_weights)
            next_nll = - next_target_ll.mean(dim=-1)
            ll_gain = torch.clamp(current_nll - next_nll, min=0.0).detach()
            ll_gain *= (cfg.gamma ** t)
            rewards.append(ll_gain)

            current_nll = next_nll
            all_nlls.append(current_nll)
            outs = next_outs

        log_probs = torch.stack(log_probs, dim=1)  # [B, T]
        rewards = torch.stack(rewards, dim=1)      # [B, T]
        all_nlls = torch.stack(all_nlls, dim=1)    # [B, T+1]

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        design_loss = - torch.mean(log_probs * rewards)
        predict_loss = torch.mean(all_nlls)

        gating_loss = 0.0

        if is_gating_active and len(gating_logits) > 0:
            gating_logits_tensor = torch.stack(gating_logits, dim=1) # [B, n_steps, n_src/n_priors, 1]
            n_steps = gating_logits_tensor.shape[1]

            if cfg.task.prior_injection:
                raw_labels = batch.prior_labels # [B, n_priors, 1]
            else:
                raw_labels = source_trajectories.reliability_labels # [B, n_src, 1]

            raw_labels_exp = raw_labels.unsqueeze(1).expand(-1, n_steps, -1, -1)
            binary_labels = (raw_labels_exp > 0.5).float()
            gating_loss = F.binary_cross_entropy_with_logits(
                gating_logits_tensor, 
                binary_labels, 
            )

            if cfg.wandb.use_wandb:
                wandb.log({"gating_loss": gating_loss}, step=epoch)

        if epoch < cfg.burning_epoch:
            loss = predict_loss + coeff_gat * gating_loss
            loss.backward()

        else:
            loss = design_loss * cfg.alpha + predict_loss * cfg.beta + coeff_gat * gating_loss
            loss.backward()

        losses.append(loss.item())

        # Gradient clipping
        if cfg.clip_grads:
            clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type="inf")

        # Setting different learning rates for shared layers after the burning stage
        if epoch == cfg.burning_epoch:
            optimizer, scheduler = set_layerwise_lr(cfg, model, epoch)

            # reset query size
            experiment.n_query_init = cfg.task.n_query_init

            # save the model
            logger.info(f"""Model has been saved at {save_state_dict(model, cfg.output_dir, f'{cfg.file_name.split(".")[0]}_burning.pth')}""")

        optimizer.step()
        scheduler.step()

        end_time = time.time()
        training_times.append(end_time - start_time)

        if cfg.wandb.use_wandb:
            wandb.log({"loss": loss, "likelihood": -predict_loss, "design_loss": design_loss}, step=epoch)

        if epoch % verbose == 0:
            logger.info(f"Epoch: {epoch}, loss: {losses[-1]:.4f}, T: {T}, likelihood: {-predict_loss}, design_loss: {design_loss}, predict_loss: {predict_loss}")

        # next_epoch = epoch + 1
        # if cfg.checkpoint and next_epoch % cfg.checkpoint == 0:
        #     save_checkpoint(cfg, model, optimizer, scheduler, next_epoch, with_epoch=True)
        
    total_time = sum(training_times)
    average_time = np.mean(training_times[cfg.burning_epoch:])
    std_time = np.std(training_times[cfg.burning_epoch:])
    logger.info(f"Total training time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours), average time per epoch: {average_time:.2f}+-{std_time:.2f} seconds")



@hydra.main(version_base=None, config_path="config", config_name="train_bed")
def main(cfg):
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name=cfg.task.name, prior_injection=cfg.task.prior_injection)

    # Setting device
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    torch.set_default_device(cfg.device)
    if cfg.device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Setting random seed
    if cfg.fix_seed:
        set_seed(cfg.seed)
    else:
        cfg.seed = torch.random.seed()

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    logger.info("Running with config:\n{}".format(OmegaConf.to_yaml(cfg)))

    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.output_dir,
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    # Data
    experiment = hydra.utils.instantiate(cfg.task)
    logger.info(experiment)

    # Model
    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)
    model = BaseTransformer(embedder, encoder, head, cfg.use_trajectories)
    logger.info(model)

    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=10)

    try:
        # Train
        train(cfg, logger, model, experiment, cfg.batch_size, cfg.T, cfg.max_epoch, verbose=cfg.verbose, eval_interval=cfg.eval_interval)

        # Save
        logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")

        experiment.n_query_init = cfg.n_query_final
        
    finally:
        if cfg.wandb.use_wandb:
            wandb.finish()
        local_path = os.path.join(cfg.output_dir, "wandb")
        if os.path.exists(local_path):
            shutil.rmtree(local_path)


if __name__ == '__main__':
    main()
