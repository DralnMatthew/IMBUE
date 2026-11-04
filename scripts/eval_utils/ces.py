"""Shared helpers for CES auxiliary-data evaluation."""

import csv

import hydra
import numpy as np
import torch
from attrdictionary import AttrDict
from omegaconf import OmegaConf

from model import BaseTransformer


def load_eval_config(args):
    cfg = OmegaConf.load(args.cfg_yaml)
    OmegaConf.set_struct(cfg, False)

    if args.n_query is not None:
        cfg.n_query_final = args.n_query
    if args.batch_size is not None:
        cfg.eval_batch_size_final = args.batch_size
    if args.L is not None:
        cfg.L_final = args.L
    if args.M is not None:
        cfg.M_final = args.M
    if args.T is not None:
        cfg.T_final = args.T

    return cfg


def load_model_and_task(cfg, pth_path, device):
    experiment = hydra.utils.instantiate(cfg.task)
    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)
    model = BaseTransformer(embedder, encoder, head, cfg.use_trajectories)

    state = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return experiment, model


@torch.no_grad()
def sample_biased_relevant_batch(
    experiment, batch_size, n_source_tasks, T,
    bias_scale=0.3, sel_prob=0.5, correlation_std=0.1,
    fixed_subset=None, fixed_signs=None,
):
    """Sample auxiliary parameters with a bias shared across the K sources.

    The biased dimensions and signs are sampled per batch element, unless
    fixed_subset and fixed_signs supply length-D vectors.
    """
    D = experiment.n_target_theta
    batch = experiment.sample_batch(batch_size)
    target_theta = batch.target_theta.reshape(batch_size, D)
    target_exp = target_theta.unsqueeze(1).expand(-1, n_source_tasks, -1)  # [B, K, D]

    range_min = experiment.range_min.view(1, 1, D)
    range_max = experiment.range_max.view(1, 1, D)
    range_span = range_max - range_min

    if fixed_subset is not None:
        mask = (
            torch.as_tensor(fixed_subset, dtype=torch.bool)
            .view(1, D).expand(batch_size, -1)
        )
    else:
        mask = torch.bernoulli(torch.full((batch_size, D), sel_prob)).bool()
        empty = ~mask.any(dim=-1)
        if empty.any():
            forced = torch.randint(0, D, (int(empty.sum()),))
            mask[empty.nonzero(as_tuple=True)[0], forced] = True

    if fixed_signs is not None:
        sign = (
            torch.as_tensor(fixed_signs, dtype=torch.float32)
            .view(1, D).expand(batch_size, -1)
        )
    else:
        sign = torch.randint(0, 2, (batch_size, D)).float() * 2.0 - 1.0

    bias = sign * mask.float() * bias_scale * range_span.view(1, D)
    bias_exp = bias.unsqueeze(1)  # [B, 1, D], shared across sources

    noise = torch.randn_like(target_exp) * correlation_std * range_span
    src_theta = torch.clamp(target_exp + bias_exp + noise, min=range_min, max=range_max)
    src_theta[..., 0] = torch.clamp(src_theta[..., 0], 0.01, 1.0)
    # Keep alpha positive and on the simplex.
    src_theta[..., 1:4] = torch.abs(src_theta[..., 1:4]) + 1e-6
    src_theta[..., 1:4] = (
        src_theta[..., 1:4] / src_theta[..., 1:4].sum(dim=-1, keepdim=True)
    )

    b1 = (
        torch.rand(batch_size, n_source_tasks, T, experiment.basket_dim)
        * experiment.design_scale
    )
    b2 = (
        torch.rand(batch_size, n_source_tasks, T, experiment.basket_dim)
        * experiment.design_scale
    )
    src_x = torch.cat([b1, b2], dim=-1)  # [B, K, T, 6]

    flat_x = src_x.reshape(-1, experiment.dim_x)
    flat_theta = src_theta.unsqueeze(2).expand(-1, -1, T, -1).reshape(-1, D)
    src_y = experiment.forward(flat_x, flat_theta).reshape(batch_size, n_source_tasks, T, 1)

    # Retain the training labels in the auxiliary batch.
    norm_diff = (src_theta - target_exp) / range_span
    reliability = torch.exp(-5.0 * torch.sum(norm_diff ** 2, dim=-1, keepdim=True))
    batch.source_trajectories = AttrDict({
        "designs": src_x,
        "observations": src_y,
        "theta": src_theta,
        "n_tasks": n_source_tasks,
        "reliability_labels": reliability,
    })
    return batch


@torch.no_grad()
def studentt_aux_forward(experiment, xi, theta, df=3.0, scale_mult=3.0):
    """Sample CES responses with scaled Student-t noise and a clipped sigmoid."""
    rho = theta[..., 0:1]
    alpha = theta[..., 1:4]
    log_u = theta[..., 4:5]
    u = torch.exp(log_u)

    xi = torch.clamp(xi, min=0.01, max=100.0)
    b1 = xi[..., :experiment.basket_dim]
    b2 = xi[..., experiment.basket_dim:]
    u1 = experiment.utility(b1, rho, alpha)
    u2 = experiment.utility(b2, rho, alpha)
    mu_eta = (u1 - u2) * u

    basket_dist = torch.norm(b1 - b2, dim=-1, p=2, keepdim=True)
    scale = (1.0 + basket_dist) * (scale_mult * experiment.noise_scale) * u
    t = torch.distributions.StudentT(df).rsample(mu_eta.shape)
    eta = mu_eta + scale * t

    y = torch.sigmoid(eta)
    y = torch.clamp(y, min=experiment.epsilon, max=1.0 - experiment.epsilon)
    return y


@torch.no_grad()
def sample_aux_batch(
    experiment, batch_size, n_source_tasks, T,
    relevance="relevant", noise_model="gaussian", df=3.0,
    scale_mult=3.0, correlation_std=0.1,
):
    """Sample auxiliary parameters and Gaussian or Student-t observations.

    Relevant parameters perturb theta; irrelevant ones shuffle it across the
    batch. Target observations always use the task's original likelihood.
    """
    D = experiment.n_target_theta
    batch = experiment.sample_batch(batch_size)
    target = batch.target_theta.reshape(batch_size, D)
    target_exp = target.unsqueeze(1).expand(-1, n_source_tasks, -1)  # [B, K, D]

    rmin = experiment.range_min.view(1, 1, D)
    rmax = experiment.range_max.view(1, 1, D)
    span = rmax - rmin

    if relevance == "relevant":
        noise = torch.randn_like(target_exp) * correlation_std * span
        src_theta = torch.clamp(target_exp + noise, min=rmin, max=rmax)
        src_theta[..., 0] = torch.clamp(src_theta[..., 0], 0.01, 1.0)
        src_theta[..., 1:4] = torch.abs(src_theta[..., 1:4]) + 1e-6
        src_theta[..., 1:4] = (
            src_theta[..., 1:4] / src_theta[..., 1:4].sum(dim=-1, keepdim=True)
        )
    elif relevance == "irrelevant":
        src_theta = torch.stack(
            [target[torch.randperm(batch_size)] for _ in range(n_source_tasks)],
            dim=1,
        )
    else:
        raise ValueError(f"unknown relevance: {relevance}")

    b1 = (
        torch.rand(batch_size, n_source_tasks, T, experiment.basket_dim)
        * experiment.design_scale
    )
    b2 = (
        torch.rand(batch_size, n_source_tasks, T, experiment.basket_dim)
        * experiment.design_scale
    )
    src_x = torch.cat([b1, b2], dim=-1)  # [B, K, T, 6]

    flat_x = src_x.reshape(-1, experiment.dim_x)
    flat_theta = src_theta.unsqueeze(2).expand(-1, -1, T, -1).reshape(-1, D)
    if noise_model == "gaussian":
        flat_y = experiment.forward(flat_x, flat_theta)
    elif noise_model == "studentt":
        flat_y = studentt_aux_forward(experiment, flat_x, flat_theta, df, scale_mult)
    else:
        raise ValueError(f"unknown noise_model: {noise_model}")
    src_y = flat_y.reshape(batch_size, n_source_tasks, T, 1)

    norm_diff = (src_theta - target_exp) / span
    reliability = torch.exp(-5.0 * torch.sum(norm_diff ** 2, dim=-1, keepdim=True))
    batch.source_trajectories = AttrDict({
        "designs": src_x,
        "observations": src_y,
        "theta": src_theta,
        "n_tasks": n_source_tasks,
        "reliability_labels": reliability,
    })
    return batch


@torch.no_grad()
def rollout_batch(model, experiment, batch, src, T_eval):
    """Return theta and the acquired history, excluding the initial context."""
    batch_size = batch.target_theta.shape[0]
    n_init = batch.context_x.shape[1]

    if experiment.with_query:
        for _ in range(T_eval):
            idx = model.forward(batch, src).design_out.idx
            batch = experiment.update_batch_with_query(batch, idx)
    else:
        theta_shape = experiment.get_theta_shape(batch_size)
        for _ in range(T_eval):
            xi = model.forward(batch, src).design_out.xi
            y = experiment.forward(
                xi.squeeze(1),
                batch.target_theta.reshape(theta_shape),
            ).unsqueeze(1)
            batch = experiment.update_batch_without_query(batch, xi, y)

    theta_0 = batch.target_theta.reshape(experiment.get_theta_shape(batch_size))
    x = experiment.unnormalise_design(batch.context_x[:, n_init:])
    y = batch.context_y[:, n_init:]
    return theta_0, x, y


def summarize_metrics(pce_list, nmc_list):
    """Aggregate batches; the legacy *_std keys store standard errors."""
    pce_all = torch.cat(pce_list, dim=0)
    nmc_all = torch.cat(nmc_list, dim=0)
    actual_M = pce_all.shape[0]
    return AttrDict(
        pce_mean=pce_all.mean(dim=0).detach().cpu(),
        pce_std=(pce_all.std(dim=0) / np.sqrt(actual_M)).detach().cpu(),
        nmc_mean=nmc_all.mean(dim=0).detach().cpu(),
        nmc_std=(nmc_all.std(dim=0) / np.sqrt(actual_M)).detach().cpu(),
        actual_M=actual_M,
    )


def print_table(metrics, header):
    print(f"\n=== Per-step PCE / NMC — {header} ===")
    print(f"{'t':>4} | {'PCE_mean':>10} {'PCE_se':>10} | {'NMC_mean':>10} {'NMC_se':>10}")
    print("-" * 60)
    for t in range(metrics.pce_mean.numel()):
        print(
            f"{t + 1:>4} | "
            f"{metrics.pce_mean[t].item():>10.4f} {metrics.pce_std[t].item():>10.4f} | "
            f"{metrics.nmc_mean[t].item():>10.4f} {metrics.nmc_std[t].item():>10.4f}"
        )
    print(
        f"final  PCE = {metrics.pce_mean[-1].item():.4f} "
        f"+- {metrics.pce_std[-1].item():.4f}"
    )


def print_final_table(summary):
    print(f"\n{'condition':>15s} | {'PCE@T':>10s}  ({'SE':>6s})")
    print("-" * 40)
    for cond, m in summary.items():
        print(
            f"{cond:>15s} | {m.pce_mean[-1].item():>10.4f}  "
            f"({m.pce_std[-1].item():>6.4f})"
        )


def save_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "t", "pce_mean", "pce_se", "nmc_mean", "nmc_se"])
        for cond, m in summary.items():
            for t in range(m.pce_mean.numel()):
                writer.writerow([
                    cond,
                    t + 1,
                    f"{m.pce_mean[t].item():.6f}",
                    f"{m.pce_std[t].item():.6f}",
                    f"{m.nmc_mean[t].item():.6f}",
                    f"{m.nmc_std[t].item():.6f}",
                ])
