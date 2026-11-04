"""Helpers for psychometric posterior RMSE evaluation."""

import csv

import hydra
import numpy as np
import torch
from torch.distributions import Categorical, Normal

from model import BaseTransformer


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


def _sample_gmm(means, stds, weights, num_samples):
    """Sample [N, num_samples] from N independent Gaussian mixtures [N, C]."""
    cat = Categorical(weights, validate_args=False)
    idx = cat.sample((num_samples,))
    sm = torch.gather(
        means.unsqueeze(0).expand(num_samples, -1, -1),
        2, idx.unsqueeze(-1),
    ).squeeze(-1)
    ss = torch.gather(
        stds.unsqueeze(0).expand(num_samples, -1, -1),
        2, idx.unsqueeze(-1),
    ).squeeze(-1)
    return Normal(sm, ss, validate_args=False).sample().t()


@torch.no_grad()
def rollout_rmse(model, experiment, batch, T, sel, lower, upper, n_samples):
    """Return posterior-sample RMSE [B, T, n_sel], before each acquisition."""
    B = batch.context_x.shape[0]
    n_sel = sel.numel()
    rmse = torch.zeros([B, T, n_sel])
    src = batch.source_trajectories

    for t in range(T):
        outs = model.forward(batch, src)
        post = outs.posterior_out
        n_comp = post.mixture_means.shape[-1]
        flat = _sample_gmm(
            post.mixture_means[:, sel, :].reshape(-1, n_comp),
            post.mixture_stds[:, sel, :].reshape(-1, n_comp),
            post.mixture_weights[:, sel, :].reshape(-1, n_comp),
            num_samples=n_samples,
        )
        samples = flat.view(B, n_sel, n_samples)
        samples = torch.clamp(
            samples, min=lower.expand_as(samples), max=upper.expand_as(samples),
        )
        target = batch.target_all[:, sel, :].expand(-1, -1, n_samples)
        rmse[:, t, :] = torch.sqrt(((samples - target) ** 2).mean(dim=2))

        if experiment.with_query:
            batch = experiment.update_batch_with_query(batch, outs.design_out.idx)
        else:
            xi = outs.design_out.xi
            theta_shape = experiment.get_theta_shape(B)
            y = experiment.forward(
                xi.squeeze(1), batch.target_theta.reshape(theta_shape),
            ).unsqueeze(1)
            batch = experiment.update_batch_without_query(batch, xi, y)

    return rmse


def summarize_metrics(chunks):
    """Combine rollout batches and report per-parameter and joint RMSE."""
    rmse_all = torch.cat(chunks, dim=0)  # [M, T, 2]
    actual_M = rmse_all.shape[0]
    joint = torch.sqrt((rmse_all ** 2).sum(dim=-1))  # [M, T]
    return {
        "rmse": rmse_all,
        "rmse_mean": rmse_all.mean(dim=0),
        "rmse_se": rmse_all.std(dim=0) / np.sqrt(actual_M),
        "rmse_combined_mean": joint.mean(dim=0),
        "rmse_combined_se": joint.std(dim=0) / np.sqrt(actual_M),
        "actual_M": actual_M,
    }


def print_table(m, header):
    print(f"\n=== {header}: per-step RMSE ===")
    print(
        f"{'t':>4} | {'θ mean':>8} {'θ se':>8} | {'β mean':>8} {'β se':>8} "
        f"| {'joint mean':>10} {'joint se':>8}"
    )
    print("-" * 72)
    for t in range(m["rmse_mean"].shape[0]):
        print(
            f"{t + 1:>4} | "
            f"{m['rmse_mean'][t, 0].item():>8.4f} {m['rmse_se'][t, 0].item():>8.4f} | "
            f"{m['rmse_mean'][t, 1].item():>8.4f} {m['rmse_se'][t, 1].item():>8.4f} | "
            f"{m['rmse_combined_mean'][t].item():>10.4f} {m['rmse_combined_se'][t].item():>8.4f}"
        )
    print(
        f"final  joint = {m['rmse_combined_mean'][-1].item():.4f} "
        f"± {m['rmse_combined_se'][-1].item():.4f}"
    )


def print_final_table(summary):
    print(f"\n{'condition':>15s} | {'joint RMSE@T':>12s}  ({'SE':>6s})")
    print("-" * 42)
    for cond, m in summary.items():
        print(
            f"{cond:>15s} | {m['rmse_combined_mean'][-1].item():>12.4f}  "
            f"({m['rmse_combined_se'][-1].item():>6.4f})"
        )


def save_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "t",
            "rmse_threshold_mean", "rmse_threshold_se",
            "rmse_slope_mean", "rmse_slope_se",
            "rmse_combined_mean", "rmse_combined_se",
        ])
        for cond, m in summary.items():
            for t in range(m["rmse_mean"].shape[0]):
                writer.writerow([
                    cond,
                    t + 1,
                    f"{m['rmse_mean'][t, 0].item():.6f}",
                    f"{m['rmse_se'][t, 0].item():.6f}",
                    f"{m['rmse_mean'][t, 1].item():.6f}",
                    f"{m['rmse_se'][t, 1].item():.6f}",
                    f"{m['rmse_combined_mean'][t].item():.6f}",
                    f"{m['rmse_combined_se'][t].item():.6f}",
                ])
