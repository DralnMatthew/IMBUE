"""Helpers for GP active-learning RMSE evaluation."""

import csv

import hydra
import numpy as np
import torch
from attrdictionary import AttrDict

from model import BaseTransformer


# Relevant analytic instances use a * f(x) + b + noise.
AFFINE_A_RANGE = (0.5, 1.5)  # multiplicative scale
AFFINE_B_RANGE = (-0.5, 0.5)  # additive bias, as a fraction of the output std


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


def make_synthetic_batch(gp_task, batch_size, n_good, n_bad, T_src):
    """Sample synthetic tasks; relevant/irrelevant counts hold in expectation."""
    n_src = n_good + n_bad
    prob_good = (n_good / n_src) if n_src > 0 else 1.0
    return gp_task.sample_correlated_batch(
        batch_size=batch_size, n_source_tasks=n_src, T=T_src,
        prob_good_trajectory=prob_good, bias_scale_range=[0.2, 2.0], mix_modes=True,
    )


def make_benchmark_batch(bench_task, gp_task, name, batch_size, n_good, n_bad, T_src):
    """Build analytic episodes with exact relevant/irrelevant source counts.

    Relevant sources use affine perturbations; irrelevant sources use the GP prior.
    """
    n_src = n_good + n_bad
    batch = bench_task.sample_batch(batch_size, name, mode="data")
    if n_src == 0:
        batch.source_trajectories = AttrDict({"n_tasks": 0})
        return batch

    dim_x = bench_task.dim_x
    x_src = bench_task.sample_data(batch_size * n_src, T_src)  # [B*K, T, dx]

    y_clean = bench_task.forward(x_src, name)  # [B*K, T, 1]
    y_std = y_clean.std()
    a = (
        AFFINE_A_RANGE[0]
        + (AFFINE_A_RANGE[1] - AFFINE_A_RANGE[0]) * torch.rand(batch_size * n_src, 1, 1)
    )
    b = (
        AFFINE_B_RANGE[0]
        + (AFFINE_B_RANGE[1] - AFFINE_B_RANGE[0]) * torch.rand(batch_size * n_src, 1, 1)
    ) * y_std
    y_good = a * y_clean + b + float(bench_task.noise_scale) * torch.randn_like(y_clean)

    bad_theta = gp_task.sample_theta(batch_size * n_src).squeeze(-1)
    y_bad, _ = gp_task.generate_gp_data(x_src, bad_theta)  # [B*K, T, 1]

    # Shuffle the n_good relevant slots within each episode.
    perm = torch.stack([torch.randperm(n_src) for _ in range(batch_size)])  # [B, K]
    is_good = perm < n_good
    y_src = torch.where(is_good.reshape(-1, 1, 1).expand_as(y_good), y_good, y_bad)

    batch.source_trajectories = AttrDict({
        "designs": x_src.reshape(batch_size, n_src, T_src, dim_x),
        "observations": y_src.reshape(batch_size, n_src, T_src, 1),
        "n_tasks": n_src,
        "reliability_labels": is_good.float().unsqueeze(-1),
    })
    return batch


@torch.no_grad()
def rollout_rmse(model, experiment, batch, T):
    """Return posterior-mean RMSE on held-out points [T, B], before acquisition."""
    B = batch.context_x.shape[0]
    rmse = torch.zeros(T, B)
    for t in range(T):
        outs = model.forward(batch, batch.source_trajectories)
        post = outs.posterior_out
        mean = torch.sum(post.mixture_means * post.mixture_weights, dim=-1)  # [B, n_target]
        sq_err = (batch.target_y.squeeze(-1) - mean) ** 2
        rmse[t] = torch.sqrt(sq_err.mean(dim=-1))
        batch = experiment.update_batch_with_query(batch, outs.design_out.idx)
    return rmse


def summarize_metrics(rmse):
    """Mean and standard error across episodes at each step."""
    mean = rmse.mean(dim=1)
    se = rmse.std(dim=1) / np.sqrt(rmse.shape[1])
    return mean, se


def print_final_table(final, benchmarks, conditions):
    tags = [c for c, _, _ in conditions]
    print(f"\n{'benchmark':<18s}" + "".join(f"{t:>18s}" for t in tags))
    print("-" * (18 + 18 * len(tags)))
    for name, title in benchmarks:
        cells = [
            f"{final[(name, t)][0]:.4f} +- {final[(name, t)][1]:.4f}"
            for t in tags
        ]
        print(f"{title:<18s}" + "".join(f"{c:>18s}" for c in cells))


def save_summary_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "condition", "t", "rmse_mean", "rmse_se"])
        for name, tag, t, m, s in rows:
            w.writerow([name, tag, t, f"{m:.6f}", f"{s:.6f}"])
