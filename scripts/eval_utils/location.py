"""Shared helpers for Location Finding prior evaluation."""

import csv

import hydra
import numpy as np
import torch
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
    experiment.n_query_init = cfg.n_query_final

    return experiment, model


@torch.no_grad()
def rollout_batch(model, experiment, batch, T_eval):
    """Return theta and the acquired history, excluding the initial context."""
    batch_size = batch.target_theta.shape[0]
    n_init = batch.context_x.shape[1]
    src = None

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
    return {
        "pce_mean": pce_all.mean(dim=0).detach().cpu(),
        "pce_std": (pce_all.std(dim=0) / np.sqrt(actual_M)).detach().cpu(),
        "nmc_mean": nmc_all.mean(dim=0).detach().cpu(),
        "nmc_std": (nmc_all.std(dim=0) / np.sqrt(actual_M)).detach().cpu(),
        "actual_M": actual_M,
    }


def print_table(metrics, header):
    print(f"\n=== Per-step PCE / NMC — {header} ===")
    print(f"{'t':>4} | {'PCE_mean':>10} {'PCE_se':>10} | {'NMC_mean':>10} {'NMC_se':>10}")
    print("-" * 60)
    pm, ps = metrics["pce_mean"], metrics["pce_std"]
    nm, ns = metrics["nmc_mean"], metrics["nmc_std"]
    for t in range(pm.numel()):
        print(
            f"{t + 1:>4} | {pm[t].item():>10.4f} {ps[t].item():>10.4f} "
            f"| {nm[t].item():>10.4f} {ns[t].item():>10.4f}"
        )
    print(f"final  PCE = {pm[-1].item():.4f} +- {ps[-1].item():.4f}")


def save_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "t", "pce_mean", "pce_se", "nmc_mean", "nmc_se"])
        for cond, m in summary.items():
            for t in range(m["pce_mean"].numel()):
                writer.writerow([
                    cond,
                    t + 1,
                    f"{m['pce_mean'][t].item():.6f}",
                    f"{m['pce_std'][t].item():.6f}",
                    f"{m['nmc_mean'][t].item():.6f}",
                    f"{m['nmc_std'][t].item():.6f}",
                ])
