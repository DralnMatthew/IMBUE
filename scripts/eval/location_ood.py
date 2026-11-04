"""Evaluate Location Finding PCE/NMC with misspecified prior support.

Truncated priors place mass on one side of theta*, separated by a bin margin.
Boundary priors place mass near both edges and leave the interior empty.
Both support uniform or Gaussian density within these regions.

Writes <prefix>_<condition>.pt and <prefix>_summary.csv under --out_dir.

Example:
    python scripts/eval/location_ood.py \\
        --pth_path <run_dir>/model/aae_Location_d2_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --conditions truncated boundary --seed 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Support running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_utils.location import (
    load_eval_config,
    load_model_and_task,
    print_table,
    rollout_batch,
    save_summary_csv,
    summarize_metrics,
)
from utils import compute_EIG_from_history, set_seed


CONDITIONS = ["truncated", "boundary"]


def _bin_of(theta_flat, edges, num_bins):
    """Bin index containing each theta value. theta_flat [B, n], edges [n, nb+1]."""
    idx = torch.sum(theta_flat.unsqueeze(-1) >= edges.unsqueeze(0), dim=-1) - 1
    return idx.clamp(0, num_bins - 1)


def build_truncated_prior(bin_star, num_bins, margin_bins, density):
    """Build a prior on one side of theta*, offset by margin_bins.

    bin_star: [B, n]. Returns normalized histograms [B, n, num_bins].
    """
    B, n = bin_star.shape
    device = bin_star.device
    bins = torch.arange(num_bins, device=device).view(1, 1, num_bins).float()

    # Put support on the side away from theta*.
    left_half = bin_star < (num_bins // 2)
    lo = torch.where(left_half, bin_star + margin_bins, torch.zeros_like(bin_star))
    hi = torch.where(
        left_half, torch.full_like(bin_star, num_bins - 1), bin_star - margin_bins
    )
    lo = lo.clamp(0, num_bins - 1).view(B, n, 1).float()
    hi = hi.clamp(0, num_bins - 1).view(B, n, 1).float()

    mask = (bins >= lo) & (bins <= hi)
    if density == "uniform":
        w = mask.float()
    elif density == "gaussian":
        center = (lo + hi) / 2.0
        std = ((hi - lo).clamp(min=1.0)) / 4.0
        w = torch.exp(-0.5 * ((bins - center) / std) ** 2) * mask.float()
    else:
        raise ValueError(f"unknown density: {density}")
    return w / (w.sum(dim=-1, keepdim=True) + 1e-12)


def build_boundary_prior(num_bins, edge_frac, density, B, n, device):
    """theta*-independent prior: mass in the two boundary bands, zero interior."""
    k = max(1, round(edge_frac * num_bins))
    bins = torch.arange(num_bins, device=device).float()
    mask = (bins < k) | (bins >= num_bins - k)
    if density == "uniform":
        w = mask.float()
    elif density == "gaussian":
        s = max(1.0, k / 2.0)
        w = (
            torch.exp(-0.5 * (bins / s) ** 2)
            + torch.exp(-0.5 * ((bins - (num_bins - 1)) / s) ** 2)
        ) * mask.float()
    else:
        raise ValueError(f"unknown density: {density}")
    w = w / (w.sum() + 1e-12)
    return w.view(1, 1, num_bins).expand(B, n, num_bins).contiguous()


@torch.no_grad()
def make_ood_batch(
    experiment, batch_size, cond, num_bins, margin_bins, edge_frac, density,
):
    """Replace the uniform prior and return (batch, excl_frac).

    excl_frac is the fraction of theta coordinates with bin mass <= 1e-9.
    """
    batch = experiment.sample_batch_eval(batch_size, prior_type="uniform")
    n_theta = experiment.n_target_theta
    theta_flat = batch.target_theta.reshape(batch_size, n_theta)
    edges = experiment.theta_prior.bin_edges  # [n_theta, num_bins+1]
    bin_star = _bin_of(theta_flat, edges, num_bins)  # [B, n_theta]

    if cond == "truncated":
        priors = build_truncated_prior(bin_star, num_bins, margin_bins, density)
    elif cond == "boundary":
        priors = build_boundary_prior(
            num_bins,
            edge_frac=edge_frac,
            density=density,
            B=batch_size,
            n=n_theta,
            device=theta_flat.device,
        )
    else:
        raise ValueError(f"not an OOD cond: {cond}")

    star_mass = torch.gather(priors, 2, bin_star.unsqueeze(-1)).squeeze(-1)  # [B, n]
    excl_frac = (star_mass <= 1e-9).float().mean().item()

    batch.priors = priors.reshape(batch_size, -1, num_bins)
    return batch, excl_frac


@torch.no_grad()
def eval_condition(
    model, experiment, T_eval, L, M, batch_size, cond, num_bins,
    margin_bins, edge_frac, density,
):
    n_steps = (M + batch_size - 1) // batch_size
    pce_list, nmc_list, excl_list = [], [], []
    for step in range(n_steps):
        batch, excl = make_ood_batch(
            experiment,
            batch_size=batch_size,
            cond=cond,
            num_bins=num_bins,
            margin_bins=margin_bins,
            edge_frac=edge_frac,
            density=density,
        )
        theta_0, x, y = rollout_batch(model, experiment, batch, T_eval)
        pce, nmc = compute_EIG_from_history(
            experiment, theta_0, x, y, L=L, batch_size=batch_size, stepwise=True,
        )
        pce_list.append(pce)
        nmc_list.append(nmc)
        excl_list.append(excl)
        print(
            f"  step {step + 1}/{n_steps}  PCE_T={pce.mean(0)[-1].item():.3f}",
            flush=True,
        )

    metrics = summarize_metrics(pce_list, nmc_list)
    metrics["excl_frac"] = float(np.mean(excl_list))
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/location")
    p.add_argument("--name_prefix", default="lf_ood_prior")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument(
        "--margin_frac", type=float, default=0.03,
        help="Truncated: gap around theta* as a fraction of the range.",
    )
    p.add_argument(
        "--edge_frac", type=float, default=0.1,
        help="Boundary: width of each boundary band (fraction of range).",
    )
    p.add_argument(
        "--density", choices=["uniform", "gaussian"], default="uniform",
        help="Density used inside the misspecified support.",
    )
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_query", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print(f"[load] cfg : {args.cfg_yaml}")
    print(f"[load] pth : {args.pth_path}")
    cfg = load_eval_config(args)

    pi = bool(cfg.task.get("prior_injection", False))
    gating = bool(cfg.get("gating", False))
    print(f"[cfg ] prior_injection={pi}  gating={gating}")
    if not pi:
        print("[warn] cfg.task.prior_injection=False — injected priors are ignored.")

    experiment, model = load_model_and_task(cfg, args.pth_path, device)

    num_bins = int(cfg.task.num_bins)
    margin_bins = max(1, round(args.margin_frac * num_bins))

    n_ctx_init = int(cfg.task.get("n_context_init", 0))
    T_eval = args.T if args.T is not None else int(cfg.T_final) - n_ctx_init
    L = int(cfg.L_final)
    M = int(cfg.M_final)
    batch_size = int(cfg.eval_batch_size_final)

    print(
        f"[eval] T={T_eval}  L={L}  M={M}  batch_size={batch_size}  "
        f"n_query={cfg.n_query_final}  num_bins={num_bins}"
    )
    print(
        f"[ood ] margin_bins={margin_bins} (frac={args.margin_frac})  "
        f"edge_frac={args.edge_frac}  density={args.density}"
    )
    print(f"[eval] conditions: {args.conditions}  seed={args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for cond_idx, cond in enumerate(args.conditions):
        cond_seed = args.seed + cond_idx
        set_seed(cond_seed)
        print(f"\n--- {cond}  (seed={cond_seed}) ---")

        t0 = time.time()
        metrics = eval_condition(
            model, experiment,
            T_eval=T_eval,
            L=L,
            M=M,
            batch_size=batch_size,
            cond=cond,
            num_bins=num_bins,
            margin_bins=margin_bins,
            edge_frac=args.edge_frac,
            density=args.density,
        )
        print(
            f"[done] {cond} in {time.time() - t0:.1f}s"
            f"  | theta*-excluded frac = {metrics['excl_frac']:.3f}"
        )
        print_table(metrics, header=f"{args.name_prefix} | {cond}")
        summary[cond] = metrics

        cond_path = os.path.join(args.out_dir, f"{args.name_prefix}_{cond}.pt")
        torch.save({
            **metrics,
            "T": T_eval,
            "L_final": L,
            "M_final": M,
            "n_query_final": int(cfg.n_query_final),
            "seed": cond_seed,
            "condition": cond,
            "name_prefix": args.name_prefix,
            "num_bins": num_bins,
            "margin_bins": margin_bins,
            "edge_frac": args.edge_frac,
            "density": args.density,
            "gating": gating,
            "pth_path": args.pth_path,
            "cfg_yaml": args.cfg_yaml,
            "tag": f"{args.name_prefix} ({cond})",
        }, cond_path)
        print(f"[saved] {cond_path}")

    csv_path = os.path.join(args.out_dir, f"{args.name_prefix}_summary.csv")
    save_summary_csv(summary, csv_path)
    print(f"\n[saved] {csv_path}")

    print(f"\n{'condition':>15s} | {'PCE@T':>10s}  ({'SE':>6s}) | {'excl%':>6s}")
    print("-" * 50)
    for cond, m in summary.items():
        print(
            f"{cond:>15s} | {m['pce_mean'][-1].item():>10.4f}  "
            f"({m['pce_std'][-1].item():>6.4f}) | {100 * m['excl_frac']:>5.0f}%"
        )


if __name__ == "__main__":
    main()
