"""Evaluate per-step PCE/NMC under strong, weak, miscalibrated and uniform priors.

Writes <prefix>_<condition>.pt and <prefix>_summary.csv under --out_dir.

Example:
    python scripts/eval/location.py \\
        --pth_path <run_dir>/model/aae_Location_d2_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --conditions strong weak miscalibrated uniform --seed 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine

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


CONDITIONS_DEFAULT = ["strong", "weak", "miscalibrated", "uniform"]


def build_pool(pool_type, B, N, dim, lo, hi, device):
    """Sample candidate designs with shape [B, N, dim] in [lo, hi]."""
    span = hi - lo
    if pool_type == "uniform":
        return lo + torch.rand(B, N, dim, device=device) * span
    if pool_type == "sobol":
        seeds = torch.randint(0, 2**31 - 1, (B,)).tolist()
        pools = [
            SobolEngine(dimension=dim, scramble=True, seed=s).draw(N)
            for s in seeds
        ]
        return lo + torch.stack(pools, dim=0).to(device) * span
    raise ValueError(f"unknown pool_type: {pool_type}")


def _override_query_pool(experiment, batch, batch_size, pool_type):
    """Replace query designs and sample their outcomes under the true theta."""
    N = experiment.n_query_init
    device = batch.target_theta.device
    pool_x = build_pool(
        pool_type,
        B=batch_size,
        N=N,
        dim=experiment.dim_x,
        lo=float(experiment.range_min),
        hi=float(experiment.range_max),
        device=device,
    )
    theta_kd = batch.target_theta.reshape(
        batch_size, experiment.K, experiment.dim_x
    )  # [B, K, D]
    theta_bcast = theta_kd.unsqueeze(1).expand(
        batch_size, N, experiment.K, experiment.dim_x
    )  # [B, N, K, D]
    batch.query_x = pool_x
    batch.query_y = experiment.forward(
        experiment.unnormalise_design(pool_x), theta_bcast
    )  # [B, N, 1]
    return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/location")
    p.add_argument(
        "--name_prefix", default="imbue",
        help="Output filename prefix. Files: <prefix>_<condition>.pt",
    )
    p.add_argument(
        "--conditions", nargs="+", default=CONDITIONS_DEFAULT,
        choices=CONDITIONS_DEFAULT,
    )
    p.add_argument(
        "--T", type=int, default=None,
        help="Eval acquisition horizon. Default cfg.T_final - n_context_init.",
    )
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_query", type=int, default=None)
    p.add_argument(
        "--pool_type", choices=["uniform", "sobol"], default="uniform",
        help="Use the sampled candidate pool (uniform) or replace it with "
             "scrambled Sobol points.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def eval_condition(
    model, experiment, T_eval, L, M, batch_size, prior_type, pool_type="uniform",
):
    n_steps = (M + batch_size - 1) // batch_size
    pce_list, nmc_list = [], []
    for step in range(n_steps):
        batch = experiment.sample_batch_eval(batch_size, prior_type=prior_type)
        if pool_type != "uniform":
            batch = _override_query_pool(experiment, batch, batch_size, pool_type)
        theta_0, x, y = rollout_batch(model, experiment, batch, T_eval)
        pce, nmc = compute_EIG_from_history(
            experiment, theta_0, x, y, L=L, batch_size=batch_size, stepwise=True,
        )
        pce_list.append(pce)
        nmc_list.append(nmc)
        print(
            f"  step {step + 1}/{n_steps}  PCE_T={pce.mean(0)[-1].item():.3f}",
            flush=True,
        )

    return summarize_metrics(pce_list, nmc_list)


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
    coeff_gat = float(cfg.get("coeff_gat", 1.0))
    prob_good_prior = float(cfg.task.get("prob_good_prior", -1))
    print(
        f"[cfg ] prior_injection={pi}  gating={gating}  "
        f"coeff_gat={coeff_gat}  task.prob_good_prior={prob_good_prior}"
    )
    if not pi:
        print(
            "[warn] cfg.task.prior_injection=False — sample_batch_eval will "
            "ignore the prior conditioning, so the four prior conditions "
            "give the same curve."
        )

    experiment, model = load_model_and_task(cfg, args.pth_path, device)

    n_ctx_init = int(cfg.task.get("n_context_init", 0))
    T_eval = args.T if args.T is not None else int(cfg.T_final) - n_ctx_init
    L = int(cfg.L_final)
    M = int(cfg.M_final)
    batch_size = int(cfg.eval_batch_size_final)

    print(
        f"[eval] T={T_eval}  L={L}  M={M}  batch_size={batch_size}  "
        f"n_query={cfg.n_query_final}"
    )
    print(f"[eval] pool_type={args.pool_type}")
    print(f"[eval] conditions: {args.conditions}")
    print(f"[eval] master seed: {args.seed}  prefix: {args.name_prefix}")

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
            prior_type=cond,
            pool_type=args.pool_type,
        )
        print(f"[done] {cond} in {time.time() - t0:.1f}s")
        print_table(metrics, header=f"{args.name_prefix} | {cond}")
        summary[cond] = metrics

        cond_path = os.path.join(args.out_dir, f"{args.name_prefix}_{cond}.pt")
        torch.save({
            **metrics,
            "T": T_eval,
            "L_final": L,
            "M_final": M,
            "n_query_final": int(cfg.n_query_final),
            "pool_type": args.pool_type,
            "seed": cond_seed,
            "condition": cond,
            "name_prefix": args.name_prefix,
            "coeff_gat": coeff_gat,
            "prob_good_prior": prob_good_prior,
            "gating": gating,
            "pth_path": args.pth_path,
            "cfg_yaml": args.cfg_yaml,
            "tag": f"{args.name_prefix} ({cond})",
        }, cond_path)
        print(f"[saved] {cond_path}")

    csv_path = os.path.join(args.out_dir, f"{args.name_prefix}_summary.csv")
    save_summary_csv(summary, csv_path)
    print(f"\n[saved] {csv_path}")

    print(f"\n{'condition':>15s} | {'PCE@T':>10s}  ({'SE':>6s})")
    print("-" * 40)
    for cond, m in summary.items():
        print(
            f"{cond:>15s} | {m['pce_mean'][-1].item():>10.4f}  "
            f"({m['pce_std'][-1].item():>6.4f})"
        )


if __name__ == "__main__":
    main()
