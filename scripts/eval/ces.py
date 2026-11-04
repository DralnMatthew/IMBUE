"""Evaluate CES PCE/NMC with relevant or irrelevant auxiliary trajectories.

Conditions are no_aux, n{K}_relevant and n{K}_irrelevant for K=1..5.
Auxiliary trajectories require a checkpoint trained with use_trajectories=True.
Writes per-condition .pt files and summary .csv/.pt files under --out_dir.

Example:
    python scripts/eval/ces.py \\
        --pth_path <run_dir>/model/aae_CES_d6_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --name_prefix ours --T 20 --seed 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from attrdictionary import AttrDict
from torch.quasirandom import SobolEngine

# Support running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_utils.ces import (
    load_eval_config,
    load_model_and_task,
    print_final_table,
    print_table,
    rollout_batch,
    save_summary_csv,
    summarize_metrics,
)
from utils import compute_EIG_from_history, set_seed


# (tag, n_src, prob_good_trajectory). prob_good is unused when n_src=0.
CONDITIONS = [
    ("no_aux", 0, 1.0),
    *[(f"n{k}_relevant", k, 1.0) for k in range(1, 6)],
    *[(f"n{k}_irrelevant", k, 0.0) for k in range(1, 6)],
]
COND_LOOKUP = {tag: (n_src, p_good) for tag, n_src, p_good in CONDITIONS}


def build_pool(pool_type, B, N, dim, scale, device):
    """Sample candidate designs with shape [B, N, dim] in [0, scale]."""
    if pool_type == "uniform":
        return torch.rand(B, N, dim, device=device) * scale
    if pool_type == "sobol":
        seeds = torch.randint(0, 2**31 - 1, (B,)).tolist()
        pools = [
            SobolEngine(dimension=dim, scramble=True, seed=s).draw(N)
            for s in seeds
        ]
        return torch.stack(pools, dim=0).to(device) * scale
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
        scale=experiment.design_scale,
        device=device,
    )
    theta = batch.target_theta.reshape(
        batch_size, experiment.n_target_theta
    ).unsqueeze(-2)  # [B, 1, 5]
    batch.query_x = experiment.normalise_design(pool_x)
    batch.query_y = experiment.forward(pool_x, theta)  # [B, N, 1]
    return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/ces")
    p.add_argument(
        "--name_prefix", default="ours",
        help="Output filename prefix. Files: <prefix>_<condition>.pt",
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=[c[0] for c in CONDITIONS],
        choices=[c[0] for c in CONDITIONS],
    )
    p.add_argument(
        "--T", type=int, default=None,
        help="Eval acquisition horizon. Default cfg.T_final.",
    )
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument(
        "--batch_size", type=int, default=None,
        help="Eval batch size. Default cfg.eval_batch_size_final.",
    )
    p.add_argument(
        "--n_query", type=int, default=None,
        help="Number of candidate query basket pairs. Default cfg.n_query_final.",
    )
    p.add_argument(
        "--source_T", type=int, default=10,
        help="Number of (x,y) pairs per auxiliary experiment.",
    )
    p.add_argument(
        "--pool_type", choices=["uniform", "sobol"], default="uniform",
        help="Use the sampled candidate pool (uniform) or replace it with "
             "scrambled Sobol points.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def rollout_one_batch(
    model, experiment, batch_size, T_eval, n_src, prob_good, source_T,
    pool_type="uniform",
):
    """Sample target and auxiliary data, then collect the acquired history."""
    if n_src == 0:
        batch = experiment.sample_batch(batch_size)
        # The embedder uses n_tasks=0 to skip source trajectories.
        src = AttrDict({"n_tasks": 0})
    else:
        batch = experiment.sample_correlated_batch_eval(
            batch_size,
            n_source_tasks=n_src,
            T=source_T,
            prob_good_trajectory=prob_good,
        )
        src = batch.source_trajectories

    if pool_type != "uniform":
        batch = _override_query_pool(experiment, batch, batch_size, pool_type)

    return rollout_batch(model, experiment, batch, src, T_eval)


@torch.no_grad()
def eval_condition(
    model, experiment, T_eval, L, M, batch_size,
    n_src, prob_good, source_T, pool_type="uniform",
) -> AttrDict:
    n_steps = (M + batch_size - 1) // batch_size
    pce_list, nmc_list = [], []
    for step in range(n_steps):
        theta_0, x, y = rollout_one_batch(
            model, experiment,
            batch_size=batch_size,
            T_eval=T_eval,
            n_src=n_src,
            prob_good=prob_good,
            source_T=source_T,
            pool_type=pool_type,
        )
        pce, nmc = compute_EIG_from_history(
            experiment, theta_0, x, y, L=L, batch_size=batch_size, stepwise=True,
        )
        pce_list.append(pce)
        nmc_list.append(nmc)
        per_step = pce.mean(dim=0).detach().cpu().tolist()
        per_step_str = "  ".join(f"t{t + 1}={v:.2f}" for t, v in enumerate(per_step))
        print(f"  step {step + 1}/{n_steps} | {per_step_str}", flush=True)

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
    ut = bool(cfg.get("use_trajectories", False))
    print(f"[cfg ] use_trajectories={ut}  prior_injection={pi}")
    if not ut:
        print(
            "[warn] cfg.use_trajectories=False. The model was not trained with "
            "trajectory input — passing source_trajectories may be a no-op."
        )

    experiment, model = load_model_and_task(cfg, args.pth_path, device)

    experiment.n_query_init = cfg.n_query_final
    T_eval = int(cfg.T_final)
    L = int(cfg.L_final)
    M = int(cfg.M_final)
    batch_size = int(cfg.eval_batch_size_final)

    print(
        f"[eval] T={T_eval}  L={L}  M={M}  batch_size={batch_size}  "
        f"n_query={cfg.n_query_final}  source_T={args.source_T}"
    )
    print(f"[eval] pool_type={args.pool_type}")
    print(f"[eval] conditions: {args.conditions}")
    print(f"[eval] master seed: {args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for cond_idx, cond in enumerate(args.conditions):
        n_src, p_good = COND_LOOKUP[cond]
        cond_seed = args.seed + cond_idx
        set_seed(cond_seed)
        print(f"\n--- {cond}  (n_src={n_src}, prob_good={p_good}, seed={cond_seed}) ---")

        t0 = time.time()
        metrics = eval_condition(
            model, experiment,
            T_eval=T_eval,
            L=L,
            M=M,
            batch_size=batch_size,
            n_src=n_src,
            prob_good=p_good,
            source_T=args.source_T,
            pool_type=args.pool_type,
        )
        print(f"[done] {cond} in {time.time() - t0:.1f}s")
        print_table(metrics, header=f"{args.name_prefix}/{cond}")
        summary[cond] = metrics

        cond_path = os.path.join(args.out_dir, f"{args.name_prefix}_{cond}.pt")
        torch.save({
            **metrics,
            "T": T_eval,
            "L_final": L,
            "M_final": M,
            "n_query_final": int(cfg.n_query_final),
            "pool_type": args.pool_type,
            "source_T": args.source_T,
            "n_src": n_src,
            "prob_good": p_good,
            "seed": cond_seed,
            "condition": cond,
            "pth_path": args.pth_path,
            "cfg_yaml": args.cfg_yaml,
            "tag": f"{args.name_prefix.upper()} ({cond})",
        }, cond_path)
        print(f"[saved] {cond_path}")

    csv_path = os.path.join(args.out_dir, f"{args.name_prefix}_summary.csv")
    save_summary_csv(summary, csv_path)
    print(f"\n[saved] {csv_path}")

    summary_path = os.path.join(args.out_dir, f"{args.name_prefix}_summary.pt")
    torch.save({
        "summary": summary,
        "T": T_eval,
        "L_final": L,
        "M_final": M,
        "n_query_final": int(cfg.n_query_final),
        "source_T": args.source_T,
        "seed": args.seed,
        "conditions": list(args.conditions),
        "pth_path": args.pth_path,
        "cfg_yaml": args.cfg_yaml,
    }, summary_path)
    print(f"[saved] {summary_path}")

    print_final_table(summary)


if __name__ == "__main__":
    main()
