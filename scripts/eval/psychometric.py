"""Evaluate threshold/slope RMSE with relevant or irrelevant auxiliary trajectories.

Conditions are no_aux, n{K}_relevant and n{K}_irrelevant for K=1..5.
Writes <prefix>_<condition>.pt and <prefix>_summary.csv under --out_dir.

Example:
    python scripts/eval/psychometric.py \\
        --pth_path <run_dir>/model/aae_Psychometric_d1_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --T 30 --T_aux 30 --M 500 --batch_size 64 --seed 0 --name_prefix imbue
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Support running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_utils.psychometric import (
    load_model_and_task,
    print_final_table,
    print_table,
    rollout_rmse,
    save_summary_csv,
    summarize_metrics,
)
from utils import set_seed


# (tag, n_src, prob_good_trajectory). prob_good is unused when n_src=0.
CONDITIONS = [
    ("no_aux", 0, 1.0),
    *[(f"n{k}_relevant", k, 1.0) for k in range(1, 6)],
    *[(f"n{k}_irrelevant", k, 0.0) for k in range(1, 6)],
]
COND_LOOKUP = {tag: (n_src, p_good) for tag, n_src, p_good in CONDITIONS}
CONDITIONS_DEFAULT = [
    "no_aux",
    "n1_relevant",
    "n5_relevant",
    "n1_irrelevant",
    "n5_irrelevant",
]
SEL = [0, 1]  # threshold, slope


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/psychometric")
    p.add_argument(
        "--name_prefix", default="imbue",
        help="Output filename prefix. Files: <prefix>_<condition>.pt",
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=CONDITIONS_DEFAULT, choices=list(COND_LOOKUP),
    )
    p.add_argument("--T", type=int, default=30, help="Adaptive trials per participant.")
    p.add_argument("--T_aux", type=int, default=30, help="Trials per previous instance.")
    p.add_argument("--M", type=int, default=500, help="Rollouts per condition.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--n_samples", type=int, default=1000,
        help="Posterior samples per step for the RMSE estimate.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def eval_condition(
    model, experiment, T, T_aux, M, batch_size,
    n_src, prob_good, sel, lower, upper, n_samples,
):
    n_steps = (M + batch_size - 1) // batch_size
    chunks = []
    for step in range(n_steps):
        bs = min(batch_size, M - step * batch_size)
        batch = experiment.sample_correlated_batch_eval(
            bs, n_source_tasks=n_src, T=T_aux, prob_good_trajectory=prob_good,
        )
        rmse_b = rollout_rmse(
            model, experiment, batch, T, sel, lower, upper, n_samples,
        )
        chunks.append(rmse_b.detach().cpu())
        joint_T = torch.sqrt((rmse_b[:, -1, :] ** 2).sum(-1)).mean().item()
        print(f"  step {step + 1}/{n_steps}  final joint RMSE = {joint_T:.3f}", flush=True)

    return summarize_metrics(chunks)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print(f"[load] cfg : {args.cfg_yaml}")
    print(f"[load] pth : {args.pth_path}")
    cfg = OmegaConf.load(args.cfg_yaml)
    OmegaConf.set_struct(cfg, False)

    if not bool(cfg.get("use_trajectories", False)):
        print(
            "[warn] cfg.use_trajectories=False: the checkpoint was not trained "
            "with previous-instance input and may not consume source_trajectories."
        )

    experiment, model = load_model_and_task(cfg, args.pth_path, device)

    sel = torch.tensor(SEL)
    n_sel = sel.numel()
    lower = experiment.range_min[sel].view(1, n_sel, 1)
    upper = experiment.range_max[sel].view(1, n_sel, 1)

    print(
        f"[eval] T={args.T}  T_aux={args.T_aux}  M={args.M}  batch_size={args.batch_size}  "
        f"n_samples={args.n_samples}"
    )
    print(f"[eval] conditions: {args.conditions}")
    print(f"[eval] master seed: {args.seed}  prefix: {args.name_prefix}")
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
            T=args.T,
            T_aux=args.T_aux,
            M=args.M,
            batch_size=args.batch_size,
            n_src=n_src,
            prob_good=p_good,
            sel=sel,
            lower=lower,
            upper=upper,
            n_samples=args.n_samples,
        )
        print(f"[done] {cond} in {time.time() - t0:.1f}s")
        print_table(metrics, header=f"{args.name_prefix} | {cond}")
        summary[cond] = metrics

        cond_path = os.path.join(args.out_dir, f"{args.name_prefix}_{cond}.pt")
        torch.save({
            **metrics,
            "T": int(args.T),
            "T_aux": int(args.T_aux),
            "M": int(args.M),
            "n_samples": int(args.n_samples),
            "condition": cond,
            "n_src": n_src,
            "prob_good": p_good,
            "seed": cond_seed,
            "method": args.name_prefix,
            "pth_path": args.pth_path,
            "cfg_yaml": args.cfg_yaml,
            "tag": f"{args.name_prefix} ({cond})",
        }, cond_path)
        print(f"[saved] {cond_path}")

    csv_path = os.path.join(args.out_dir, f"{args.name_prefix}_summary.csv")
    save_summary_csv(summary, csv_path)
    print(f"\n[saved] {csv_path}")

    print_final_table(summary)


if __name__ == "__main__":
    main()
