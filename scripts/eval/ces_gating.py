"""Log CES gating scores under relevant and OOD auxiliary data.

Per-step metrics, averaged over batch and sources:
    alpha_t = mean(sigmoid(logit))
    keep_t = mean(logit >= 0)
Steps without gating, including an empty initial context, yield NaN.

Conditions: relevant, biased, irrelevant and studentt.
Writes <prefix>.pt and <prefix>.csv under --out_dir.

Example:
    python scripts/eval/ces_gating.py \\
        --pth_path <run_dir>/model/aae_CES_d6_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --K 5 --T 10 --M 2000 --seed 0
"""

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from attrdictionary import AttrDict
from omegaconf import OmegaConf

# Support running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_utils.ces import (
    load_model_and_task,
    sample_aux_batch,
    sample_biased_relevant_batch,
)
from utils import set_seed


@torch.no_grad()
def rollout_gating(model, experiment, batch, src, T_eval):
    """Average gating scores over batch and sources; skipped steps yield NaN."""
    alpha_t, keep_t = [], []
    for _ in range(T_eval):
        outs = model.forward(batch, src)
        logits = outs.gating_logits  # [B, n_src, 1] or None
        if logits is not None:
            alpha_t.append(torch.sigmoid(logits).mean().item())
            keep_t.append((logits >= 0).float().mean().item())
        else:
            alpha_t.append(float("nan"))
            keep_t.append(float("nan"))
        idx = outs.design_out.idx
        batch = experiment.update_batch_with_query(batch, idx)
    return alpha_t, keep_t


@torch.no_grad()
def eval_condition(
    model, experiment, kind, K, T_eval, M, batch_size, source_T,
    bias_scale, sel_prob, correlation_std, df, scale_mult,
):
    n_steps = (M + batch_size - 1) // batch_size
    alphas, keeps = [], []  # each: [n_steps, T_eval]
    for step in range(n_steps):
        if kind == "relevant":
            batch = experiment.sample_correlated_batch(
                batch_size, n_source_tasks=K, T=source_T,
                prob_good_trajectory=1.0, correlation_std=correlation_std,
            )
        elif kind == "biased":
            batch = sample_biased_relevant_batch(
                experiment,
                batch_size=batch_size,
                n_source_tasks=K,
                T=source_T,
                bias_scale=bias_scale,
                sel_prob=sel_prob,
                correlation_std=correlation_std,
            )
        elif kind == "irrelevant":
            batch = experiment.sample_correlated_batch(
                batch_size, n_source_tasks=K, T=source_T,
                prob_good_trajectory=0.0, correlation_std=correlation_std,
            )
        elif kind == "studentt":
            batch = sample_aux_batch(
                experiment,
                batch_size=batch_size,
                n_source_tasks=K,
                T=source_T,
                relevance="relevant",
                noise_model="studentt",
                df=df,
                scale_mult=scale_mult,
                correlation_std=correlation_std,
            )
        else:
            raise ValueError(kind)
        src = batch.source_trajectories
        a, k = rollout_gating(model, experiment, batch, src, T_eval)
        alphas.append(a)
        keeps.append(k)
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(
                f"  [{kind}] step {step + 1}/{n_steps} | {format_trace(a)}",
                flush=True,
            )

    alphas = np.array(alphas, dtype=float)  # [n_steps, T]
    keeps = np.array(keeps, dtype=float)
    # Standard errors are computed over batch means.
    # Ignore all-NaN warnings at the initial empty-context step.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        alpha_mean = np.nanmean(alphas, axis=0)
        alpha_se = np.nanstd(alphas, axis=0) / np.sqrt(alphas.shape[0])
        keep_mean = np.nanmean(keeps, axis=0)
    return AttrDict(
        alpha_mean=alpha_mean,
        alpha_se=alpha_se,
        keep_mean=keep_mean,
        n_rollouts=alphas.shape[0] * batch_size,
    )


def format_trace(values):
    return "  ".join(
        f"t{t + 1}={v:.3f}" if not np.isnan(v) else f"t{t + 1}=  -  "
        for t, v in enumerate(values)
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/ces")
    p.add_argument("--name_prefix", default="ces_ood_alpha")
    p.add_argument(
        "--kinds", nargs="+", default=["relevant", "biased", "irrelevant"],
        choices=["relevant", "biased", "irrelevant", "studentt"],
    )
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--T", type=int, default=10, help="Acquisition horizon.")
    p.add_argument("--M", type=int, default=2000, help="Total rollouts.")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--source_T", type=int, default=10)
    p.add_argument("--bias_scale", type=float, default=0.3)
    p.add_argument("--sel_prob", type=float, default=0.5)
    p.add_argument("--correlation_std", type=float, default=0.1)
    p.add_argument("--df", type=float, default=3.0, help="Student-t df (studentt kind).")
    p.add_argument(
        "--scale_mult", type=float, default=3.0,
        help="Aux noise scale x tau (studentt kind).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    print(f"[load] cfg={args.cfg_yaml}  pth={args.pth_path}")
    cfg = OmegaConf.load(args.cfg_yaml)
    OmegaConf.set_struct(cfg, False)
    if not bool(cfg.get("gating", False)):
        print(
            "[warn] cfg.gating is not True — this checkpoint has no filter, so "
            "gating_logits will be None and alpha_t undefined."
        )

    experiment, model = load_model_and_task(cfg, args.pth_path, device)
    experiment.n_query_init = int(cfg.n_query_final)

    print(
        f"[eval] K={args.K}  T={args.T}  M={args.M}  batch_size={args.batch_size}  "
        f"source_T={args.source_T}  seed={args.seed}"
    )
    gating_mode = cfg.encoder.get("gating_mode", "n/a") if "encoder" in cfg else "n/a"
    print(f"[eval] gating={bool(cfg.get('gating', False))}  gating_mode={gating_mode}")

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for ci, kind in enumerate(args.kinds):
        set_seed(args.seed + ci)
        print(f"\n--- gating trace: {kind} (K={args.K}, seed={args.seed + ci}) ---")
        m = eval_condition(
            model, experiment,
            kind=kind,
            K=args.K,
            T_eval=args.T,
            M=args.M,
            batch_size=args.batch_size,
            source_T=args.source_T,
            bias_scale=args.bias_scale,
            sel_prob=args.sel_prob,
            correlation_std=args.correlation_std,
            df=args.df,
            scale_mult=args.scale_mult,
        )
        summary[kind] = m
        print(f"[{kind}] alpha_t = {format_trace(m.alpha_mean)}")
        print(f"[{kind}] keep_t  = {format_trace(m.keep_mean)}")

    csv_path = os.path.join(args.out_dir, f"{args.name_prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "t", "alpha_mean", "alpha_se", "keep_mean"])
        for kind, m in summary.items():
            for t in range(len(m.alpha_mean)):
                w.writerow([
                    kind,
                    t + 1,
                    f"{m.alpha_mean[t]:.6f}",
                    f"{m.alpha_se[t]:.6f}",
                    f"{m.keep_mean[t]:.6f}",
                ])
    print(f"\n[saved] {csv_path}")

    saved_summary = {}
    for k, m in summary.items():
        saved_summary[k] = {}
        for kk, vv in m.items():
            if isinstance(vv, np.ndarray):
                vv = torch.tensor(vv)
            saved_summary[k][kk] = vv

    summary_path = os.path.join(args.out_dir, f"{args.name_prefix}.pt")
    torch.save({
        "summary": saved_summary,
        "K": args.K,
        "T": args.T,
        "M": args.M,
        "seed": args.seed,
        "bias_scale": args.bias_scale,
        "sel_prob": args.sel_prob,
        "correlation_std": args.correlation_std,
        "pth_path": args.pth_path,
        "cfg_yaml": args.cfg_yaml,
    }, summary_path)
    print(f"[saved] {summary_path}")

    order = [k for k in ["relevant", "studentt", "biased", "irrelevant"] if k in summary]
    if order:
        T = len(summary[order[0]].alpha_mean)
        header = f"\n{'t':>4} | " + " ".join(f"{'a_' + k[:4]:>9}" for k in order)
        print(header)
        print("-" * len(header))
        for t in range(T):
            vals = [summary[k].alpha_mean[t] for k in order]
            if any(np.isnan(v) for v in vals):
                continue
            print(f"{t + 1:>4} | " + " ".join(f"{v:>9.4f}" for v in vals))
        print(
            "keep-rate @ t=10: "
            + "  ".join(f"{k}={summary[k].keep_mean[-1]:.3f}" for k in order)
        )


if __name__ == "__main__":
    main()
