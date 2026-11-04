"""Evaluate CES PCE with Student-t noise in auxiliary observations.

Auxiliary responses use
    eta = u * (U(z) - U(z')) + scale_mult * tau * u * (1 + ||z - z'||) * t,
    t ~ StudentT(df), followed by the clipped sigmoid.
Target observations and the PCE likelihood use CensoredSigmoidNormal.

Auxiliary parameters can be relevant or shuffled across the batch.
Writes per-condition .pt files and summary .csv/.pt files under --out_dir.

Example:
    python scripts/eval/ces_ood_noise.py \\
        --pth_path <run_dir>/model/aae_CES_d6_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --name_prefix ood_noise --K 5 --T 10 --source_T 10 \\
        --df 3 --scale_mult 3 --seed 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from attrdictionary import AttrDict

# Support running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_utils.ces import (
    load_eval_config,
    load_model_and_task,
    print_final_table,
    print_table,
    rollout_batch,
    sample_aux_batch,
    save_summary_csv,
    summarize_metrics,
)
from utils import compute_EIG_from_history, set_seed


# kind -> (relevance, noise_model)
KIND_SPEC = {
    "relevant": ("relevant", "gaussian"),
    "studentt": ("relevant", "studentt"),
    "irrelevant": ("irrelevant", "gaussian"),
    "studentt_irrelevant": ("irrelevant", "studentt"),
}
KINDS = ["no_aux", "relevant", "studentt", "irrelevant", "studentt_irrelevant"]


@torch.no_grad()
def rollout_one_batch(
    model, experiment, batch_size, T_eval, kind, K, source_T,
    df, scale_mult, correlation_std,
):
    if kind == "no_aux":
        batch = experiment.sample_batch(batch_size)
        src = AttrDict({"n_tasks": 0})
    elif kind in KIND_SPEC:
        relevance, noise_model = KIND_SPEC[kind]
        batch = sample_aux_batch(
            experiment,
            batch_size=batch_size,
            n_source_tasks=K,
            T=source_T,
            relevance=relevance,
            noise_model=noise_model,
            df=df,
            scale_mult=scale_mult,
            correlation_std=correlation_std,
        )
        src = batch.source_trajectories
    else:
        raise ValueError(f"unknown kind: {kind}")

    return rollout_batch(model, experiment, batch, src, T_eval)


@torch.no_grad()
def eval_condition(
    model, experiment, T_eval, L, M, batch_size, kind, K,
    source_T, df, scale_mult, correlation_std,
) -> AttrDict:
    n_steps = (M + batch_size - 1) // batch_size
    pce_list, nmc_list = [], []
    for step in range(n_steps):
        theta_0, x, y = rollout_one_batch(
            model, experiment,
            batch_size=batch_size,
            T_eval=T_eval,
            kind=kind,
            K=K,
            source_T=source_T,
            df=df,
            scale_mult=scale_mult,
            correlation_std=correlation_std,
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/ces")
    p.add_argument("--name_prefix", default="ood_noise")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    p.add_argument(
        "--df", type=float, default=3.0,
        help="Student-t degrees of freedom.",
    )
    p.add_argument(
        "--scale_mult", type=float, default=3.0,
        help="Auxiliary noise scale multiplier on tau.",
    )
    p.add_argument(
        "--correlation_std", type=float, default=0.1,
        help="Relevant-parameter perturbation std as a fraction of each range.",
    )
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_query", type=int, default=None)
    p.add_argument("--source_T", type=int, default=10)
    p.add_argument(
        "--no_filter", action="store_true",
        help="Bypass reliability gating at evaluation, retaining all K sources "
             "with the same model weights.",
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

    print(f"[load] cfg : {args.cfg_yaml}")
    print(f"[load] pth : {args.pth_path}")
    cfg = load_eval_config(args)

    ut = bool(cfg.get("use_trajectories", False))
    print(f"[cfg ] use_trajectories={ut}")

    experiment, model = load_model_and_task(cfg, args.pth_path, device)

    if args.no_filter:
        model.encoder.has_gating = False
        print(
            "[filter] BYPASSED (encoder.has_gating=False) — IMBUE w/o filter: "
            "all K sources kept ungated (same weights as full)."
        )

    experiment.n_query_init = cfg.n_query_final
    T_eval = int(cfg.T_final)
    L = int(cfg.L_final)
    M = int(cfg.M_final)
    batch_size = int(cfg.eval_batch_size_final)

    tau = float(experiment.noise_scale)
    print(
        f"[eval] T={T_eval}  L={L}  M={M}  batch_size={batch_size}  "
        f"n_query={cfg.n_query_final}  K={args.K}  source_T={args.source_T}"
    )
    print(f"[eval] kinds={args.kinds}  seed={args.seed}")
    print(
        f"[noise] train tau={tau}  ->  studentt aux: df={args.df}, "
        f"scale={args.scale_mult}*tau={args.scale_mult * tau}"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    conditions = [
        ("no_aux" if kind == "no_aux" else f"n{args.K}_{kind}", kind)
        for kind in args.kinds
    ]

    summary = {}
    for cond_idx, (tag, kind) in enumerate(conditions):
        cond_seed = args.seed + cond_idx
        set_seed(cond_seed)
        print(f"\n--- {tag}  (kind={kind}, K={args.K}, seed={cond_seed}) ---")

        t0 = time.time()
        metrics = eval_condition(
            model, experiment,
            T_eval=T_eval,
            L=L,
            M=M,
            batch_size=batch_size,
            kind=kind,
            K=args.K,
            source_T=args.source_T,
            df=args.df,
            scale_mult=args.scale_mult,
            correlation_std=args.correlation_std,
        )
        print(f"[done] {tag} in {time.time() - t0:.1f}s")
        print_table(metrics, header=f"{args.name_prefix}/{tag}")
        summary[tag] = metrics

        cond_path = os.path.join(args.out_dir, f"{args.name_prefix}_{tag}.pt")
        torch.save({
            **metrics,
            "T": T_eval,
            "L_final": L,
            "M_final": M,
            "n_query_final": int(cfg.n_query_final),
            "source_T": args.source_T,
            "K": args.K,
            "kind": kind,
            "df": args.df,
            "scale_mult": args.scale_mult,
            "tau": tau,
            "correlation_std": args.correlation_std,
            "no_filter": args.no_filter,
            "seed": cond_seed,
            "condition": tag,
            "pth_path": args.pth_path,
            "cfg_yaml": args.cfg_yaml,
            "tag": f"{args.name_prefix.upper()} ({tag})",
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
        "K": args.K,
        "df": args.df,
        "scale_mult": args.scale_mult,
        "tau": tau,
        "seed": args.seed,
        "conditions": [t for t, _ in conditions],
        "pth_path": args.pth_path,
        "cfg_yaml": args.cfg_yaml,
    }, summary_path)
    print(f"[saved] {summary_path}")

    print_final_table(summary)


if __name__ == "__main__":
    main()
