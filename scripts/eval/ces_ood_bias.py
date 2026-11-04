"""Evaluate CES PCE with shared bias in auxiliary parameters.

For each target theta*, auxiliary parameters are
    theta^(k) = Pi(theta* + b * 1_S + eta_k),  eta_k ~ N(0, Sigma_rel)
    b_l = s_l * bias_scale * (r_l^max - r_l^min),  s_l in {-1, +1}

The subset S and signed bias are shared across K instances for the rollout.
Pi clips parameters to the task bounds, clips rho to [0.01, 1], and projects
alpha onto the simplex. Designs and likelihood follow CESTask.

Conditions: no_aux, n{K}_relevant, n{K}_biased and n{K}_irrelevant.
Writes per-condition .pt files and summary .csv/.pt files under --out_dir.

Example:
    python scripts/eval/ces_ood_bias.py \\
        --pth_path <run_dir>/model/aae_CES_d6_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --name_prefix ood_bias --K 5 --T 10 --source_T 10 \\
        --bias_scale 0.3 --seed 0
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
    sample_biased_relevant_batch,
    save_summary_csv,
    summarize_metrics,
)
from utils import compute_EIG_from_history, set_seed


KINDS = ["no_aux", "relevant", "biased", "irrelevant"]


@torch.no_grad()
def rollout_one_batch(
    model, experiment, batch_size, T_eval, kind, K, source_T,
    bias_scale, sel_prob, correlation_std, fixed_subset, fixed_signs,
):
    """Sample auxiliary data for the condition and collect the acquired history."""
    if kind == "no_aux":
        batch = experiment.sample_batch(batch_size)
        src = AttrDict({"n_tasks": 0})
    elif kind == "relevant":
        batch = experiment.sample_correlated_batch(
            batch_size, n_source_tasks=K, T=source_T,
            prob_good_trajectory=1.0, correlation_std=correlation_std,
        )
        src = batch.source_trajectories
    elif kind == "biased":
        batch = sample_biased_relevant_batch(
            experiment,
            batch_size=batch_size,
            n_source_tasks=K,
            T=source_T,
            bias_scale=bias_scale,
            sel_prob=sel_prob,
            correlation_std=correlation_std,
            fixed_subset=fixed_subset,
            fixed_signs=fixed_signs,
        )
        src = batch.source_trajectories
    elif kind == "irrelevant":
        batch = experiment.sample_correlated_batch(
            batch_size, n_source_tasks=K, T=source_T,
            prob_good_trajectory=0.0, correlation_std=correlation_std,
        )
        src = batch.source_trajectories
    else:
        raise ValueError(f"unknown kind: {kind}")

    return rollout_batch(model, experiment, batch, src, T_eval)


@torch.no_grad()
def eval_condition(
    model, experiment, T_eval, L, M, batch_size, kind, K,
    source_T, bias_scale, sel_prob, correlation_std, fixed_subset, fixed_signs,
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
            bias_scale=bias_scale,
            sel_prob=sel_prob,
            correlation_std=correlation_std,
            fixed_subset=fixed_subset,
            fixed_signs=fixed_signs,
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


def _parse_vec(s, D, cast):
    """Parse a comma/space-separated length-D vector, or None."""
    if s is None:
        return None
    parts = s.replace(",", " ").split()
    vals = [cast(p) for p in parts]
    if len(vals) != D:
        raise ValueError(f"expected {D} entries, got {len(vals)}: {s!r}")
    return vals


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument("--cfg_yaml", required=True)
    p.add_argument("--out_dir", default="results/ces")
    p.add_argument(
        "--name_prefix", default="ood_bias",
        help="Output filename prefix. Files: <prefix>_<condition>.pt",
    )
    p.add_argument(
        "--K", type=int, default=5,
        help="Number of auxiliary instances.",
    )
    p.add_argument(
        "--kinds", nargs="+", default=KINDS, choices=KINDS,
        help="Which conditions to run.",
    )
    p.add_argument(
        "--bias_scale", type=float, default=0.3,
        help="Fraction of each parameter's range used as the bias magnitude.",
    )
    p.add_argument(
        "--sel_prob", type=float, default=0.5,
        help="Per-dimension probability of inclusion in the biased subset S. "
             "At least one dimension is selected.",
    )
    p.add_argument(
        "--fixed_subset", default=None,
        help="Pin S globally, e.g. '1 0 0 0 1' (length n_target_theta). "
             "Overrides --sel_prob.",
    )
    p.add_argument(
        "--fixed_signs", default=None,
        help="Pin the sign pattern s globally, e.g. '1 -1 1 1 -1'.",
    )
    p.add_argument(
        "--correlation_std", type=float, default=0.1,
        help="Relevant-parameter perturbation std as a fraction of each range.",
    )
    p.add_argument("--T", type=int, default=None, help="Eval horizon. Default cfg.T_final.")
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--M", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_query", type=int, default=None)
    p.add_argument(
        "--source_T", type=int, default=10,
        help="Number of (x, y) pairs per auxiliary experiment.",
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
    if not ut:
        print(
            "[warn] cfg.use_trajectories=False — this OOD script feeds aux via "
            "source_trajectories. The checkpoint should be a gated "
            "trajectory model."
        )

    experiment, model = load_model_and_task(cfg, args.pth_path, device)

    experiment.n_query_init = cfg.n_query_final
    T_eval = int(cfg.T_final)
    L = int(cfg.L_final)
    M = int(cfg.M_final)
    batch_size = int(cfg.eval_batch_size_final)
    D = int(experiment.n_target_theta)

    fixed_subset = _parse_vec(args.fixed_subset, D, lambda v: int(float(v)))
    fixed_signs = _parse_vec(args.fixed_signs, D, float)

    range_span = (experiment.range_max - experiment.range_min).detach().cpu().tolist()
    bias_mag = [args.bias_scale * s for s in range_span]
    dim_names = ["rho", "alpha1", "alpha2", "alpha3", "log_u"]

    print(
        f"[eval] T={T_eval}  L={L}  M={M}  batch_size={batch_size}  "
        f"n_query={cfg.n_query_final}  K={args.K}  source_T={args.source_T}"
    )
    print(f"[eval] kinds={args.kinds}  seed={args.seed}")
    print(
        f"[ood ] bias_scale={args.bias_scale}  sel_prob={args.sel_prob}  "
        f"correlation_std={args.correlation_std}"
    )
    print(f"[ood ] fixed_subset={fixed_subset}  fixed_signs={fixed_signs}")
    print("[ood ] per-dim range span / |bias| :")
    for nm, sp, bm in zip(dim_names, range_span, bias_mag):
        print(f"         {nm:>7s} : span={sp:>7.3f}   |b|={bm:>7.3f}")

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
            bias_scale=args.bias_scale,
            sel_prob=args.sel_prob,
            correlation_std=args.correlation_std,
            fixed_subset=fixed_subset,
            fixed_signs=fixed_signs,
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
            "bias_scale": args.bias_scale,
            "sel_prob": args.sel_prob,
            "correlation_std": args.correlation_std,
            "fixed_subset": fixed_subset,
            "fixed_signs": fixed_signs,
            "bias_magnitude": bias_mag,
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
        "bias_scale": args.bias_scale,
        "sel_prob": args.sel_prob,
        "correlation_std": args.correlation_std,
        "seed": args.seed,
        "conditions": [t for t, _ in conditions],
        "pth_path": args.pth_path,
        "cfg_yaml": args.cfg_yaml,
    }, summary_path)
    print(f"[saved] {summary_path}")

    print_final_table(summary)


if __name__ == "__main__":
    main()
