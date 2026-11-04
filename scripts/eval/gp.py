"""Evaluate posterior-mean RMSE on four 1D GP active-learning benchmarks.

Conditions are k0 (no auxiliary data) and k5_0bad through k5_5bad.
Analytic benchmarks use exact source counts; synthetic1d samples relevance.

Writes <prefix>_<benchmark>.pt and <prefix>_summary.csv under --out_dir.

Example:
    python scripts/eval/gp.py \\
        --pth_path <run_dir>/model/aae_GP_d1_1.pth \\
        --cfg_yaml <run_dir>/.hydra/config.yaml \\
        --seed 0
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

from scripts.eval_utils.gp import (
    load_model_and_task,
    make_benchmark_batch,
    make_synthetic_batch,
    print_final_table,
    rollout_rmse,
    save_summary_csv,
    summarize_metrics,
)
from tasks import BenchmarkTask
from utils import set_seed


BENCHMARKS = [
    ("synthetic1d", "Synthetic 1D"),
    ("forrester", "Forrester 1D"),
    ("higdon", "Higdon 1D"),
    ("gramacy1d", "Gramacy & Lee 1D"),
]

# (tag, n_relevant, n_irrelevant)
CONDITIONS = [
    ("k0", 0, 0),
    ("k5_0bad", 5, 0),
    ("k5_1bad", 4, 1),
    ("k5_2bad", 3, 2),
    ("k5_3bad", 2, 3),
    ("k5_4bad", 1, 4),
    ("k5_5bad", 0, 5),
]
COND_TAGS = [tag for tag, _, _ in CONDITIONS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth_path", required=True)
    p.add_argument(
        "--cfg_yaml", required=True,
        help="Resolved training config (.hydra/config.yaml of the run).",
    )
    p.add_argument("--out_dir", default="results/gp")
    p.add_argument("--name_prefix", default="imbue")
    p.add_argument(
        "--benchmarks", nargs="+",
        default=[b for b, _ in BENCHMARKS],
        choices=[b for b, _ in BENCHMARKS],
    )
    p.add_argument("--conditions", nargs="+", default=COND_TAGS, choices=COND_TAGS)
    p.add_argument("--T", type=int, default=15, help="Acquisition steps.")
    p.add_argument(
        "--source_T", type=int, default=None,
        help="Observations per previous instance. Default: --T.",
    )
    p.add_argument(
        "--n_episodes", type=int, default=100,
        help="Episodes (rollouts) per benchmark and condition.",
    )
    p.add_argument("--n_context_init", type=int, default=1)
    p.add_argument(
        "--n_query", type=int, default=500,
        help="Candidate design pool size.",
    )
    p.add_argument(
        "--n_target_data", type=int, default=None,
        help="Held-out target points per episode. Default: 100 on the "
             "analytic benchmarks, training-config value on synthetic1d.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print(f"[load] cfg : {args.cfg_yaml}")
    print(f"[load] pth : {args.pth_path}")
    cfg = OmegaConf.load(args.cfg_yaml)
    OmegaConf.set_struct(cfg, False)
    cfg.T = args.T
    cfg.task.n_context_init = args.n_context_init
    cfg.task.n_query_init = args.n_query
    if args.n_target_data is not None:
        cfg.task.n_target_data = args.n_target_data
    if not bool(cfg.get("use_trajectories", False)):
        print(
            "[warn] cfg.use_trajectories=False: the checkpoint was not trained "
            "with previous-instance input, so the K>0 conditions are a no-op."
        )

    gp_task, model = load_model_and_task(cfg, args.pth_path, device)

    bench_task = BenchmarkTask(
        dim_x=int(cfg.task.dim_x),
        n_context_init=args.n_context_init,
        n_query_init=args.n_query,
        n_target_data=(args.n_target_data if args.n_target_data is not None else int(cfg.task.n_target_data)),
        design_scale=float(cfg.task.design_scale),
        noise_scale=float(cfg.task.noise_scale),
    )
    T_src = args.source_T if args.source_T is not None else args.T
    conditions = [c for c in CONDITIONS if c[0] in args.conditions]
    benchmarks = [b for b in BENCHMARKS if b[0] in args.benchmarks]

    print(
        f"[eval] T={args.T}  source_T={T_src}  n_episodes={args.n_episodes}  "
        f"n_context_init={args.n_context_init}  n_query={args.n_query}  "
        f"n_target_data: analytic={bench_task.n_target_data} "
        f"synthetic1d={gp_task.n_target_data}"
    )
    print(
        f"[eval] benchmarks={[b for b, _ in benchmarks]}  "
        f"conditions={[c for c, _, _ in conditions]}  seed={args.seed}"
    )
    os.makedirs(args.out_dir, exist_ok=True)

    rows, final = [], {}
    for name, title in benchmarks:
        rollout_exp = gp_task if name == "synthetic1d" else bench_task
        per_cond = {}
        for cond_idx, (tag, n_good, n_bad) in enumerate(conditions):
            set_seed(args.seed + cond_idx)
            if name == "synthetic1d":
                batch = make_synthetic_batch(
                    gp_task,
                    batch_size=args.n_episodes,
                    n_good=n_good,
                    n_bad=n_bad,
                    T_src=T_src,
                )
            else:
                batch = make_benchmark_batch(
                    bench_task, gp_task, name,
                    batch_size=args.n_episodes,
                    n_good=n_good,
                    n_bad=n_bad,
                    T_src=T_src,
                )
            t0 = time.time()
            rmse = rollout_rmse(model, rollout_exp, batch, args.T).cpu()  # [T, B]
            mean, se = summarize_metrics(rmse)
            per_cond[tag] = {
                "rmse": rmse,
                "rmse_mean": mean,
                "rmse_se": se,
                "n_relevant": n_good,
                "n_irrelevant": n_bad,
                "seed": args.seed + cond_idx,
            }
            final[(name, tag)] = (mean[-1].item(), se[-1].item())
            rows += [
                (name, tag, t + 1, mean[t].item(), se[t].item())
                for t in range(args.T)
            ]
            print(
                f"  [{title}] {tag:<8s} final RMSE = {mean[-1].item():.4f} "
                f"+- {se[-1].item():.4f}   ({time.time() - t0:.1f}s)",
                flush=True,
            )

        path = os.path.join(args.out_dir, f"{args.name_prefix}_{name}.pt")
        torch.save({
            "benchmark": name,
            "T": args.T,
            "source_T": T_src,
            "n_episodes": args.n_episodes,
            "n_query": args.n_query,
            "n_target_data": int(rollout_exp.n_target_data),
            "conditions": per_cond,
            "pth_path": args.pth_path,
            "cfg_yaml": args.cfg_yaml,
        }, path)
        print(f"[saved] {path}")

    csv_path = os.path.join(args.out_dir, f"{args.name_prefix}_summary.csv")
    save_summary_csv(rows, csv_path)
    print(f"[saved] {csv_path}")

    print_final_table(final, benchmarks, conditions)


if __name__ == "__main__":
    main()
