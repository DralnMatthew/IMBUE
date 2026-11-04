# IMBUE: In-Context Amortized Bayesian Experimental Design with Unified Knowledge Encoding

Official implementation of **Amortized Bayesian Experimental Design with In-Context Knowledge Conditioning** (NeurIPS 2026).

IMBUE trains a Transformer design policy offline and accepts external knowledge at deployment without retraining:

- **User-specified priors:** marginal beliefs encoded as histogram tokens.
- **Previous instances:** design–outcome trajectories from earlier instances of the same experiment, compressed into one token per instance.
- **Reliability filtering:** each external token is scored against the observations collected so far and masked if judged unreliable. With no observations, external tokens are excluded until the first observation arrives.

The policy is trained with both reliable and unreliable inputs. Across four benchmarks, reliable knowledge accelerates information acquisition, while unreliable knowledge yields performance close to the no-external baseline.

## Setup

Requires Linux, [uv](https://docs.astral.sh/uv/) (tested with 0.12), and Python 3.10. The Python version is pinned in `.python-version`; uv downloads it if needed.

From the repository root:

```bash
uv sync
```

Dependencies are declared in `pyproject.toml`. Commit the generated `uv.lock`; use `uv sync --locked` to install from an existing lockfile. The project is not installed as a package, so run all commands from the repository root.

The commands below use `uv run` and disable Weights & Biases. To enable logging, run `uv run wandb login` (or set `WANDB_API_KEY`) and use `wandb.use_wandb=True`, the default in `config/train_bed.yaml`.

## Training

Training uses Hydra with `config/train_bed.yaml`. Select a task with `task=<name>` and override configuration fields on the command line.


| Task                       | Entry point           | In-context knowledge                        | Main metric                  |
| -------------------------- | --------------------- | ------------------------------------------- | ---------------------------- |
| Location finding           | `run_bed.py`          | Priors over two hidden sources' coordinates | sPCE ↑                       |
| CES preference elicitation | `run_ces.py`          | Previous respondents' trajectories          | sPCE ↑                       |
| GP active learning         | `run_gp.py`           | Previous function observations              | Predictive RMSE ↓            |
| Psychometric estimation    | `run_psychometric.py` | Previous participants' trajectories         | Joint threshold–slope RMSE ↓ |


**Location finding**

```bash
uv run python -u run_bed.py task=location_finding \
    task.n_target_theta=4 task.K=2 lr=1e-3 T=20 \
    max_epoch=70000 burning_epoch=15000 eval_interval=1000 \
    eval_batch_size=1000 L_final=1000000 eval_batch_size_final=200 T_final=20 \
    task.prior_injection=True task.theta_dist=ace task.prob_good_prior=0.7 \
    gating=True wandb.use_wandb=False
```

**CES**

```bash
uv run python -u run_ces.py task=ces T=10 \
    max_epoch=250000 burning_epoch=30000 eval_interval=1000 \
    L_final=10000000 eval_batch_size_final=20 T_final=10 \
    use_trajectories=True n_summary_tokens=2 task.prior_injection=False \
    task.prob_good_trajectory=0.7 gating=True wandb.use_wandb=False
```

**GP active learning**

```bash
uv run python -u run_gp.py task=gp task.dim_x=1 task.n_target_theta=2 \
    task.lengthscale_lower=0.1 task.lengthscale_upper=2.0 task.design_scale=5 \
    max_epoch=250000 burning_epoch=30000 T=15 gamma=1 eval_interval=1000 \
    use_trajectories=True n_summary_tokens=4 \
    task.prob_good_trajectory=0.7 gating=True wandb.use_wandb=False
```

**Psychometric estimation**

```bash
uv run python -u run_psychometric.py task=psychometric T=30 gamma=1 \
    max_epoch=120000 burning_epoch=15000 eval_interval=1000 \
    use_trajectories=True n_summary_tokens=4 task.prior_injection=False \
    task.prob_good_trajectory=0.7 gating=True wandb.use_wandb=False
```

The location-finding configuration uses the prior channel; the other three use the previous-instance channel. `n_summary_tokens` sets the number of internal summary slots. The number of previous instances is sampled from `n_source_tasks_range` (default: 0–5).

`gating=True` enables the reliability filter. To reproduce the **IMBUE w/o filter** ablation, train a separate model with `gating=False`, retaining the same external inputs and training mixture.

### Outputs

Each training run writes to `outputs/hydra/<date>/<time>/`:

- `model/aae_<Task>_d<dim>_<run_id>.pth`: model weights.
- `.hydra/config.yaml`: configuration required to load the checkpoint.

Text logs are stored in `outputs/logs/`; W&B files are stored in the run directory when enabled.

## Evaluation

Evaluation scripts live in `scripts/eval/`. All evaluation entry points are listed below. Each one loads a trained policy from a training run: `--pth_path` points at `model/aae_<Task>_d<dim>_<run_id>.pth` and `--cfg_yaml` at `.hydra/config.yaml` of that run. Run them from the repository root. Results are saved under `results/<task>/` (`--out_dir`) as `.pt` files and a CSV summary. The GP evaluator groups all selected conditions into one `.pt` file per benchmark.


| Task                      | Script                          | What it measures                                                                                                                                                        |
| ------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `location_finding`        | `scripts/eval/location.py`      | sPCE along the rollout under strong, weak, miscalibrated and uniform injected priors (`--pool_type sobol` swaps in a low-discrepancy candidate design pool)             |
| `location_finding`        | `scripts/eval/location_ood.py`  | sPCE under priors whose support excludes the true parameters (truncated and boundary priors); the in-distribution prior conditions come from `scripts/eval/location.py` |
| `ces`                     | `scripts/eval/ces.py`           | sPCE with K = 0 ... 5 previous instances drawn from the relevant or the irrelevant regime (`--pool_type`, `--n_query` for candidate-pool sweeps)                        |
| `ces`                     | `scripts/eval/ces_ood_bias.py`  | sPCE with previous instances that share a systematic parameter bias (a deployment shift absent from training)                                                           |
| `ces`                     | `scripts/eval/ces_ood_noise.py` | sPCE with previous instances whose outcomes carry Student-t noise (`--no_filter` bypasses the reliability filter)                                                       |
| `ces`                     | `scripts/eval/ces_gating.py`    | mean reliability score of the previous-instance tokens at every step of the rollout                                                                                     |
| `gp`                      | `scripts/eval/gp.py`            | RMSE on the four 1D benchmarks (Synthetic, Forrester, Higdon, Gramacy & Lee) with K = 0 or K = 5 previous instances of varying relevance                                |
| `psychometric`            | `scripts/eval/psychometric.py`  | joint RMSE on (threshold, slope) with K = 0, 1 or 5 relevant or irrelevant previous instances                                                                           |
| `ces`, `location_finding` | `scripts/time_overhead.py`      | per-step inference latency and parameter counts, with the reliability filter active or bypassed                                                                         |


Example (CES):

```bash
RUN_DIR=outputs/hydra/<date>/<time>
python scripts/eval/ces.py \
    --pth_path $RUN_DIR/model/aae_CES_d6_1.pth \
    --cfg_yaml $RUN_DIR/.hydra/config.yaml \
    --name_prefix imbue --T 10 --source_T 10 --seed 0
```

Every script takes `--seed`; the conditions of one run are evaluated with `seed + condition_index`, so two checkpoints evaluated with the same seed see the same evaluation batches. `--help` lists the remaining options (acquisition horizon `--T`, number of rollouts `--M` or `--n_episodes`, contrastive samples `--L`, candidate pool size `--n_query`, ...).

## Slurm

Submit jobs from the repository root. The available jobs are:


| Task             | Training                          | Evaluation                                           |
| ---------------- | --------------------------------- | ---------------------------------------------------- |
| Location Finding | `slurms/train_location.slurm`     | `slurms/eval_location.slurm`                         |
| CES              | `slurms/train_ces.slurm`          | `slurms/eval_ces.slurm`, `slurms/eval_ces_ood.slurm` |
| GP               | `slurms/train_gp.slurm`           | `slurms/eval_gp.slurm`                               |
| Psychometric     | `slurms/train_psychometric.slurm` | `slurms/eval_psychometric.slurm`                     |


Create `out/` before submitting so Slurm can write its logs. Evaluation jobs require `RUN_DIR` to point to the training run containing the checkpoint and Hydra config:

```bash
mkdir -p out
RUN_DIR="outputs/hydra/<date>/<time>"
sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurms/eval_location.slurm
```



## Project layout

```text
config/          Hydra training, task, and model configurations
tasks/           Experiment simulators and active-learning benchmarks
model/           Token embedders, Transformer encoders, filter, and output heads
loss/            EIG bounds
distributions/   Custom distributions
utils/           Training, evaluation, logging, and prior sampling
run_*.py         Training entry points
scripts/         Evaluation scripts
```



## Citation

```bibtex
@inproceedings{zhao2026imbue,
  title     = {Amortized {Bayesian} Experimental Design with In-Context Knowledge Conditioning},
  author    = {Zhao, Zhanghu and Huang, Daolang and Wen, Xinyi and Kaski, Samuel},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

