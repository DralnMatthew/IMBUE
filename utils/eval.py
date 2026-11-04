import torch
from torch.distributions import Normal, Uniform
import numpy as np
from loss import PCELoss, NMCLoss
from loss.eig import EIGStepLoss
from attrdictionary import AttrDict

from torch.distributions import Normal, Categorical, MixtureSameFamily
import math


@torch.no_grad()
def get_traces(
    model, 
    experiment, 
    T=30, 
    batch_size=40, 
    use_trajectories=False, 
    n_source_tasks=0, 
    prob_good_trajectory=1.0, 
    prior_type='good'
):
    """ 
    Run active learning rollouts and collect history
    """
    model.eval()

    theta_shape = experiment.get_theta_shape(batch_size)

    if use_trajectories:
        batch = experiment.sample_correlated_batch_eval(batch_size, n_source_tasks, prob_good_trajectory=prob_good_trajectory)
        source_trajectories = batch.source_trajectories
    else:
        if prior_type:
            batch = experiment.sample_batch_eval(batch_size, prior_type=prior_type)
            source_trajectories = None
        else:
            batch = experiment.sample_batch(batch_size)
            source_trajectories = None

    
    if experiment.with_query:
        for t in range(T):
            idx = model.forward(batch, source_trajectories).design_out.idx      # [B, 1]
            batch = experiment.update_batch_with_query(batch, idx)
    else:
        for t in range(T):
            outs = model.forward(batch, source_trajectories)
            xi = outs.design_out.xi                        # [B, 1, dim_x]
            y = experiment.forward(xi.squeeze(1), batch.target_theta.reshape(theta_shape)).unsqueeze(1)  # [B, 1, 1]
            batch = experiment.update_batch_without_query(batch, xi, y)

    # collect theta0
    theta_0 = batch.target_theta.reshape(theta_shape)      # [B, K, D]

    # collect history
    x = experiment.unnormalise_design(batch.context_x)      # [B, T, D_x]
    y = batch.context_y                                     # [B, T, D_y]

    return theta_0, x, y


@torch.no_grad()
def compute_EIG_from_history(experiment, theta_0, x, y, L=int(1e6), batch_size=40, stepwise=False):
    """ Evaluate the lower and upper bounds of EIG from a minibatch of the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        priors:  # [B, K, D, num_bins]
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]

    criterion = EIGStepLoss(L, batch_size, experiment.log_likelihood, reduction='none')

    pce_losses = []
    nmc_losses = []

    thetas = experiment.sample_globalprior_theta((L, batch_size))

    thetas = torch.concat([theta_0.unsqueeze(0), thetas], dim=0)          # [L+1, B, K, D]

    if stepwise:
        for t in range(T):
            pce_loss, nmc_loss = criterion(y[:, t], x[:, t], thetas)  # [B]
            pce_losses.append(pce_loss)
            nmc_losses.append(nmc_loss)
                
        pce_losses = torch.stack(pce_losses, dim=-1)  # [B, T]
        nmc_losses = torch.stack(nmc_losses, dim=-1)  # [B, T]
    else:
        for t in range(T):
            pce_losses, nmc_losses = criterion(y[:, t], x[:, t], thetas)  # [B]

    # Calculate bounds
    pce_losses = torch.log(torch.tensor(L + 1)) - pce_losses  # [B(, T)]
    nmc_losses = torch.log(torch.tensor(L)) - nmc_losses      # [B(, T)]  

    return pce_losses, nmc_losses

@torch.no_grad()
def eval_boed(
    model, 
    experiment, 
    T=30, 
    L=int(1e6), 
    M=2000, 
    batch_size=40, 
    stepwise=True, 
    use_trajectories=False, 
    prob_good_trajectory=1.0, 
    conditions=None, # [0, 1, 5] / ['good', 'bad']
    verbose=False
):
    """
    Evaluate EIG (PCE/NMC) across specified conditions.
    Returns a dictionary mapping condition -> AttrDict of metrics.
    """
    model.eval()
    max_step = (M + batch_size - 1) // batch_size

    if conditions is None:
        if use_trajectories:
            conditions = [0, 1, 2, 3, 4, 5]
        elif experiment.prior_injection:
            conditions = ['good']
        else:
            conditions = [None]

    results_dict = {}

    for condition in conditions:
        if verbose:
            print(f"\n--- Evaluating Condition: {condition} ---")

        pce_list, nmc_list = [], []

        for step in range(max_step):
            n_src = condition if use_trajectories else 0
            prior_type = condition if (experiment.prior_injection and not use_trajectories) else None

            theta_0, x, y = get_traces(
                model, experiment, T, batch_size, 
                use_trajectories=use_trajectories, 
                n_source_tasks=n_src, 
                prob_good_trajectory=prob_good_trajectory, 
                prior_type=prior_type
            )
            
            pce, nmc = compute_EIG_from_history(experiment, theta_0, x, y, L, batch_size, stepwise)
            
            pce_list.append(pce)
            nmc_list.append(nmc)

            if verbose:
                print(f"  Step {step+1}/{max_step} | PCE: {pce.mean():.3f} ")

        pce_all = torch.cat(pce_list, dim=0)       # [M, T]
        nmc_all = torch.cat(nmc_list, dim=0)       # [M, T]

        actual_M = pce_all.shape[0]

        metrics = AttrDict({
            'pce_mean': pce_all.mean(dim=0),
            'pce_std': (pce_all.std(dim=0) / np.sqrt(actual_M)),
            'nmc_mean': nmc_all.mean(dim=0),
            'nmc_std': (nmc_all.std(dim=0) / np.sqrt(actual_M)),
        })

        cond_key = f"{condition}"

        results_dict[cond_key] = metrics

    return results_dict


def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes log-likelihood loss for a Gaussian mixture model.
    """
    components = Normal(means, stds, validate_args=False)
    log_probs = components.log_prob(value)
    weighted_log_probs = log_probs + torch.log(weights)
    return torch.logsumexp(weighted_log_probs, dim=-1)


def compute_rmse(target_values, mixture_means, mixture_stds, mixture_weights):
    """Compute RMSE between target values and predictions from Gaussian mixture model

    Args:
        target_values: Ground truth values [batch_size, n_targets, dim_y]
        mixture_means: Means of mixture components [batch_size, n_targets, n_components]
        mixture_stds: Standard deviations of mixture components [batch_size, n_targets, n_components]
        mixture_weights: Weights of mixture components [batch_size, n_targets, n_components]

    Returns:
        rmse: RMSE values [batch_size]
    """
    weighted_means = torch.sum(mixture_weights * mixture_means, dim=-1)  # [batch_size, n_targets]

    squared_error = (target_values.squeeze(-1) - weighted_means) ** 2  # [batch_size, n_targets]

    rmse = torch.sqrt(torch.mean(squared_error, dim=-1))  # [batch_size]

    return rmse

def compute_gmm_entropy(means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
    """
    Computes the approximate entropy of a Gaussian mixture model using Monte Carlo sampling.
    H(X) = - E[log p(x)] approx - mean(log p(x_sampled))
    
    Args:
        means: [batch_size, dim_x, num_components]
        stds: [batch_size, dim_x, num_components]
        weights: [batch_size, dim_x, num_components] (should sum to 1)
        num_samples: Number of MC samples to estimate entropy
    
    Returns:
        entropy: [batch_size]
    """
    mix_dist = Categorical(probs=weights) 
    comp_dist = Normal(means, stds)
    gmm = MixtureSameFamily(mix_dist, comp_dist)
    x_samples = gmm.sample((num_samples,))
    log_probs = gmm.log_prob(x_samples)
    entropy = -log_probs.mean(dim=0)
    
    return entropy