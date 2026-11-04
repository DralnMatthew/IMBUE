import torch
from torch.distributions import Normal, Categorical, Uniform
from omegaconf import ListConfig


class PriorSampler:
    """
    Synthetic prior sampler used for amortised training.

    Prior generative process (per-dimension, independent across `dim_theta`):
        - With prob 0.2 the prior is the flat uniform over [range_min, range_max].
        - With prob 0.8 the prior is a single Gaussian whose mean is uniform over
          [range_min, range_max] and whose std is uniform over [std_min, std_max],
          discretised onto `num_bins` bins.

    The injected prior is then "good" (theta sampled inside it) or "bad"
    (theta deliberately shifted by 0.3–0.5 of the domain range) with the
    probability `prob_good_prior` passed to `sample_bins_and_theta`. The
    reliability label is sigmoid(beta · log(p_prior(true bin) / uniform_prob)).

    Args:
        dim_theta:    Total number of parameter dimensions (e.g. K * dim_x).
        num_bins:     Number of bins for the discrete prior.
        range_min:    Lower bound(s) for theta.
        range_max:    Upper bound(s) for theta.
        std_min:      Min std for the Gaussian prior component.
        std_max:      Max std for the Gaussian prior component.
        theta_dist:   'uniform' (always flat) or 'ace' (the 0.2/0.8 mixture above).
        beta:         Temperature of the sigmoid mapping LLR→reliability label.
        p_uniform:    Mass on the flat-uniform prior in 'ace' mode (default 0.2).
    """

    def __init__(
        self,
        dim_theta,
        num_bins,
        range_min,
        range_max,
        std_min,
        std_max,
        theta_dist='ace',
        beta: float = 2.0,
        p_uniform: float = 0.2,
    ):
        self.dim_theta = dim_theta
        self.num_bins = num_bins
        self.theta_dist = theta_dist
        self.beta = beta
        self.p_uniform = p_uniform

        self.range_min = self._to_tensor(range_min)
        self.range_max = self._to_tensor(range_max)
        self.std_min = self._to_tensor(std_min)
        self.std_max = self._to_tensor(std_max)

        step = (self.range_max - self.range_min) / num_bins # (dim_theta,)
        steps = torch.arange(num_bins + 1).unsqueeze(0) # (1, num_bins+1)

        self.bin_edges = self.range_min.unsqueeze(1) + step.unsqueeze(1) * steps # [dim_theta, num_bins+1]
        
        self.uniform_weights = torch.ones(dim_theta, num_bins) / num_bins # (dim_theta, num_bins)
    

    def _to_tensor(self, val):
        """Helper to convert scalars/lists to (dim_theta,) tensor."""
        if isinstance(val, (float, int)):
            return torch.full((self.dim_theta,), val, dtype=torch.float32)
        if isinstance(val, (list, tuple, ListConfig)):
            val = torch.tensor(val, dtype=torch.float32)
        if isinstance(val, torch.Tensor):
            val = val.float()

        if val.numel() == 1:
            return val.expand(self.dim_theta)
        assert val.shape == (self.dim_theta,), f"Expected shape ({self.dim_theta},), got {val.shape}"
        return val


    def _generate_gaussian_weights(self, batch_size):
        """
        Discretise a single Gaussian per dim onto `num_bins` bins.

        For each (b, d):
            mean ~ Uniform(range_min[d], range_max[d])
            std  ~ Uniform(std_min[d],   std_max[d])

        Returns:
            bin_weights: (batch_size, dim_theta, num_bins)
        """
        rand_loc = torch.rand(batch_size, self.dim_theta)
        means = self.range_min.unsqueeze(0) + rand_loc * (self.range_max - self.range_min).unsqueeze(0)

        rand_scale = torch.rand(batch_size, self.dim_theta)
        stds = self.std_min.unsqueeze(0) + rand_scale * (self.std_max - self.std_min).unsqueeze(0)

        edges = self.bin_edges.unsqueeze(0)                                   # (1, dim_theta, num_bins+1)
        cdf = Normal(means.unsqueeze(-1), stds.unsqueeze(-1)).cdf(edges)      # (batch_size, dim_theta, num_bins+1)

        bin_weights = cdf[..., 1:] - cdf[..., :-1]                            # (batch_size, dim_theta, num_bins)
        bin_weights = bin_weights / (bin_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return bin_weights


    def sample_bins_and_theta(self, batch_size, prob_good_prior=0.7):
        """
        Sample injected priors and the true theta they "describe".

        Returns:
            bin_weights:        (batch_size, dim_theta, num_bins)
            theta:              (batch_size, dim_theta)
            reliability_labels: (batch_size, dim_theta, 1) — sigmoid(beta · log R)
                                with R = p_prior(true_bin) / (1 / num_bins).
                                ≈ 1 when theta lies in a high-density bin
                                (well-calibrated prior), ≈ 0 when it does not.
        """
        if self.theta_dist == 'uniform':
            bin_weights = self.uniform_weights.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # (1 - p_uniform) Gaussian, p_uniform flat
            is_gauss = torch.rand(batch_size) >= self.p_uniform
            num_gauss = is_gauss.sum().item()

            bin_weights = torch.empty(batch_size, self.dim_theta, self.num_bins)
            if num_gauss < batch_size:
                bin_weights[~is_gauss] = self.uniform_weights.unsqueeze(0).expand((~is_gauss).sum(), -1, -1)
            if num_gauss > 0:
                bin_weights[is_gauss] = self._generate_gaussian_weights(num_gauss)

        flat_weights = bin_weights.reshape(-1, self.num_bins)
        flat_bin_idx = Categorical(flat_weights).sample()

        dim_indices = torch.arange(self.dim_theta).repeat(batch_size)
        lower = self.bin_edges[dim_indices, flat_bin_idx]
        upper = self.bin_edges[dim_indices, flat_bin_idx + 1]
        theta_good_flat = Uniform(lower, upper).sample()
        theta_good = theta_good_flat.reshape(batch_size, self.dim_theta)

        # "Bad" theta: deliberately pushed 0.3–0.5 of the domain range away from
        # the prior, toward whichever side has more room. Reliability label
        # turns out near 0 because the prior assigns near-zero mass to this bin.
        range_min_exp = self.range_min.unsqueeze(0).expand(batch_size, -1)
        range_max_exp = self.range_max.unsqueeze(0).expand(batch_size, -1)
        domain_range = range_max_exp - range_min_exp
        midpoint = (range_max_exp + range_min_exp) / 2.0
        direction = torch.where(theta_good < midpoint, 1.0, -1.0)
        shift_mag = domain_range * torch.empty_like(theta_good).uniform_(0.3, 0.5)
        theta_bad = torch.clamp(theta_good + direction * shift_mag, range_min_exp, range_max_exp)

        is_good = torch.bernoulli(torch.full((batch_size, self.dim_theta), prob_good_prior)).bool()
        theta = torch.where(is_good, theta_good, theta_bad)

        theta_exp = theta.unsqueeze(-1)                                       # [B, D, 1]
        edges_exp = self.bin_edges.unsqueeze(0).expand(batch_size, -1, -1)    # [B, D, num_bins+1]
        bin_indices = torch.sum(theta_exp >= edges_exp, dim=-1) - 1           # [B, D]
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        selected_bin_probs = torch.gather(bin_weights, 2, bin_indices.unsqueeze(-1))  # [B, D, 1]

        baseline_prob = 1.0 / self.num_bins
        LLR = torch.log(selected_bin_probs / baseline_prob + 1e-9)
        reliability_labels = torch.sigmoid(self.beta * LLR)                   # [B, D, 1]

        return bin_weights, theta, reliability_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg') 
    import time
    import numpy as np
    import torch

    dim_theta = 4
    num_bins = 100
    
    # range_min = [-3, 0.1, 0.1, 0]
    # range_max = [ 3, 2, 0.9, 0.5]

    # std_min = [0.3, 0.1, 0.04, 0.025]
    # std_max = [2.4, 0.8, 0.3, 0.2]

    range_min = [-3, 0.1, 0.1, 0]
    range_max = [ 3, 2, 0.9, 0.5]

    std_min = [0.5, 0.5, 0.05, 0.05]
    std_max = [1, 1, 0.1, 0.1]

    sampler = PriorSampler(
        dim_theta=dim_theta,
        num_bins=num_bins,
        range_min=range_min,
        range_max=range_max,
        std_min=std_min,
        std_max=std_max,
        theta_dist='ace',
    )

    start = time.time()

    bin_weights, theta, reliability_labels = sampler.sample_bins_and_theta(25) # [25, 5, 100], [25, 5]
    
    print(f"Sampling time: {time.time() - start:.4f}s")
    print(f"bin_weights shape: {bin_weights.shape}")
    print(f"theta shape: {theta.shape}")

    target_dim = 2
    print(f"\nVisualizing Dimension {target_dim} (Range: [{sampler.range_min[target_dim]}, {sampler.range_max[target_dim]}])")

    dim_edges = sampler.bin_edges[target_dim].detach().cpu().numpy() # [dim_theta, num_bins + 1]
    bin_centers = (dim_edges[:-1] + dim_edges[1:]) / 2

    weights_np = bin_weights[:, target_dim, :].detach().cpu().numpy() # [25, 100]
    theta_np = theta[:, target_dim].detach().cpu().numpy() # [25]

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    fig.suptitle(f"Prior Samples for Dimension {target_dim}\n(Blue: Prior Distribution, Red: Sampled Theta)", fontsize=16)

    for i in range(25):
        ax = axes[i // 5, i % 5]
        
        ax.plot(bin_centers, weights_np[i], color='royalblue', linewidth=1.5, alpha=0.9)
        ax.fill_between(bin_centers, weights_np[i], alpha=0.2, color='royalblue')
        ax.axvline(theta_np[i], color='crimson', linestyle='--', linewidth=1.5)
        
        ax.set_title(f"Sample {i+1} reliability: {reliability_labels[i, target_dim].item():.2f}")
        ax.set_xlim(sampler.range_min[target_dim], sampler.range_max[target_dim])
        y_max = np.max(weights_np[i])
        ax.set_ylim(0, y_max * 1.2 if y_max > 0 else 1.0)

        if i % 5 != 0:
            ax.set_ylabel('')
        if i < 20:
            ax.set_xticklabels([])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = f"prior_samples_5x5.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")