import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Normal, constraints

class MixtureOfGaussians(Distribution):
    arg_constraints = {
        'means': constraints.real_vector,
        'stds': constraints.positive,
        'weights': constraints.simplex
    }

    def __init__(self, means, stds, weights, validate_args=None):
        """
        Initialize Mixture of Gaussians with per-dimension weights.
        
        Args:
            means: [B, D, N] - means of Gaussian components
            stds: [B, D, N] - standard deviations of Gaussian components
            weights: [B, D, N] - mixing weights (must sum to 1 along last dimension)
        """
        assert means.shape == stds.shape == weights.shape
        
        self.means = means      # [B, D, N] B: batch size, D: dimensions, N: components [B, n_target/K*dim_x, dim_y*components/1*10]
        self.stds = stds        # [B, D, N]
        self.weights = weights  # [B, D, N]
        
        # Create a categorical distribution for the mixing coefficients per dimension
        # Note: Categorical expects weights of shape [..., N] where last dim sums to 1
        self.mixing_distribution = Categorical(weights)  # handles [B, D, N]
        self.component_distribution = Normal(means, stds)
        
        Distribution.__init__(self, batch_shape=means.shape[:-1], validate_args=validate_args)

    def log_prob(self, x):
        """
        Compute log probability of samples for each dimension independently.
        
        Args:
            x: [L, B, D] or [B, D] - sample values
            
        Returns:
            [L, B, D] or [B, D] - log probabilities for each dimension
        """
        
        x_expanded = x.unsqueeze(-1) # [..., B, D, 1]
        component_log_probs = self.component_distribution.log_prob(x_expanded) # [L, B, D, N]

        log_weights = torch.log(self.weights + 1e-12)
        log_mixture_probs = torch.logsumexp(component_log_probs + log_weights, dim=-1) # [L, B, D]
        
        return log_mixture_probs

    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the mixture distribution.
        
        Args:
            sample_shape: shape of samples to generate
            
        Returns:
            [..., B, D] - samples from the mixture
        """
        # Sample component indices according to the mixing probabilities
        # mixing_distribution handles [B, D, N] -> samples shape [..., B, D]
        component_indices = self.mixing_distribution.sample(sample_shape)
        
        # Sample from all Gaussian components
        samples = self.component_distribution.sample(sample_shape)  # [..., B, D, N]
        
        # component_indices: [..., B, D] -> [..., B, D, 1]
        idx = component_indices.unsqueeze(-1)  # [..., B, D, 1]
        
        gathered = torch.gather(samples, -1, idx)  # [..., B, D, 1]
        
        return gathered.squeeze(-1)  # [..., B, D]

    @property
    def mean(self):
        """
        Return the mean of the mixture distribution.
        
        Returns:
            [B, D] - mean for each dimension
        """
        return torch.sum(self.weights * self.means, dim=-1)  # [B, D]

    @property
    def variance(self):
        """
        Return the variance of the mixture distribution.
        
        Returns:
            [B, D] - variance for each dimension
        """
        # E[X^2] = sum_i w_i * (mu_i^2 + sigma_i^2)
        mean_sq = torch.sum(self.weights * (self.means ** 2 + self.stds ** 2), dim=-1)  # [B, D]
        # Var[X] = E[X^2] - E[X]^2
        return mean_sq - self.mean ** 2

    @property
    def entropy(self):
        """
        Compute entropy for each dimension using Monte Carlo sampling.
        
        Returns:
            [B, D] - entropy for each dimension
        """
        num_samples = 1000
        
        # Generate samples: [num_samples, B, D]
        samples = self.sample(sample_shape=(num_samples,))
        
        # Compute log probability of those samples: [num_samples, B, D]
        log_probs = self.log_prob(samples)
        
        # Entropy H(X) = -E[log p(X)]
        return -log_probs.mean(dim=0)


if __name__ == "__main__":
    torch.manual_seed(42)

    def run_tests():
        print("--- Running Tests ---")
        B, D, N = 2, 3, 4
        means = torch.randn(B, D, N)
        stds = torch.rand(B, D, N) + 0.1
        weights = F.softmax(torch.randn(B, D, N), dim=-1)
        mog = MixtureOfGaussians(means, stds, weights)

        L = 10
        x_seq = torch.randn(L, B, D)
        log_p_seq = mog.log_prob(x_seq)
        print(f"1. Input [L, B, D]: {x_seq.shape}")
        print(f"   Output:          {log_p_seq.shape}")
        assert log_p_seq.shape == (L, B, D), "Failed L, B, D shape check"

        x_batch = torch.randn(B, D)
        log_p_batch = mog.log_prob(x_batch)
        print(f"2. Input [B, D]:    {x_batch.shape}")
        print(f"   Output:          {log_p_batch.shape}")
        assert log_p_batch.shape == (B, D), "Failed B, D shape check"

        ety = mog.entropy
        print(f"3. Entropy Shape:   {ety.shape}")
        assert ety.shape == (B, D)

        print("All tests passed.")
    
    run_tests()
