import torch
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task
from utils import PriorSampler
from torch.distributions import Categorical, Uniform, Normal


class HiddenLocation(Task):
    """Simulate Location Finding Experiment"""

    def __init__(
        self,
        name: str = "Location",
        dim_x: int = 2,             # dimension of location
        dim_y: int = 1,             # dimension of outcome
        n_target_theta: int = 2,  # number of theta, treat each coordinate as one theta
        n_context_init: int = 1,    # number of initial context points
        n_query_init: int = 200,    # number of initial query points
        K: int = 1,                 # number of source points
        prior_injection: bool = True,
        num_bins: int = 100,
        theta_dist="uniform",       # prior distribution type: uniform / ace
        range_min=0,
        range_max=1,
        design_scale=None,          # scale of the design space
        outcome_scale=10,           # scale of the experiment outcomes
        noise_scale=0.5,
        base_signal: float = 0.1,   # param of signal
        max_signal: float = 1e-4,   # param of signal
        with_query: bool = True,    # whether to include query points
        prob_good_prior: float = 0.7,
        **kwargs,
    ) -> None:
        super(HiddenLocation, self).__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=n_target_theta, K=K)

        self.num_bins = num_bins
        self.prior_injection = prior_injection
        self.dim_x = dim_x
        self.dim_theta =K * self.dim_x

        self.prob_good_prior = prob_good_prior
        
        self.with_query = with_query

        self.range_min = range_min
        self.range_max = range_max

        # prior of theta
        self.theta_dist = theta_dist

        self.theta_prior = PriorSampler(dim_theta = K * self.dim_x, 
                                        num_bins = self.num_bins, 
                                        range_min = self.range_min, 
                                        range_max = self.range_max, 
                                        std_min = 0.05, 
                                        std_max = 0.3, 
                                        theta_dist=self.theta_dist
                                        )
        
        # sampler of the data
        low = range_min * torch.ones((K, self.dim_x))
        high = range_max * torch.ones((K, self.dim_x))
        self.data_sampler = dist.Uniform(low, high)

        # scale of design space
        self.design_scale = (
            design_scale if design_scale is not None else torch.max(self.theta_cov)
        )
        # signal params
        noise_scale = noise_scale * torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("noise_scale", noise_scale)
        self.base_signal = base_signal
        self.max_signal = max_signal
        
        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init if with_query else 1

        self.K = K
        assert self.n_target_theta == self.K * self.dim_x, "n_theta must be equal to K * dim_x"

        self.conditional = dist.Normal(0, self.noise_scale)

    def _sort_sources(self, theta, bins=None, prior_labels=None):
        """Sort sources, priors and labels together by the first coordinate.

        Args:
            theta:        [..., K, D]
            bins:         [..., K, D, num_bins] or None
            prior_labels: [..., K, D] or None

        Returns:
            Sorted (theta, bins, prior_labels)
        """
        sort_idx = theta[..., 0].argsort(dim=-1, stable=True)  # [..., K]

        idx_theta = sort_idx.unsqueeze(-1).expand_as(theta)
        theta = torch.gather(theta, -2, idx_theta)

        if bins is not None:
            idx_bins = sort_idx.unsqueeze(-1).unsqueeze(-1).expand_as(bins)
            bins = torch.gather(bins, -3, idx_bins)

        if prior_labels is not None:
            idx_labels = sort_idx.unsqueeze(-1).expand_as(prior_labels)
            prior_labels = torch.gather(prior_labels, -2, idx_labels)

        return theta, bins, prior_labels

    @torch.no_grad()
    def sample_theta_prior(self, batch_size, prob_good_prior):
        bins, theta, prior_labels = self.theta_prior.sample_bins_and_theta(batch_size, prob_good_prior) # [B, K*D, num_bins] [B, K*D]
        bins = bins.reshape(batch_size, self.K, self.dim_x, self.num_bins) # [B, K, D, num_bins]
        theta = theta.reshape(batch_size, self.K, self.dim_x) # [B, K, D]
        prior_labels = prior_labels.reshape(batch_size, self.K, self.dim_x) # [B, K, D]
        if self.K > 1:
            theta, bins, prior_labels = self._sort_sources(theta, bins, prior_labels)
        return bins, theta, prior_labels
    

    @torch.no_grad()
    def sample_globalprior_theta(self, shape):
        theta_prior = dist.Uniform(torch.zeros((self.K, self.dim_x)), torch.ones((self.K, self.dim_x)))
        theta = theta_prior.sample(shape) # [*shape, K, D]
        if self.K > 1:
            theta, _, _ = self._sort_sources(theta)
        return theta


    @torch.no_grad()
    def get_theta_shape(self, batch_size):
        return (batch_size, self.K, self.dim_x)


    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """ Sample designs """
        data = self.data_sampler.sample([batch_size, n_data])
        return data[..., 0, :] # [B, N, K, D] -> [B, N, D]

        # return data.reshape(batch_size, n_data, self.dim_x)  # [B, N, 2]

    def total_density(self, xi, theta):
        """Total density

        Shape:
            xi: [:, D] - [B, D] or [1/L, B, T, D]
            theta: [:, K, D] - [B, K, D] or [L, B, T, K, D]

        Returns:
            density: [:, 1]
        """
        # two norm squared
        sq_two_norm = (
            (xi.unsqueeze(-2).expand(theta.shape) - theta).pow(2).sum(axis=-1)
        )  # [:, K]
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        density = torch.log(
            self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True)
        )  # [:, 1]

        return density

    def forward(self, xi, theta):
        """Sample observations with reparameterized Gaussian noise.

        Args:
            xi [B, D]: normalised design
            theta [B, K, D]: sources

        Returns:
            observations: [B, 1]
        """

        signal = self.total_density(xi, theta)  # [B, 1]
        # Add noise
        noised_signal = dist.Normal(signal, self.noise_scale).rsample()
        return noised_signal

    def log_likelihood(self, y, xi, theta):
        """Log likelihood from gaussian noise

        Args:
            y [:, 1]
            xi [:, D]: real designs
            theta [:, K, D]

        Returns:
            log_prob: log likelihoods, [:, 1]
        """
        # uncorrupted signal
        signal = self.total_density(xi, theta)
        # Calculate the log likelihood
        log_prob = dist.Normal(signal, self.noise_scale).log_prob(y)
        return log_prob

    
    @torch.no_grad()
    def sample_batch(self, batch_size):
        """Sample a batch of data

        Args:
            batch_size (int): number of batches
            eval_mode (bool): whether to turn on evalution mode
        Returns:
            batch (AttrDict): a batch of data containing context and query points
        """
        if self.prior_injection:
            bin_weights, theta, prior_labels = self.sample_theta_prior(batch_size, self.prob_good_prior)  # [B, K, D, num_bins], [B, K, D]
        else:
            theta = self.sample_globalprior_theta([batch_size])

        if not self.with_query:
            self.n_query_init = 1

        num_samples = self.n_context_init + self.n_query_init

        # normalised design
        x = self.sample_data(batch_size, num_samples) # [B, n_context+n_query, D]
        y = self.forward(self.unnormalise_design(x), theta.unsqueeze(1).expand(batch_size, num_samples, self.K, self.dim_x))  # [B, n_context+n_query, 1]

        batch = AttrDict()
        if self.prior_injection:
            batch.priors = bin_weights.reshape(batch_size, -1, self.num_bins) # [B, K*D, num_bins]
            batch.prior_labels = prior_labels.reshape(batch_size, -1, 1) # [B, K*D, 1]
        batch.context_x = x[:, :self.n_context_init] # [B, n_context_init, D]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:]
        batch.query_y = y[:, self.n_context_init:]
        batch.target_theta = theta.reshape(batch_size, self.n_target_theta, 1) # [B, K*D, 1]
        batch.n_target_theta = self.n_target_theta

        batch.target_all = batch.target_theta

        return batch
    

    @torch.no_grad()
    def sample_batch_eval(self, batch_size, largs=None, prior_type='good'):
        """Construct an evaluation batch with a controlled injected prior.

        prior_type:
            'strong'        — Gaussian prior with std=0.10, mean ~ N(theta_true, 0.10).
            'weak'          — Gaussian prior with std=0.25, mean ~ N(theta_true, 0.25).
            'good'          — Gaussian prior whose mean is sampled around theta_true
                              with stds given by `largs` (default 0.1 per dim).
            'miscalibrated' — Gaussian prior whose mean is shifted 0.3–0.5 of the
                              domain range away from theta_true (theta_true sits in
                              the prior's tail). Same std as `good`.
            'bad'           — alias for 'miscalibrated'.
            'uniform'       — flat prior over [range_min, range_max], reliability=0.5.
        """
        n_theta = self.K * self.dim_x
        if largs is None:
            largs = [0.1] * n_theta
        theta_true = self.sample_globalprior_theta([batch_size]) # [B, K, D]; sorted by x-coord if K>1

        if not self.prior_injection:
            return self._build_eval_batch(batch_size, theta_true, bin_weights=None, prior_labels=None)

        # Resolve aliases
        if prior_type == 'strong':
            largs = [0.1] * n_theta
            prior_type = 'good'
        elif prior_type == 'weak':
            largs = [0.25] * n_theta
            prior_type = 'good'
        elif prior_type == 'bad':
            prior_type = 'miscalibrated'

        stds_list = (largs * n_theta)[:n_theta]
        stds = torch.tensor(stds_list, dtype=torch.float32)
        stds = stds.reshape(1, n_theta).expand(batch_size, -1)        # [B, K*D]
        flat_theta_true = theta_true.reshape(batch_size, n_theta)     # [B, K*D]

        # Fast path: uniform prior — skip CDF integration.
        if prior_type == 'uniform':
            bin_weights = torch.ones(batch_size, n_theta, self.num_bins) / self.num_bins
            prior_labels = torch.full((batch_size, n_theta, 1), 0.5)
            return self._build_eval_batch(batch_size, theta_true, bin_weights, prior_labels)

        if prior_type == 'good':
            theta_prior_mean = Normal(loc=flat_theta_true, scale=stds).sample()
        elif prior_type == 'miscalibrated':
            range_min_exp = self.theta_prior.range_min.unsqueeze(0).expand(batch_size, -1)
            range_max_exp = self.theta_prior.range_max.unsqueeze(0).expand(batch_size, -1)
            domain_range = range_max_exp - range_min_exp
            midpoint = (range_max_exp + range_min_exp) / 2.0
            direction = torch.where(flat_theta_true < midpoint, 1.0, -1.0)
            shift_mag = domain_range * torch.empty_like(flat_theta_true).uniform_(0.3, 0.5)
            theta_prior_mean = torch.clamp(flat_theta_true + direction * shift_mag, range_min_exp, range_max_exp)
        else:
            raise ValueError(f"Unknown prior_type={prior_type!r}")

        # Discretise the Gaussian prior into bins.
        edges_exp = self.theta_prior.bin_edges.unsqueeze(0)            # [1, K*D, num_bins+1]
        cdf = Normal(loc=theta_prior_mean.unsqueeze(-1), scale=stds.unsqueeze(-1)).cdf(edges_exp)
        bin_probs = cdf[..., 1:] - cdf[..., :-1]
        bin_weights = bin_probs / (bin_probs.sum(dim=-1, keepdim=True) + 1e-12)

        # Reliability label = sigmoid(beta · log p_prior(true_bin) / uniform).
        # Same formula as PriorSampler so train/eval supervision is on the same scale.
        bin_indices = torch.sum(flat_theta_true.unsqueeze(-1) >= edges_exp.expand(batch_size, -1, -1), dim=-1) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        selected_bin_probs = torch.gather(bin_weights, 2, bin_indices.unsqueeze(-1))
        LLR = torch.log(selected_bin_probs / (1.0 / self.num_bins) + 1e-9)
        prior_labels = torch.sigmoid(self.theta_prior.beta * LLR)

        return self._build_eval_batch(batch_size, theta_true, bin_weights, prior_labels)

    @torch.no_grad()
    def _build_eval_batch(self, batch_size, theta_true, bin_weights, prior_labels):
        """Shared assembly: simulate observations and pack into AttrDict."""
        if not self.with_query:
            self.n_query_init = 1

        num_samples = self.n_context_init + self.n_query_init
        x = self.sample_data(batch_size, num_samples)
        y = self.forward(
            self.unnormalise_design(x),
            theta_true.unsqueeze(1).expand(batch_size, num_samples, self.K, self.dim_x),
        )

        batch = AttrDict()
        if self.prior_injection and bin_weights is not None:
            batch.priors = bin_weights.reshape(batch_size, -1, self.num_bins)
            batch.prior_labels = prior_labels
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:]
        batch.query_y = y[:, self.n_context_init:]
        batch.target_theta = theta_true.reshape(batch_size, self.n_target_theta, 1)
        batch.n_target_theta = self.n_target_theta
        batch.target_all = batch.target_theta
        return batch

    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"HiddenLocation({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


if __name__ == "__main__":
    task = HiddenLocation()
    batch = task.sample_batch(10)
    print(batch.target_theta.shape)