import torch
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task
from distributions import CensoredSigmoidNormal


class CESTask(Task):
    """Simulate Constant Elasticity of Substitution (CES) utility experiments"""

    def __init__(
            self,
            name: str = "CES",
            dim_x: int = 6,  # dimension of input (2 baskets x 3 commodities)
            dim_y: int = 1,  # dimension of outcome (preference rating)
            n_target_theta: int = 5,  # number of parameters: [rho, alpha1, alpha2, alpha3, u]
            range_min=[0, 0, 0, 0, -8],
            range_max=[1, 1, 1, 1, 8],
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 300,  # number of initial query points
            design_scale: int = 100,  # scale of the design space (baskets in [0,100])
            noise_scale: float = 0.005,  # sigma_eta in the model
            epsilon: float = 2 ** (-22),  # epsilon for response clipping
            with_query: bool = True,    # whether to include query points
            prior_injection: bool = False,  # read by the shared model configs; no prior channel here
            **kwargs
    ) -> None:
        super(CESTask, self).__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=n_target_theta)

        if prior_injection:
            raise ValueError("CESTask has no prior channel; set task.prior_injection=False.")
        self.prior_injection = False

        self.basket_dim = 3  # Each basket has 3 commodities
        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.design_scale = design_scale
        self.noise_scale = noise_scale
        self.epsilon = epsilon
        self.with_query = with_query

        # Parameter ranges of [rho, alpha1, alpha2, alpha3, log u], used to
        # normalise parameter distances and to clip perturbed parameters.
        self.range_min = torch.tensor(list(range_min), dtype=torch.float32)
        self.range_max = torch.tensor(list(range_max), dtype=torch.float32)


    @torch.no_grad()
    def sample_globalprior_theta(self, shape):
        # Sample rho ~ Beta(1,1)
        rho_prior = dist.Beta(1, 1)
        rho = rho_prior.sample(shape)
        # regularization trick
        rho = 0.01 + 0.99 * rho  # [L, B, 1]

        # Sample alpha ~ Dirichlet(1,1,1)
        alpha_concentration = torch.ones(self.basket_dim)
        alpha_prior = dist.Dirichlet(alpha_concentration)
        alpha = alpha_prior.sample(shape)  # [L, B, 3]

        # Sample log(u) ~ N(1,3) then transform to u
        log_u_prior = dist.Normal(1, 3)
        log_u = log_u_prior.sample(shape)

        # Stack parameters into a single tensor [L, B, 5]
        theta = torch.cat([
            rho.unsqueeze(-1),  # [L, B, 1]
            alpha,  # [L, B, 3]
            log_u.unsqueeze(-1)  # [L, B, 1]
        ], dim=-1)  # [L, B, 5]

        return theta

    @torch.no_grad()
    def get_theta_shape(self, batch_size):
        return (batch_size, self.n_target_theta)

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """Sample basket pairs"""
        # Sample uniform baskets across the range [0, design_scale]
        # Each data point consists of two baskets (x, x')
        baskets1 = torch.rand(batch_size, n_data, self.basket_dim) * self.design_scale
        baskets2 = torch.rand(batch_size, n_data, self.basket_dim) * self.design_scale

        data = torch.cat([baskets1, baskets2], dim=-1)  # [B, N, 6]

        return data  # [B, N, 6]

    def utility(self, x, rho, alpha):
        """Calculate CES utility for a basket U(x) = (sum_i alpha_i * x_i^rho)^(1/rho)

        Args:
            x: basket of goods [B, basket_dim] or [B, T, basket_dim]
            rho: elasticity parameter [(L,) B, 1]
            alpha: weights for each good [(L,) B, (T), basket_dim] # or [B, 1, basket_dim]

        Returns:
            utility value [B, 1] or [B, T, 1]
        """
        x_pow_rho = x ** rho

        weighted_sum = torch.sum(alpha * x_pow_rho, dim=-1, keepdim=True)

        utility = weighted_sum ** (1. / rho)

        return utility


    def unnormalise_design(self, x):
        """Convert from normalized to real design space."""
        return x


    def normalise_design(self, x):
        """Normalize design if needed"""
        return x

    def normalise_outcomes(self, y):
        """Normalize preference ratings if needed"""
        return y

    def forward(self, xi, theta):
        """Generate preference rating

        Args:
            xi: basket pairs [B, 6] or [B, T, 6]
            theta: parameters [B, 5]

        Returns:
            preference rating [B, 1] or [B, T, 1]
        """
        rho = theta[..., 0:1]  # [B, (T), 1]
        alpha = theta[..., 1:4]  # [B, (T), 3]
        log_u = theta[..., 4:5]  # [B, (T), 1]
        u = torch.exp(log_u)

        xi = torch.clamp(xi, min=0.01, max=100.0)
        basket1 = xi[..., :self.basket_dim]  # [B, (T), 3]
        basket2 = xi[..., self.basket_dim:]  # [B, (T), 3]


        u1 = self.utility(basket1, rho, alpha)  # [B, 1] or [B, T, 1]
        u2 = self.utility(basket2, rho, alpha)  # [B, 1] or [B, T, 1]

        utility_diff = u1 - u2  # [B, 1] or [B, T, 1]

        mu_eta = utility_diff * u  # [B, 1] or [B, T, 1]

        basket_diff = basket1 - basket2
        basket_dist = torch.norm(basket_diff, dim=-1, p=2, keepdim=True)  # [B, 1] or [B, T, 1]
        sigma_eta = (1 + basket_dist) * self.noise_scale * u  # [B, 1] or [B, T, 1]

        y = CensoredSigmoidNormal(mu_eta, sigma_eta, self.epsilon, 1-self.epsilon).rsample()

        return y

    def log_likelihood(self, y, xi, theta):
        """Calculate log likelihood of observation

        Args:
            y: preference rating [1, B, (T), 1]
            xi: basket pairs [1, B, (T), 6]
            theta: parameters [L, B, (T), 5]

        Returns:
            log likelihood [L, B, (T), 1]
        """
        rho = theta[..., 0:1]  # [L, B, (T), 1]
        alpha = theta[..., 1:4]  # [L, B, (T), 3]
        log_u = theta[..., 4:5]  # [L, B, (T), 1]
        u = torch.exp(log_u)

        # Split the input into two baskets
        xi = torch.clamp(xi, min=0.01, max=100.0)
        basket1 = xi[..., :self.basket_dim]  # [1, B, T, 3]
        basket2 = xi[..., self.basket_dim:]  # [1, B, T, 3]

        u1 = self.utility(basket1, rho, alpha)  # [L, B, T, 1]
        u2 = self.utility(basket2, rho, alpha)  # [L, B, T, 1]

        utility_diff = u1 - u2  # [L, B, T, 1]

        mu_eta = utility_diff * u  # [L, B, T, 1]

        basket_diff = basket1 - basket2
        basket_dist = torch.norm(basket_diff, dim=-1, p=2, keepdim=True)  # [L, B, T, 1]
        sigma_eta = (1 + basket_dist) * self.noise_scale * u  # [L, B, T, 1]

        log_prob = CensoredSigmoidNormal(mu_eta, sigma_eta, self.epsilon, 1-self.epsilon).log_prob(y)

        return log_prob

    @torch.no_grad()
    def sample_batch(self, batch_size):
        """Sample a batch of normalised data"""
        theta = self.sample_globalprior_theta([batch_size])              # [B, 5]

        if not self.with_query:
            self.n_query_init = 1

        x = self.sample_data(batch_size, self.n_context_init + self.n_query_init) # [B, N, 6]
        # Generate responses for each basket pair parallely
        y = self.forward(x, theta.unsqueeze(-2)) # theta [B, N, 5]

        x = self.normalise_design(x)

        batch = AttrDict()
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:]
        batch.query_y = y[:, self.n_context_init:]
        batch.target_theta = theta.reshape(batch_size, self.n_target_theta, 1)
        batch.n_target_theta = self.n_target_theta

        batch.target_all = batch.target_theta

        return batch


    @torch.no_grad()
    def sample_correlated_batch(
        self,
        batch_size: int,
        n_source_tasks: int = 5,
        T: int = 30,
        prob_good_trajectory: float = 0.7,
        correlation_std: float = 0.1,
    ) -> AttrDict:
        """Sample target and auxiliary CES experiments.

        Good sources perturb target parameters with Gaussian noise scaled by
        range_span, then clip rho and project alpha onto the simplex.
        Bad sources use batch-shuffled target parameters, one shuffle per source.
        """
        if n_source_tasks == 0:
            batch = self.sample_batch(batch_size)
            batch.source_trajectories = AttrDict({'n_tasks': 0})
            return batch

        batch = self.sample_batch(batch_size)
        target_theta = batch.target_theta.reshape(batch_size, self.n_target_theta)  # [B, 5]
        target_theta_exp = target_theta.unsqueeze(1).expand(-1, n_source_tasks, -1)  # [B, n_src, 5]

        range_min_t = self.range_min.view(1, 1, self.n_target_theta)
        range_max_t = self.range_max.view(1, 1, self.n_target_theta)
        range_span = range_max_t - range_min_t

        # Good: target + small per-dim Gaussian noise
        noise = torch.randn_like(target_theta_exp) * correlation_std * range_span
        good_thetas = torch.clamp(target_theta_exp + noise, min=range_min_t, max=range_max_t)
        good_thetas[..., 0] = torch.clamp(good_thetas[..., 0], 0.01, 1.0)
        # alpha must remain on the simplex after additive noise.
        good_thetas[..., 1:4] = torch.abs(good_thetas[..., 1:4]) + 1e-6
        good_thetas[..., 1:4] = good_thetas[..., 1:4] / good_thetas[..., 1:4].sum(dim=-1, keepdim=True)

        # Bad sources: shuffle target parameters across the batch.
        bad_thetas = torch.stack(
            [target_theta[torch.randperm(batch_size)] for _ in range(n_source_tasks)],
            dim=1,
        )  # [B, n_src, 5]   — already on simplex / valid range by construction

        # Mix
        is_good = torch.bernoulli(
            torch.full((batch_size, n_source_tasks, 1), prob_good_trajectory)
        ).bool()
        source_thetas = torch.where(is_good, good_thetas, bad_thetas)

        # Reliability label: soft, based on normalised theta distance
        # γ chosen so that good (~0.05 dist_sq) maps to ~0.78 and shuffled-bad
        # (~5/6 dist_sq for 5-d uniform diff) maps to ~0.02.
        normalized_diff = (source_thetas - target_theta_exp) / range_span
        dist_sq = torch.sum(normalized_diff ** 2, dim=-1, keepdim=True)
        gamma = 5.0
        reliability_labels = torch.exp(-gamma * dist_sq)  # [B, n_src, 1]

        # Designs: uniform basket pairs in [0, design_scale] = [0, 100]
        baskets1 = torch.rand(batch_size, n_source_tasks, T, self.basket_dim) * self.design_scale
        baskets2 = torch.rand(batch_size, n_source_tasks, T, self.basket_dim) * self.design_scale
        source_x = torch.cat([baskets1, baskets2], dim=-1)  # [B, n_src, T, 6]

        flat_source_x = source_x.reshape(-1, self.dim_x)
        flat_source_thetas = source_thetas.unsqueeze(2).expand(
            -1, -1, T, -1).reshape(-1, self.n_target_theta)
        flat_source_y = self.forward(flat_source_x, flat_source_thetas)
        source_y = flat_source_y.reshape(batch_size, n_source_tasks, T, 1)

        batch.source_trajectories = AttrDict({
            'designs':            source_x,           # [B, n_src, T, 6]   in [0, 100]
            'observations':       source_y,           # [B, n_src, T, 1]
            'theta':              source_thetas,       # [B, n_src, 5]
            'n_tasks':            n_source_tasks,
            'reliability_labels': reliability_labels,  # [B, n_src, 1]
        })
        return batch


    @torch.no_grad()
    def sample_correlated_batch_eval(
        self,
        batch_size: int,
        n_source_tasks: int = 1,
        T: int = 30,
        prob_good_trajectory: float = 0.7,
        correlation_std: float = 0.1,
    ) -> AttrDict:
        """Compatibility wrapper for sample_correlated_batch."""
        return self.sample_correlated_batch(
            batch_size=batch_size,
            n_source_tasks=n_source_tasks,
            T=T,
            prob_good_trajectory=prob_good_trajectory,
            correlation_std=correlation_std,
        )


    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"CESTask({', '.join('{}={}'.format(key, val) for key, val in info.items() if not callable(val))})"


if __name__ == "__main__":
    task = CESTask()
    print(task.sample_batch(10))
