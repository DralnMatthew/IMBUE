import torch
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task


class PsychometricTask(Task):
    """Simulate Psychometric Function Experiment"""

    def __init__(
            self,
            name: str = "Psychometric",
            dim_x: int = 1,  # dimension of stimulus intensity
            dim_y: int = 1,  # dimension of outcome (binary)
            range_min=[-3, 0.1, 0.0, 0.0],
            range_max=[3, 2, 0.15, 0.06],
            n_target_theta: int = 4,  # number of parameters: [alpha, beta, gamma, lambda]
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 300,  # number of initial query points
            design_scale: int = 5,  # scale of the design space
            with_query: bool = True,    # whether to include query points
            prior_injection: bool = False,  # read by the shared model configs; no prior channel here
            **kwargs
    ) -> None:
        super(PsychometricTask, self).__init__(dim_x=dim_x, dim_y=dim_y, n_target_theta=n_target_theta)

        if prior_injection:
            raise ValueError("PsychometricTask has no prior channel; set task.prior_injection=False.")
        self.prior_injection = False

        self.dim_x = dim_x
        self.n_target_theta = n_target_theta
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.design_scale = design_scale
        self.with_query = with_query

        # Parameter ranges of [alpha, beta, gamma, lambda]; the global prior is
        # uniform on them.
        self.range_min = torch.tensor(list(range_min), dtype=torch.float32)
        self.range_max = torch.tensor(list(range_max), dtype=torch.float32)


    @torch.no_grad()
    def sample_globalprior_theta(self, shape):
        """Draw theta from the uniform global prior (bounds match range_min/max)."""
        alpha_prior  = dist.Uniform(-3.0, 3.0)
        beta_prior   = dist.Uniform(0.1, 2.0)
        gamma_prior  = dist.Uniform(0.0, 0.15)
        lambda_prior = dist.Uniform(0.0, 0.06)

        alpha = alpha_prior.sample(shape).unsqueeze(-1)
        beta = beta_prior.sample(shape).unsqueeze(-1)
        gamma = gamma_prior.sample(shape).unsqueeze(-1)
        lmbda = lambda_prior.sample(shape).unsqueeze(-1)
        return torch.cat([alpha, beta, gamma, lmbda], dim=-1)


    @torch.no_grad()
    def get_theta_shape(self, batch_size):
        return (batch_size, self.n_target_theta)


    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """Sample uniform stimuli in [-design_scale, design_scale]."""
        data = torch.rand(batch_size, n_data, self.dim_x) * 2 * self.design_scale - self.design_scale

        return data  # [B, N, 1]

    def psychometric_function(self, x, theta):
        """Evaluate the psychometric function

        Args:
            x: stimulus intensity [B, 1] or [B, T, 1]
            theta: parameters [B, 4, 1]

        Returns:
            probability of correct response [B, 1] or [B, T, 1]
        """
        if len(theta.shape) == 2:
            theta = theta.unsqueeze(-1)  # [B, 4] -> [B, 4, 1]

        alpha = theta[:, 0, :]  # threshold [B, 1]
        beta = theta[:, 1, :]  # slope
        gamma = theta[:, 2, :]  # guess rate
        lmbda = theta[:, 3, :]  # lapse rate

        # Calculate internal value using Gumbel function (like in Prins 2013)
        z = (x - alpha) / beta
        F = 1 - torch.exp(-10 ** (z))  # Gumbel function

        # Apply guessing and lapsing
        p = lmbda*gamma + (1 - lmbda) * F

        return p

    def normalise_design(self, x):
        """Convert from real to normalized design space."""
        return x

    def unnormalise_design(self, x):
        """Convert from normalized to real design space."""
        return x


    def normalise_outcomes(self, y):
        """Normalize binary outcomes if needed"""
        return y

    def forward(self, xi, theta):
        """Generate binary response

        Args:
            xi: normalized stimulus intensity [B, 1]
            theta: parameters [B, 4, 1]

        Returns:
            binary response [B, 1]
        """
        p = self.psychometric_function(xi, theta)

        y = torch.bernoulli(p)

        return y

    def log_likelihood(self, y, xi, theta):
        """Calculate log likelihood of observation

        Args:
            y: binary response [B, 1] or [L, B, T, 1]
            xi: stimulus intensity [B, 1] or [L, B, T, 1]
            theta: parameters [B, 4, 1] or [L, B, 4, 1]

        Returns:
            log likelihood [B, 1] or [L, B, T, 1]
        """
        p = self.psychometric_function(xi, theta)

        log_prob = y * torch.log(p + 1e-10) + (1 - y) * torch.log(1 - p + 1e-10)

        return log_prob

    def sample_batch(self, batch_size):
        """Sample a batch of data"""
        theta = self.sample_globalprior_theta([batch_size])               # [B, 4]
        theta = theta.reshape(batch_size, self.n_target_theta, 1)         # [B, 4, 1]

        if not self.with_query:
            self.n_query_init = 1

        x = self.sample_data(batch_size, self.n_context_init + self.n_query_init) # [B, N, 1]
        y = torch.empty(batch_size, self.n_context_init + self.n_query_init, self.dim_y)

        for i in range(self.n_context_init + self.n_query_init):
            y[:, i, :] = self.forward(x[:, i, :], theta)

        batch = AttrDict()
        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:]
        batch.query_y = y[:, self.n_context_init:]
        batch.target_theta = theta
        batch.n_target_theta = self.n_target_theta

        batch.target_all = batch.target_theta

        return batch


    @torch.no_grad()
    def sample_correlated_batch(
        self,
        batch_size: int,
        n_source_tasks: int = 1,
        T: int = 30,
        prob_good_trajectory: float = 0.7,
        correlation_std: float = 0.1,
    ) -> AttrDict:
        batch = self.sample_batch(batch_size)

        if n_source_tasks == 0:
            batch.source_trajectories = AttrDict({'n_tasks': 0})
            return batch

        target_theta = batch.target_theta.reshape(batch_size, self.n_target_theta)   # [B, 4]
        target_theta_exp = target_theta.unsqueeze(1).expand(-1, n_source_tasks, -1)  # [B, n_src, 4]

        range_min_t = self.range_min.view(1, 1, self.n_target_theta)
        range_max_t = self.range_max.view(1, 1, self.n_target_theta)
        range_span  = range_max_t - range_min_t

        # Good sources: small perturbation around target theta
        noise = torch.randn_like(target_theta_exp) * correlation_std * range_span
        good_thetas = torch.clamp(target_theta_exp + noise, min=range_min_t, max=range_max_t)

        # Shuffle target parameters across the batch for each bad source.
        bad_thetas = torch.stack(
            [target_theta[torch.randperm(batch_size)] for _ in range(n_source_tasks)],
            dim=1,
        )  # [B, n_src, 4]

        is_good = torch.bernoulli(
            torch.full((batch_size, n_source_tasks, 1), prob_good_trajectory)
        ).bool()  # [B, n_src, 1]

        source_thetas = torch.where(is_good, good_thetas, bad_thetas)  # [B, n_src, 4]

        # Reliability labels from normalised theta distance (same formula as CES)
        normalized_diff = (source_thetas - target_theta_exp) / (range_span + 1e-8)
        dist_sq = torch.sum(normalized_diff ** 2, dim=-1, keepdim=True)  # [B, n_src, 1]
        reliability_labels = torch.exp(-4.0 * dist_sq)                    # [B, n_src, 1]

        source_x = torch.rand(batch_size, n_source_tasks, T, self.dim_x) * 2 * self.design_scale - self.design_scale

        flat_source_x      = source_x.reshape(-1, self.dim_x)
        flat_source_thetas = source_thetas.unsqueeze(2).expand(-1, -1, T, -1).reshape(-1, self.n_target_theta)
        flat_source_y      = self.forward(flat_source_x, flat_source_thetas)
        source_y           = flat_source_y.reshape(batch_size, n_source_tasks, T, 1)

        source_trajectories = AttrDict({
            'designs':            source_x,           # [B, n_src, T, 1]
            'observations':       source_y,           # [B, n_src, T, 1]
            'theta':              source_thetas,       # [B, n_src, 4]
            'n_tasks':            n_source_tasks,
            'reliability_labels': reliability_labels,  # [B, n_src, 1]
        })

        batch.source_trajectories = source_trajectories
        return batch


    @torch.no_grad()
    def sample_correlated_batch_eval(
        self,
        batch_size: int,
        n_source_tasks: int = 1,
        T: int = 30,
        prob_good_trajectory: float = 0.7,
    ) -> AttrDict:
        """Eval wrapper — delegates to sample_correlated_batch."""
        return self.sample_correlated_batch(
            batch_size=batch_size,
            n_source_tasks=n_source_tasks,
            T=T,
            prob_good_trajectory=prob_good_trajectory,
        )


    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"PsychometricTask({', '.join('{}={}'.format(key, val) for key, val in info.items())})"


if __name__ == "__main__":
    task = PsychometricTask()
    print(task.sample_batch(10))
