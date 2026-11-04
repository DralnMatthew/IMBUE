import torch
import numpy as np
import torch.nn as nn
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task
from omegaconf import ListConfig


class GPTask(Task):
    """GP active learning, evaluated by RMSE / log-likelihood at target_x.

    Supports auxiliary trajectories only. target_theta contains lengthscales
    and variance. The default kernel mixture excludes Matérn-1/2.
    """

    def __init__(
            self,
            name: str = "GP",
            dim_x: int = 1,             # dimension of input
            dim_y: int = 1,             # dimension of output
            n_context_init: int = 5,    # number of initial context points
            n_query_init: int = 10,     # number of initial query points
            n_target_theta: int = 2,    # must be dim_x + 1 (lengthscales, variance)
            n_target_data: int = 5,
            design_scale=None,
            noise_scale: float = 0.01,
            p_iso: float = 0.5,         # probability of isotropic kernel
            kernel_weights=None,        # default excludes Matérn-1/2
            lengthscale_lower=0.1,
            lengthscale_upper=2.0,
            with_query: bool = True,
            **kwargs                    # absorbs unused legacy keys
    ) -> None:
        super().__init__(dim_x=dim_x, dim_y=dim_y)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_theta = n_target_theta
        self.n_target_data = n_target_data

        self.jitter = 1e-5
        self.p_iso = p_iso
        # Matérn-1/2 weight 0 by design — excluded from the training family.
        self.kernel_weights = kernel_weights if kernel_weights is not None else [1/3, 0, 1/3, 1/3]
        self.kernel_types = ["rbf", "matern12", "matern32", "matern52"]

        if self.n_target_theta != dim_x + 1:
            raise ValueError("n_target_theta must equal dim_x + 1 (lengthscales + variance).")

        # Lengthscale prior — per-dim, scaled by sqrt(dim_x) so squared
        # distances are dimension-comparable.
        base_ls_factor = torch.sqrt(torch.tensor(dim_x, dtype=torch.float))
        self.lengthscale_lower = lengthscale_lower * base_ls_factor
        self.lengthscale_upper = lengthscale_upper * base_ls_factor

        # Variance prior
        self.scale_lower = 0.1
        self.scale_upper = 1.0

        self.noise_scale = noise_scale
        self.design_scale = torch.tensor(design_scale) if design_scale is not None else torch.tensor(5.0)
        self.with_query = with_query

    @torch.no_grad()
    def sample_theta(self, batch_size):
        """Sample hyperparameters from the prior, supporting both isotropic and anisotropic kernels"""
        # Sample per-dimension lengthscales
        ls_range = self.lengthscale_upper - self.lengthscale_lower
        length_scales = self.lengthscale_lower + ls_range * torch.rand(batch_size, self.dim_x)

        # Determine if each batch uses isotropic kernel (same lengthscale for all dimensions)
        is_iso = torch.bernoulli(torch.ones(batch_size) * self.p_iso).bool()

        new_length_scales = length_scales.clone()
        for b in range(batch_size):
            if is_iso[b]:
                new_length_scales[b, :] = length_scales[b, 0]
        length_scales = new_length_scales

        # Sample variance
        scale = self.scale_lower + (self.scale_upper - self.scale_lower) * torch.rand(batch_size)

        # Stack all parameters: [lengthscales, variance]
        theta = torch.cat([
            length_scales,  # [B, D]
            scale.unsqueeze(1),  # [B, 1]
        ], dim=1)

        return theta.unsqueeze(2)  # [B, D+1, 1]

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """Sample input points from the domain"""
        return torch.rand(batch_size, n_data, self.dim_x) * 2 * self.design_scale - self.design_scale

    def to_design_space(self, xi):
        """Convert normalized design to actual input domain"""
        return xi * self.design_scale

    def normalise_outcomes(self, y):
        """Normalize outcomes if needed"""
        return y

    @torch.no_grad()
    def get_theta_shape(self, batch_size):
        return (batch_size, self.n_target_theta)

    def rbf_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute RBF kernel between x1 and x2 with support for both isotropic and anisotropic kernels

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x] (can be all same for isotropic)
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting: [N, 1, dim_x] - [1, M, dim_x]
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        sq_dist = weighted_sq_diff.sum(dim=-1)  # [N, M]

        kernel = scale * torch.exp(-0.5 * sq_dist)

        return kernel  # [N, M]

    def matern12_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 1/2 kernel (exponential kernel) between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        kernel = scale * torch.exp(-dist)

        return kernel  # [N, M]

    def matern32_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 3/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        sqrt3 = torch.sqrt(torch.tensor(3.0, device=x1.device))
        kernel = scale * (1 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)

        return kernel  # [N, M]

    def matern52_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 5/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device))
        kernel = scale * (1 + sqrt5 * dist + (5.0 / 3.0) * dist ** 2) * torch.exp(-sqrt5 * dist)

        return kernel  # [N, M]

    def compute_kernel_matrix(self, x1, x2, lengthscales, scale, kernel_type):
        """
        Compute kernel matrix based on the specified kernel type

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)
            kernel_type: one of "rbf", "matern12", "matern32", "matern52"

        Returns:
            kernel matrix [N, M]
        """
        if kernel_type == "rbf":
            return self.rbf_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern12":
            return self.matern12_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern32":
            return self.matern32_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern52":
            return self.matern52_kernel(x1, x2, lengthscales, scale)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def sample_kernel_type(self, batch_size):
        """
        Sample kernel types for each batch element based on weights

        Args:
            batch_size: Number of batches to generate kernel types for

        Returns:
            List of kernel types, one for each batch element
        """
        weights = torch.tensor(self.kernel_weights, dtype=torch.float)
        weights = weights / weights.sum()

        indices = torch.multinomial(weights, batch_size, replacement=True)

        kernel_types = [self.kernel_types[idx] for idx in indices]

        return kernel_types

    def generate_gp_data(self, x, theta, fixed_kernel_types=None):
        """
        Generate GP data.
        """
        batch_size, n_points, _ = x.shape
        lengthscales = theta[:, :self.dim_x]  # [B, dim_x]
        scale = theta[:, self.dim_x]  # [B]

        # Use provided kernels or sample new ones
        if fixed_kernel_types is not None:
            kernel_types = fixed_kernel_types
        else:
            kernel_types = self.sample_kernel_type(batch_size)

        samples = torch.zeros(batch_size, n_points, 1)

        for b in range(batch_size):
            K = self.compute_kernel_matrix(
                x[b], x[b], lengthscales[b], scale[b], kernel_types[b]
            )
            K = K + self.jitter * torch.eye(n_points)

            try:
                L = torch.linalg.cholesky(K)
                z = torch.randn(n_points, device=x.device)
                f = torch.matmul(L, z)
            except:
                mvn = dist.MultivariateNormal(
                    loc=torch.zeros(n_points),
                    covariance_matrix=K
                )
                f = mvn.sample()

            # Add observation noise
            samples[b, :, 0] = f + self.noise_scale * torch.randn(n_points)

        return samples, kernel_types
    
    def forward(self, x, theta):
        """
        Generate observations from a GP

        Args:
            xi: normalized input locations [B, N, dim_x] or [B, dim_x]
            theta: GP hyperparameters [B, dim_x+2, 1]

        Returns:
            noisy observations [B, N, 1] or [B, 1]
        """
        # Check if input is for single points per batch or multiple points
        is_single_point_per_batch = len(x.shape) == 2

        if is_single_point_per_batch:
            # Handle single point per batch case
            batch_size = x.shape[0]
            x = x.unsqueeze(1)  # [B, 1, dim_x]

            y, _ = self.generate_gp_data(x, theta)

            return y.squeeze(1) # [B, 1]
        else:
            # Handle multiple points per batch case
            y, _ = self.generate_gp_data(x, theta)

            return y # [B, N, 1]


    def sample_batch(self, batch_size):
        if not self.with_query:
            self.n_query_init = 1
            
        batch = AttrDict()
        theta = self.sample_theta(batch_size)
        
        n_total = self.n_context_init + self.n_query_init + self.n_target_data
        x = self.sample_data(batch_size, n_total)

        # Generate data and keep track of kernels used
        y, kernel_types = self.generate_gp_data(x, theta)

        batch.context_x = x[:, :self.n_context_init]
        batch.context_y = y[:, :self.n_context_init]
        batch.query_x = x[:, self.n_context_init:self.n_context_init + self.n_query_init]
        batch.query_y = y[:, self.n_context_init:self.n_context_init + self.n_query_init]
        batch.target_x = x[:, self.n_context_init + self.n_query_init:]
        batch.target_y = y[:, self.n_context_init + self.n_query_init:]

        batch.target_theta = theta
        batch.target_all = batch.target_y
        
        batch.kernel_types = kernel_types 

        return batch


    @torch.no_grad()
    def sample_correlated_batch(
        self,
        batch_size: int,
        n_source_tasks: int = 1,
        T: int = 30,
        prob_good_trajectory: float = 0.7,
        bias_scale_range: list = [0.2, 2.0],
        mix_modes: bool = True,
    ) -> AttrDict:
        """Sample target and auxiliary GP trajectories.

        Good sources share the target GP with additive/multiplicative bias.
        Bad sources use separate GP draws with batch-shuffled target parameters.
        Reliability labels are the binary good/bad assignments.
        """
        target_batch = self.sample_batch(batch_size)

        if n_source_tasks == 0:
            target_batch.source_trajectories = AttrDict({'n_tasks': 0})
            return target_batch
        target_theta = target_batch.target_theta          # [B, n_target_theta, 1]
        target_theta_flat = target_theta.squeeze(-1)       # [B, n_target_theta]
        target_kernels = target_batch.kernel_types

        x_tgt_all = torch.cat([target_batch.context_x, target_batch.query_x, target_batch.target_x], dim=1)
        N_tgt = x_tgt_all.shape[1]

        # Good trajectories: joint GP sampling + additive/multiplicative bias
        N_src_total = n_source_tasks * T
        x_src_good = self.sample_data(batch_size, N_src_total)  # [B, N_src_total, dim_x]

        x_joint = torch.cat([x_tgt_all, x_src_good], dim=1)
        y_joint, _ = self.generate_gp_data(x_joint, target_theta_flat, fixed_kernel_types=target_kernels)

        y_tgt_new  = y_joint[:, :N_tgt, :]
        y_src_base = y_joint[:, N_tgt:, :].reshape(batch_size, n_source_tasks, T, 1)  # [B, n_src, T, 1]

        flat_bias_theta       = self.sample_theta(batch_size * n_source_tasks).squeeze(-1)
        flat_x_src_good       = x_src_good.reshape(batch_size * n_source_tasks, T, self.dim_x)
        flat_target_kernels   = [k for k in target_kernels for _ in range(n_source_tasks)]

        f_bias, _ = self.generate_gp_data(flat_x_src_good, flat_bias_theta, fixed_kernel_types=flat_target_kernels)
        f_bias = f_bias.reshape(batch_size, n_source_tasks, T, 1)

        scales = (bias_scale_range[1] - bias_scale_range[0]) * torch.rand(batch_size, n_source_tasks, 1, 1) + bias_scale_range[0]

        if mix_modes:
            mode_mask = torch.randint(0, 2, (batch_size, n_source_tasks, 1, 1)).float()
        else:
            mode_mask = torch.zeros((batch_size, n_source_tasks, 1, 1))

        y_src_good_final = (1 - mode_mask) * (y_src_base + scales * f_bias) + \
                           mode_mask * (y_src_base * (1.0 + scales * torch.tanh(f_bias)))  # [B, n_src, T, 1]
        x_src_good_res = x_src_good.reshape(batch_size, n_source_tasks, T, self.dim_x)      # [B, n_src, T, dim_x]

        # Bad trajectories: shuffle target thetas across the batch
        bad_theta = torch.stack(
            [target_theta_flat[torch.randperm(batch_size)] for _ in range(n_source_tasks)],
            dim=1,
        )  # [B, n_src, n_target_theta]  — each source task shuffled independently
        bad_theta_flat  = bad_theta.reshape(batch_size * n_source_tasks, self.n_target_theta)
        x_src_bad_flat  = self.sample_data(batch_size * n_source_tasks, T)
        y_src_bad_flat, _ = self.generate_gp_data(x_src_bad_flat, bad_theta_flat)
        y_src_bad = y_src_bad_flat.reshape(batch_size, n_source_tasks, T, 1)
        x_src_bad = x_src_bad_flat.reshape(batch_size, n_source_tasks, T, self.dim_x)

        # Mix good and bad
        is_good     = torch.bernoulli(torch.full((batch_size, n_source_tasks), prob_good_trajectory)).bool()  # [B, n_src]
        is_good_obs = is_good.unsqueeze(-1).unsqueeze(-1)                                      # [B, n_src, 1, 1]
        is_good_x   = is_good_obs.expand(batch_size, n_source_tasks, T, self.dim_x)

        y_src_final = torch.where(is_good_obs, y_src_good_final, y_src_bad)  # [B, n_src, T, 1]
        x_src_final = torch.where(is_good_x,   x_src_good_res,   x_src_bad)  # [B, n_src, T, dim_x]

        # Source parameters: target for good sources, batch-shuffled for bad.
        good_thetas   = target_theta_flat.unsqueeze(1).expand(-1, n_source_tasks, -1)  # [B, n_src, n_target_theta]
        source_thetas = torch.where(is_good.unsqueeze(-1), good_thetas, bad_theta)      # [B, n_src, n_target_theta]

        # Reliability labels
        reliability_labels = is_good.float().unsqueeze(-1)  # [B, n_src, 1]

        # Update target observations
        target_batch.context_y  = y_tgt_new[:, :self.n_context_init, :]
        target_batch.query_y    = y_tgt_new[:, self.n_context_init:self.n_context_init + self.n_query_init, :]
        target_batch.target_y   = y_tgt_new[:, self.n_context_init + self.n_query_init:, :]
        target_batch.target_all = target_batch.target_y

        source_trajectories = AttrDict({
            'designs':            x_src_final,        # [B, n_src, T, dim_x]
            'observations':       y_src_final,        # [B, n_src, T, 1]
            'theta':              source_thetas,       # [B, n_src, n_target_theta]
            'n_tasks':            n_source_tasks,
            'reliability_labels': reliability_labels,  # [B, n_src, 1]
        })

        target_batch.source_trajectories = source_trajectories
        return target_batch


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
