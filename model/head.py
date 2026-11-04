import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.distributions.utils import broadcast_all
from typing import Optional, Dict, Any, List, Tuple
from attrdictionary import AttrDict
from distributions import MixtureOfGaussians
from attrdictionary import AttrDict
from omegaconf import ListConfig


class ValueHead(nn.Module):
    """Value function head for critic"""
    def __init__(self, dim_embedding: int, dim_feedforward: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_embedding: [B, N, dim_embedding]
        Returns:
            value: [B]
        """
        state_embedding = z.mean(dim=1) # [B, dim_embedding]
        value = self.value_head(state_embedding) # [B, 1]
        value = value.squeeze(-1) # [B]

        return value


class AcquisitionHead(nn.Module):
    def __init__(self, dim_embedding: int, dim_feedforward: int, **kwargs: Any) -> None:
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),  # [B, N, dim_feedforward]
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1), # [B, N, 1]
            nn.Flatten(start_dim=-2),  # [B, N]
            nn.Softmax(dim=-1) # [B, N]
        )

    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Embedding feature tensor [B, N, dim_embedding]
        Returns:
            acquisition_scores: Acquisition scores [B, N]
        """
        return self.predictor(z)


class ContinuousAcquisitionHead(nn.Module):
    def __init__(self, dim_embedding: int, dim_feedforward: int, dim_x: int, **kwargs: Any) -> None:
        super().__init__()

        self.mean_predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_x)
        )
        self.log_std_predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_x),
        )

        nn.init.uniform_(self.mean_predictor[-1].weight, -0.005, 0.005)
        nn.init.constant_(self.mean_predictor[-1].bias, 0)

        nn.init.constant_(self.log_std_predictor[-1].weight, 0.0)
        nn.init.constant_(self.log_std_predictor[-1].bias, -2.0) # bias 0 -> std = 1.0
    
    def forward(self, z):
        """
        Args:
            z: Embedding feature tensor [B, N, dim_embedding]
        Returns:
            mean: Mean of Gaussian policy [B, dim_x]
            std: Standard deviation of Gaussian policy [B, dim_x]
        """
        mean = self.mean_predictor(z)                   # [B, n_query=1, dim_x]
        log_std = self.log_std_predictor(z)             # [B, n_query=1, dim_x]
        log_std = torch.clamp(log_std, min=-10, max=2)
        
        std = log_std.exp()
        
        return mean, std


class GMMTargetHead(nn.Module):
    def __init__(
        self,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int,
        single_head: bool = False,
        std_min: float = 1e-4,
        **kwargs: Any
        ) -> None:
        
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward
        self.dim_y = dim_y
        self.single_head = single_head
        self.num_components = num_components
        self.heads = self.initialize_head(self.dim_embedding, self.dim_feedforward, self.dim_y, self.single_head,
                                          num_components)

        self.std_min = std_min

    def forward(self, batch: AttrDict, z_target: torch.Tensor) -> AttrDict:

        if self.single_head:
            output = self.heads(z_target) # [B, n_target, dim_y * 2 or dim_y * 3 * components]
            if self.num_components == 1:
                raw_mean, raw_std = torch.chunk(output, 2, dim=-1) # [B, n_target, dim_y], [B, n_target, dim_y]
                raw_weights = torch.ones_like(raw_mean)
            else:
                raw_mean, raw_std, raw_weights = torch.chunk(output, 3, dim=-1) # [B, n_target, dim_y * components], [B, n_target, dim_y * components], [B, n_target, dim_y * components]
        else:
            outputs = [head(z_target) for head in self.heads]  # list of [B, n_target, dim_y * 3] * components
            raw_mean, raw_std, raw_weights = self._map_raw_output(outputs) # [B, n_target, dim_y * components], [B, n_target, dim_y * components], [B, n_target, dim_y * components]

        mean = raw_mean
        std = F.softplus(raw_std) + self.std_min
        weights = F.softmax(raw_weights, dim=-1)

        gmm = MixtureOfGaussians(mean, std, weights) # [B, n_target/K*dim_x, dim_y*components/10]
            
        outs = AttrDict()

        # also outputs params
        outs.mixture_means = mean
        outs.mixture_stds = std
        outs.mixture_weights = weights
        outs.dist = gmm

        return outs
    
    def initialize_head(self,
                        dim_embedding: int,
                        dim_feedforward: int,
                        dim_y: int,
                        single_head: bool,
                        num_components: int,
                        ) -> nn.Module:
        """Build a shared head or one head per mixture component."""
        if single_head:
            if num_components > 1:
                model = nn.Sequential(
                    nn.Linear(dim_embedding, dim_feedforward), # [B, n_target, dim_feedforward]
                    nn.ReLU(), 
                    nn.Linear(dim_feedforward, dim_y * 3 * num_components), # [B, n_target, dim_y * 3 * components]
                )
            else:
                model = nn.Sequential(
                    nn.Linear(dim_embedding, dim_feedforward), # [B, n_target, dim_feedforward]
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, dim_y * 2), # [B, n_target, dim_y * 2]
                )
        else:
            model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim_embedding, dim_feedforward),
                        nn.ReLU(),
                        nn.Linear(dim_feedforward, dim_y * 3),
                    )
                    for _ in range(num_components)
                ]
            )
        return model

    @staticmethod
    def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Computes log-likelihood loss for a Gaussian mixture model.
        """
        components = Normal(means, stds, validate_args=False)
        log_probs = components.log_prob(value)
        weighted_log_probs = log_probs + torch.log(weights)
        return torch.logsumexp(weighted_log_probs, dim=-1)

    @staticmethod
    def _map_raw_output(outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concatenated = torch.stack(outputs).movedim(0, -1).flatten(-2, -1) # [B, T, dim_y * 3 * components]
        raw_mean, raw_std, raw_weights = torch.chunk(concatenated, 3, dim=-1) # 3 x [B, T, components]
        return raw_mean, raw_std, raw_weights


class OutputHead(nn.Module):
    """Acquisition and posterior heads for query and target embeddings."""
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int = 10,
        single_head: bool = False,
        value_head:bool = False,
        std_min: float = 1e-4,
        with_query: bool = True,
        design_range_min: int = 0.0, 
        design_range_max: int = 1.0,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.design_range_min = self._to_tensor(design_range_min, self.dim_x)
        self.design_range_max = self._to_tensor(design_range_max, self.dim_x)

        self.with_query = with_query
        
        # Acquisition head for design selection
        if self.with_query:
            self.acquisition_head = AcquisitionHead(
                dim_embedding=dim_embedding,
                dim_feedforward=dim_feedforward,
            )
        else:
            self.acquisition_head = ContinuousAcquisitionHead(
                dim_embedding=dim_embedding,
                dim_feedforward=dim_feedforward,
                dim_x=dim_x,
            )
        
        # Target head for posterior prediction
        self.target_head = GMMTargetHead(
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            num_components=num_components,
            single_head=single_head,
            std_min=std_min,
        )

        if value_head:
            self.value_head = ValueHead(
                dim_embedding=dim_embedding,
                dim_feedforward=dim_feedforward
            )
    

    def _to_tensor(self, val, target_size):
        """Convert a scalar or list to a (target_size,) tensor."""
        if isinstance(val, (float, int)):
            return torch.full((target_size,), val, dtype=torch.float32)
        if isinstance(val, (list, tuple, ListConfig)):
            val = torch.tensor(val, dtype=torch.float32)
        if isinstance(val, torch.Tensor):
            val = val.float()

        if val.numel() == 1:
            return val.expand(target_size)
            
        assert val.shape == (target_size,), f"Expected shape ({target_size},), got {val.shape}"
        return val

    
    def forward(self, batch: AttrDict, z: torch.Tensor) -> AttrDict:
        """
        Process batch by splitting into context, query and target parts
        
        Args:
            batch: Batch containing context and target tasks
            z: Embedding tensor [B, N, dim_embedding] where N = n_context + n_query + n_target
                Context embeddings are on the left, query in the middle, target on the right
            
        Returns:
            outs: AttrDict containing acquisition and posterior prediction results
        """
        batch_size = z.shape[0]

        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]

        context_end = n_context
        query_start = context_end
        query_end = query_start + n_query
        target_start = query_end

        z_context = z[:, :context_end]
        z_query = z[:, query_start:query_end]
        z_target = z[:, target_start:]

        if hasattr(self, 'value_head'):
            value = self.value_head(z_context) # [B]
        else:
            value = torch.zeros(batch_size) # Fallback

        # Continuous
        if not self.with_query:             
            mean, std = self.acquisition_head(z_query)  # [B, n_query=1, dim_x]
            # print("mean, std:", mean, std)
            dist = Normal(mean, std)  # [B, n_query=1, dim_x]
            
            # Select design with the highest probability during inference, sample during training
            if self.training:
                x_query = dist.sample() # [B, n_query=1, dim_x]
            else:
                x_query = mean # [B, n_query=1, dim_x]

            scale = 0.5 * (self.design_range_max - self.design_range_min) # [dim_x]

            scale_expanded = scale.unsqueeze(0).unsqueeze(0) # [B, 1, dim_x]
            min_expanded = self.design_range_min.unsqueeze(0).unsqueeze(0) # [B, 1, dim_x]

            a_query = min_expanded + scale_expanded * (torch.tanh(x_query) + 1.0)
            log_prob = dist.log_prob(x_query)  # [B, 1, dim_x]
            correction = 2 * (torch.log(torch.tensor(2.0)) - x_query - F.softplus(-2 * x_query)) # [B, n_query=1, dim_x]
            log_prob -= correction

            log_prob = log_prob.sum(-1) - (torch.log(scale_expanded).sum(-1))
            log_prob = log_prob.squeeze(-1) # [B]

            estimated_entropy = -log_prob # [B]

            raw_x_query = x_query
            x_query = a_query

            raw_log_prob = dist.log_prob(raw_x_query).sum(-1).squeeze(-1) # [B]
            
            posterior_out = self.target_head(batch, z_target)  # [B, n_target, dim_y]
            posterior_out_query = self.target_head(batch, z_query)

            z_context = z[:, :context_end]
            value = self.value_head(z_context)
            
            outs = AttrDict(
                posterior_out_query=posterior_out_query,
                posterior_out=posterior_out,
                guide_dist=posterior_out.dist,
                design_out=AttrDict(
                    xi=x_query,       # [B, n_query=1, dim_x]
                    raw_xi=raw_x_query, # [B, n_query=1, dim_x]
                    log_prob=log_prob,  # [B]
                    raw_log_prob=raw_log_prob, # [B]
                    mean=mean,      # [B, n_query=1, dim_x]
                    std=std,         # [B, n_query=1, dim_x]
                    dist=dist,  # [B, n_query=1, dim_x]
                    entropy=estimated_entropy, # [B]
                ),
                value=value   #[B]
            )
        
        # Discrete
        else:
            zt = self.acquisition_head(z_query)  # [B, n_query]
        
            # Select design with the highest probability during inference, sample during training
            if self.training:
                m_design = Categorical(zt)
                idx_next = m_design.sample()  # [B]
                log_prob = m_design.log_prob(idx_next)
                entropy = m_design.entropy() # [B]
            else:
                log_prob, idx_next = torch.max(zt, -1)
                log_prob = torch.log(log_prob)
                entropy = torch.zeros_like(log_prob)
            
            idx_next = idx_next.unsqueeze(1) # [B, 1]
            
            posterior_out = self.target_head(batch, z_target)
            
            outs = AttrDict(
                posterior_out=posterior_out,
                design_out=AttrDict(
                    idx=idx_next,       # [B, 1]
                    log_prob=log_prob,  # [B]
                    entropy=entropy,    # [B]
                    zt=zt               # [B, N_query]
                ),
                value=value,             # [B] Critic Value
            )

        return outs
