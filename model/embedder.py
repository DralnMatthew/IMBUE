import torch
import torch.nn as nn
from typing import Any
from attrdictionary import AttrDict


class Embedder(nn.Module):
    """    
    The embedder takes a batch with context, query, and target tasks,
    and produces embeddings in the order: context, query, target.
    """
    
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        prior_injection: bool = True,
        num_bins: int = 100,
        n_target_theta: int = 0,
        with_query: bool = True,
        use_trajectories: bool = False,
        n_summary_tokens: int = 1,
        embedding_type: str = "theta",
        **kwargs: Any
    ) -> None:
        """
        Initialize the embedder module
        
        Args:
            dim_x: Dimension of x tasks
            dim_y: Dimension of y tasks
            dim_embedding: Dimension of embedding vectors
            n_target_theta: Number of latent variables (theta)
            n_summary_tokens: Number of summary tokens for trajectories and context
        """
        super().__init__()

        self.prior_injection = prior_injection
        self.with_query = with_query
        self.use_trajectories = use_trajectories
        self.n_summary_tokens = n_summary_tokens
        self.embedding_type = embedding_type

        if prior_injection and num_bins > 0:
            self.p_embedder = nn.Sequential(
                nn.Linear(num_bins, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, dim_embedding)
            )
        
        self.x_embedder = nn.Sequential(
            nn.Linear(dim_x, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        self.y_embedder = nn.Sequential(
            nn.Linear(dim_y, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        if embedding_type == 'theta':
            self.theta_tokens = nn.Parameter(torch.randn(n_target_theta, dim_embedding))

        if not self.with_query:
            self.query_tokens = nn.Parameter(torch.randn(1, dim_embedding))
        
        if use_trajectories:
            self.task_token = nn.Parameter(torch.randn(n_summary_tokens, dim_embedding))
    
    
    def forward(self, batch: AttrDict, source_trajectories: AttrDict = None) -> torch.Tensor:
        B = batch.context_x.shape[0]
        n_ctx = batch.context_x.shape[1]
        n_qry = batch.query_x.shape[1]
        
        x_context_query = torch.cat([batch.context_x, batch.query_x], dim=1) # [B, n_ctx + n_qry, dim_x]
        x_embeddings = self.x_embedder(x_context_query) # [B, n_ctx + n_qry, dim]
        y_embeddings_context = self.y_embedder(batch.context_y) # [B, n_ctx, dim]
        
        context_emb = x_embeddings[:, :n_ctx] + y_embeddings_context
        query_emb = x_embeddings[:, n_ctx:] if self.with_query else self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 1, dim]
        theta_emb = self.theta_tokens.unsqueeze(0).expand(B, -1, -1) if self.embedding_type == 'theta' else self.x_embedder(batch.target_x)
        
        target_base_embeddings = torch.cat([context_emb, query_emb, theta_emb], dim=1)

        prior_embeddings = None
        source_embeddings = None

        if self.prior_injection and hasattr(batch, 'priors'):
            prior_embeddings = self.p_embedder(batch.priors)
            if hasattr(batch, 'prior_mask') and batch.prior_mask is not None:
                prior_embeddings = prior_embeddings + theta_emb[:, batch.prior_mask, :]
            else:
                prior_embeddings = prior_embeddings + theta_emb
                
        elif self.use_trajectories and source_trajectories is not None:
            n_src = source_trajectories.n_tasks
            if n_src > 0:
                xy_emb = self.x_embedder(source_trajectories.designs) + self.y_embedder(source_trajectories.observations) # [B, n_src, T, dim]
                source_task_tokens = self.task_token.reshape(1, 1, self.n_summary_tokens, -1).expand(B, n_src, -1, -1) # [B, n_src, n_summary_tokens, dim]
                source_embeddings = torch.cat([source_task_tokens, xy_emb], dim=2) # [B, n_src, n_summary_tokens+T, dim]
                
        return target_base_embeddings, prior_embeddings, source_embeddings
