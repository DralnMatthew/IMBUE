import torch
import torch.nn as nn
from .embedder import Embedder
from .encoder import Encoder, HierarchicalEncoder
from .head import OutputHead
from attrdictionary import AttrDict


class BaseTransformer(nn.Module):
    """Transformer for design selection and posterior prediction."""

    def __init__(
        self, embedder: Embedder, encoder, head: OutputHead, use_trajectories: bool = False
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.head = head
        self.use_trajectories = use_trajectories

    def forward(
        self,
        batch: AttrDict,
        source_trajectories: AttrDict = None
    ) -> torch.Tensor:
        """Return head outputs and gating logits for the batch."""
        target_base_embeddings, prior_embeddings, source_embeddings = self.embedder(batch, source_trajectories)
        encoding = self.encoder(batch, target_base_embeddings, prior_embeddings, source_embeddings) 
        outs = self.head(batch, encoding)

        outs.gating_logits = self.encoder.current_logits
        
        return outs
