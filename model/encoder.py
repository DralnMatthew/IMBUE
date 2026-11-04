import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from typing import Optional, List


class TaskGatingNetwork(nn.Module):
    def __init__(self, dim_embedding: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_embedding, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim_embedding)
        self.ffn = nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_embedding * 2, dim_embedding)
        )
        self.norm2 = nn.LayerNorm(dim_embedding)
        
        self.scorer = nn.Sequential(
            nn.Linear(dim_embedding, dim_embedding // 2),
            nn.ReLU(),
            nn.Linear(dim_embedding // 2, 1) 
        )

    def forward(self, candidate_priors: torch.Tensor, context_embeddings: torch.Tensor):
        # Train the gate from detached embeddings.
        cp_sg = candidate_priors.detach()
        ce_sg = context_embeddings.detach()
        attended, _ = self.cross_attn(query=cp_sg, key=ce_sg, value=ce_sg)
        x = self.norm1(cp_sg + attended)
        x = self.norm2(x + self.ffn(x))

        logits = self.scorer(x)  # [B, N, 1]
        
        return candidate_priors, logits


class Encoder(nn.Module):
    def __init__(
            self,
            dim_embedding,
            dim_feedforward,
            n_head,
            dropout,
            num_layers,
            prior_injection: bool = True,
    ):
        super().__init__()
        self.prior_injection = prior_injection
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True 
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def create_mask(self, batch):
        if hasattr(batch, 'priors') and self.prior_injection:
            num_prior = batch.priors.shape[1]
        else:
            num_prior = 0

        num_context = batch.context_x.shape[1]
        num_query = batch.query_x.shape[1]
        num_target = batch.target_all.shape[1]
        num_all = num_prior + num_context + num_query + num_target

        mask = torch.zeros(num_all, num_all).fill_(float("-inf"))

        query_start = num_context + num_prior
        query_end = query_start + num_query
        target_start = query_end

        mask[:, :query_start] = 0.0

        mask[query_start:query_end, target_start:] = 0.0

        if num_prior == 0 and num_context == 0 and num_target > 0:
            idx = torch.arange(target_start, target_start + num_target)
            mask[idx, idx] = 0.0

        return mask

    def forward(self, batch, embeddings):
        mask = self.create_mask(batch)
        out = self.encoder(embeddings, mask=mask)
        return out


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical Encoder that interleaves Intra-Task and Inter-Task encoders.
    
    Flow:
    1. Intra-Task Encoder: Process each task's trajectory separately
    2. Inter-Task Encoder: Process relationships between tasks via [TASK] tokens
    3. Repeat N times
    """
    
    def __init__(
        self,
        dim_embedding: int,
        dim_feedforward: int,
        n_head: int,
        dropout: float,
        n_intra_layers: int = 3,
        n_inter_layers: int = 3,
        prior_injection: bool = True,
        use_trajectories: bool = False,
        n_summary_tokens: int = 8,
        gating: bool=False,
        gating_mode: str = "hard",
        **kwargs
    ):
        super().__init__()

        self.prior_injection = prior_injection
        self.use_trajectories = use_trajectories
        self.dim_embedding = dim_embedding
        self.n_summary_tokens = n_summary_tokens

        self.has_gating = (prior_injection or use_trajectories) and gating
        # "hard": drop priors with logits <= 0 via boolean key_padding_mask.
        # "soft": down-weight every prior in attention by log σ(logits) via
        #         float key_padding_mask (PyTorch adds it to attention scores).
        if gating_mode not in ("hard", "soft"):
            raise ValueError(f"gating_mode must be 'hard' or 'soft', got {gating_mode!r}")
        self.gating_mode = gating_mode

        if self.has_gating:
            self.gating_net = TaskGatingNetwork(dim_embedding, n_head, dropout)
        
        if self.use_trajectories:
            intra_layer = nn.TransformerEncoderLayer(
                d_model=dim_embedding, nhead=n_head, 
                dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
            )
            self.intra_encoder = nn.TransformerEncoder(intra_layer, num_layers=n_intra_layers)

        inter_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding, nhead=n_head, 
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.main_encoder = nn.TransformerEncoder(inter_layer, num_layers=n_inter_layers)
    

    def create_mask(self, n_priors, n_ctx, n_qry, n_tgt):
        num_all = n_priors + n_ctx + n_qry + n_tgt
        mask = torch.ones(num_all, num_all, dtype=torch.bool)

        query_start = n_priors + n_ctx
        query_end = query_start + n_qry
        target_start = query_end

        mask[:, :query_start] = False
        mask[query_start:query_end, target_start:] = False

        if n_priors == 0 and n_ctx == 0 and n_tgt > 0:
            idx = torch.arange(target_start, target_start + n_tgt)
            mask[idx, idx] = False

        return mask


    def _build_gating_mask(self, logits: Tensor, n_target_tokens: int) -> Tensor:
        # logits: [B, n_priors, 1]; mask: [B, n_priors + n_target_tokens].
        # Hard: mask logits <= 0. Soft: add log σ(logits) to attention scores.
        B = logits.shape[0]
        scores = logits.squeeze(-1)  # [B, n_priors]
        target_pad = torch.zeros(
            B, n_target_tokens, dtype=scores.dtype, device=scores.device,
        )
        if self.gating_mode == "hard":
            bad = (scores <= 0.0)
            return torch.cat([bad, target_pad.bool()], dim=1)
        # soft: log σ(logits) ∈ (-∞, 0]; clamp to avoid -inf for very low logits.
        prior_bias = torch.nn.functional.logsigmoid(scores).clamp(min=-30.0)
        return torch.cat([prior_bias, target_pad], dim=1)

    def forward(self, batch, target_base_embeddings: Tensor, prior_embeddings: Optional[Tensor] = None, source_embeddings: Optional[Tensor] = None) -> Tensor:
        B = target_base_embeddings.shape[0]
        n_ctx = batch.context_x.shape[1]
        n_qry = batch.query_x.shape[1]
        n_tgt = target_base_embeddings.shape[1] - n_ctx - n_qry
        
        context_emb = target_base_embeddings[:, :n_ctx, :]
        
        self.current_logits = None
        key_padding_mask = None
        n_priors = 0
        embeddings_to_encode = target_base_embeddings

        # With gating enabled, skip injection while the context is empty.
        skip_injection = self.has_gating and n_ctx == 0

        if self.prior_injection and prior_embeddings is not None and not skip_injection:
            n_priors = prior_embeddings.shape[1]
            if self.has_gating:
                modulated_priors, logits = self.gating_net(prior_embeddings, context_emb)
                self.current_logits = logits # [B, n_priors, 1]
                key_padding_mask = self._build_gating_mask(
                    logits, target_base_embeddings.shape[1]
                )
            else:
                modulated_priors = prior_embeddings

            embeddings_to_encode = torch.cat([modulated_priors, target_base_embeddings], dim=1)


        elif self.use_trajectories and source_embeddings is not None and not skip_injection:
            n_src = source_embeddings.shape[1]
            if n_src > 0:
                n_priors = n_src
                B, _, seq_len, dim = source_embeddings.shape

                flat_sources = source_embeddings.view(B * n_src, seq_len, dim)
                encoded_sources = self.intra_encoder(flat_sources) # [B * n_src, T+n_summary_token, dim]

                source_priors = encoded_sources[:, :self.n_summary_tokens, :].mean(dim=1).view(B, n_src, dim)

                if self.has_gating:
                    modulated_priors, logits = self.gating_net(source_priors, context_emb)
                    self.current_logits = logits
                    key_padding_mask = self._build_gating_mask(
                        logits, target_base_embeddings.shape[1]
                    )
                else:
                    modulated_priors = source_priors

                embeddings_to_encode = torch.cat([modulated_priors, target_base_embeddings], dim=1)

        attn_mask = self.create_mask(n_priors, n_ctx, n_qry, n_tgt)
        # Match the structural mask to the float padding mask.
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=key_padding_mask.dtype).masked_fill(
                attn_mask, float("-inf")
            )
        out = self.main_encoder(embeddings_to_encode, mask=attn_mask, src_key_padding_mask=key_padding_mask)

        if n_priors > 0:
            out = out[:, n_priors:, :]
            
        return out
