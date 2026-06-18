from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class WavLMMALConfig:
    embedding_dim: int
    num_labels: int = 4
    memory_dim: int = 256
    temporal_feature_dim: int = 16
    dropout: float = 0.2


class MALMemoryModule(nn.Module):
    """Causal read-before-write dialogue memory."""

    def __init__(self, embedding_dim: int, memory_dim: int, temporal_feature_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.memory_dim = int(memory_dim)
        self.temporal_feature_dim = int(temporal_feature_dim)
        self.input_projection = nn.Linear(self.embedding_dim, self.memory_dim)
        self.temporal_projection = nn.Linear(self.temporal_feature_dim, self.memory_dim, bias=False)
        self.memory_cell = nn.GRUCell(self.memory_dim, self.memory_dim)
        self.readout = nn.Sequential(
            nn.Linear(self.memory_dim * 2, self.memory_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.memory_dim, self.embedding_dim),
        )

    def initial_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.memory_dim, device=device, dtype=dtype)

    def forward(
        self,
        embeddings: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings shape [num_utterances, embedding_dim], got {tuple(embeddings.shape)}.")
        if temporal_features is None:
            temporal_features = embeddings.new_zeros((embeddings.shape[0], self.temporal_feature_dim))
        if temporal_features.shape != (embeddings.shape[0], self.temporal_feature_dim):
            raise ValueError(
                "Expected temporal_features shape "
                f"({embeddings.shape[0]}, {self.temporal_feature_dim}), got {tuple(temporal_features.shape)}."
            )

        state = initial_state if initial_state is not None else self.initial_state(embeddings.device, embeddings.dtype)
        memory_reads: list[torch.Tensor] = []
        for utterance_embedding, temporal_feature in zip(embeddings, temporal_features):
            z_i = self.input_projection(utterance_embedding) + self.temporal_projection(temporal_feature)
            memory_reads.append(self.readout(torch.cat([z_i, state], dim=-1)))
            state = self.memory_cell(z_i.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
        return torch.stack(memory_reads, dim=0), state


class WavLM_MALSerModel(nn.Module):
    def __init__(self, config: WavLMMALConfig) -> None:
        super().__init__()
        self.config = config
        self.memory = MALMemoryModule(
            embedding_dim=config.embedding_dim,
            memory_dim=config.memory_dim,
            temporal_feature_dim=config.temporal_feature_dim,
            dropout=config.dropout,
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.num_labels),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **metadata,
    ) -> dict:
        del metadata
        memory_read, final_state = self.memory(embeddings, temporal_features=temporal_features)
        fused = embeddings + torch.tanh(self.alpha) * memory_read
        logits = self.classifier(fused)
        output = {"logits": logits, "final_memory_state": final_state}
        if labels is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits, labels)
        return output


def build_wavlm_mal_ser_model(model_cfg: dict, embedding_dim: int) -> WavLM_MALSerModel:
    config = WavLMMALConfig(
        embedding_dim=int(embedding_dim),
        num_labels=int(model_cfg.get("num_labels", 4)),
        memory_dim=int(model_cfg.get("memory_dim", 256)),
        temporal_feature_dim=int(model_cfg.get("temporal_feature_dim", 16)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    return WavLM_MALSerModel(config)
