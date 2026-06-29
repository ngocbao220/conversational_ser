from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class WavLMTIMConfig:
    embedding_dim: int
    num_labels: int = 4
    temporal_feature_dim: int = 16
    temporal_emb_dim: int = 64
    temporal_hidden_dim: int = 64
    memory_dim: int = 256
    dropout: float = 0.2
    residual_gate_init: float = 0.0


class TemporalFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        temporal_emb_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, temporal_emb_dim),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.net(temporal_features)


class TIMMemoryModule(nn.Module):
    """Causal read-before-write temporal interaction memory."""

    def __init__(self, input_dim: int, output_dim: int, memory_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, memory_dim)
        self.memory_cell = nn.GRUCell(memory_dim, memory_dim)
        self.readout = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, output_dim),
        )
        self.memory_dim = int(memory_dim)

    def initial_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.memory_dim, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError(f"Expected inputs shape [num_utterances, input_dim], got {tuple(inputs.shape)}.")
        state = initial_state if initial_state is not None else self.initial_state(inputs.device, inputs.dtype)
        memory_reads: list[torch.Tensor] = []
        for item in inputs:
            z_i = self.input_projection(item)
            memory_reads.append(self.readout(torch.cat([z_i, state], dim=-1)))
            state = self.memory_cell(z_i.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
        return torch.stack(memory_reads, dim=0), state


class WavLMTIMSerModel(nn.Module):
    def __init__(self, config: WavLMTIMConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_encoder = TemporalFeatureEncoder(
            input_dim=config.temporal_feature_dim,
            temporal_emb_dim=config.temporal_emb_dim,
            hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
        )
        self.memory = TIMMemoryModule(
            input_dim=config.embedding_dim + config.temporal_emb_dim,
            output_dim=config.embedding_dim,
            memory_dim=config.memory_dim,
            dropout=config.dropout,
        )
        self.alpha = nn.Parameter(torch.tensor(float(config.residual_gate_init)))
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.num_labels),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        temporal_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **metadata,
    ) -> dict:
        del metadata
        if temporal_features.shape != (embeddings.shape[0], self.config.temporal_feature_dim):
            raise ValueError(
                "Expected temporal_features shape "
                f"({embeddings.shape[0]}, {self.config.temporal_feature_dim}), got {tuple(temporal_features.shape)}."
            )
        temporal_embeddings = self.temporal_encoder(temporal_features)
        memory_inputs = torch.cat([embeddings, temporal_embeddings], dim=-1)
        memory_read, final_state = self.memory(memory_inputs)
        fused = embeddings + torch.tanh(self.alpha) * memory_read
        logits = self.classifier(fused)
        output = {"logits": logits, "final_memory_state": final_state}
        if labels is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits, labels)
        return output


def build_wavlm_tim_ser_model(model_cfg: dict, embedding_dim: int) -> WavLMTIMSerModel:
    config = WavLMTIMConfig(
        embedding_dim=int(embedding_dim),
        num_labels=int(model_cfg.get("num_labels", 4)),
        temporal_feature_dim=int(model_cfg.get("temporal_feature_dim", 16)),
        temporal_emb_dim=int(model_cfg.get("temporal_emb_dim", 64)),
        temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 64)),
        memory_dim=int(model_cfg.get("memory_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        residual_gate_init=float(model_cfg.get("residual_gate_init", 0.0)),
    )
    return WavLMTIMSerModel(config)
