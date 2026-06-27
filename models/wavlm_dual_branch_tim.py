from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class WavLMDualBranchTIMConfig:
    embedding_dim: int
    num_labels: int = 4
    temporal_feature_dim: int = 16
    temporal_emb_dim: int = 64
    memory_dim: int = 128
    dropout: float = 0.2
    alpha_init: float = 0.0
    beta_init: float = 0.0


class TemporalInteractionEncoder(nn.Module):
    def __init__(self, input_dim: int = 16, temporal_emb_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, temporal_emb_dim),
            nn.LayerNorm(temporal_emb_dim),
            nn.GELU(),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.net(temporal_features)


class DialogueMemoryBranch(nn.Module):
    """MAL-style causal read-before-write branch that receives only acoustic embeddings."""

    def __init__(self, embedding_dim: int, memory_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_projection = nn.Linear(embedding_dim, memory_dim)
        self.memory_cell = nn.GRUCell(memory_dim, memory_dim)
        self.readout = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, embedding_dim),
        )
        self.memory_dim = int(memory_dim)

    def initial_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.memory_dim, device=device, dtype=dtype)

    def read(self, embedding: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_i = self.input_projection(embedding)
        residual = self.readout(torch.cat([z_i, state], dim=-1))
        return residual, z_i

    def update(self, projected_input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.memory_cell(projected_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)


class TemporalMemoryBranch(nn.Module):
    """Causal temporal memory branch that receives only temporal interaction embeddings."""

    def __init__(self, temporal_emb_dim: int, embedding_dim: int, memory_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_projection = nn.Linear(temporal_emb_dim, memory_dim)
        self.memory_cell = nn.GRUCell(memory_dim, memory_dim)
        self.readout = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, embedding_dim),
        )
        self.memory_dim = int(memory_dim)

    def initial_state(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(self.memory_dim, device=device, dtype=dtype)

    def read(self, temporal_embedding: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_i = self.input_projection(temporal_embedding)
        residual = self.readout(torch.cat([z_i, state], dim=-1))
        return residual, z_i

    def update(self, projected_input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.memory_cell(projected_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)


class WavLMDualBranchTIMSerModel(nn.Module):
    def __init__(self, config: WavLMDualBranchTIMConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_encoder = TemporalInteractionEncoder(
            input_dim=config.temporal_feature_dim,
            temporal_emb_dim=config.temporal_emb_dim,
            dropout=config.dropout,
        )
        self.dialogue_branch = DialogueMemoryBranch(
            embedding_dim=config.embedding_dim,
            memory_dim=config.memory_dim,
            dropout=config.dropout,
        )
        self.temporal_branch = TemporalMemoryBranch(
            temporal_emb_dim=config.temporal_emb_dim,
            embedding_dim=config.embedding_dim,
            memory_dim=config.memory_dim,
            dropout=config.dropout,
        )
        self.alpha = nn.Parameter(torch.tensor(float(config.alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(config.beta_init)))
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
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings shape [num_utterances, embedding_dim], got {tuple(embeddings.shape)}.")
        if temporal_features.shape != (embeddings.shape[0], self.config.temporal_feature_dim):
            raise ValueError(
                "Expected temporal_features shape "
                f"({embeddings.shape[0]}, {self.config.temporal_feature_dim}), got {tuple(temporal_features.shape)}."
            )

        temporal_embeddings = self.temporal_encoder(temporal_features)
        dialogue_state = self.dialogue_branch.initial_state(embeddings.device, embeddings.dtype)
        temporal_state = self.temporal_branch.initial_state(embeddings.device, embeddings.dtype)
        logits: list[torch.Tensor] = []
        dialogue_residuals: list[torch.Tensor] = []
        temporal_residuals: list[torch.Tensor] = []
        fused_embeddings: list[torch.Tensor] = []

        alpha_gate = torch.tanh(self.alpha)
        beta_gate = torch.tanh(self.beta)
        for embedding, temporal_embedding in zip(embeddings, temporal_embeddings):
            dialogue_residual, dialogue_z = self.dialogue_branch.read(embedding, dialogue_state)
            temporal_residual, temporal_z = self.temporal_branch.read(temporal_embedding, temporal_state)
            fused = embedding + alpha_gate * dialogue_residual + beta_gate * temporal_residual
            logits.append(self.classifier(fused))
            dialogue_residuals.append(dialogue_residual)
            temporal_residuals.append(temporal_residual)
            fused_embeddings.append(fused)

            dialogue_state = self.dialogue_branch.update(dialogue_z, dialogue_state)
            temporal_state = self.temporal_branch.update(temporal_z, temporal_state)

        logits_tensor = torch.stack(logits, dim=0)
        dialogue_residual_tensor = torch.stack(dialogue_residuals, dim=0)
        temporal_residual_tensor = torch.stack(temporal_residuals, dim=0)
        output = {
            "logits": logits_tensor,
            "final_dialogue_state": dialogue_state,
            "final_temporal_state": temporal_state,
            "dialogue_residuals": dialogue_residual_tensor,
            "temporal_residuals": temporal_residual_tensor,
            "fused_embeddings": torch.stack(fused_embeddings, dim=0),
            "alpha_value": float(alpha_gate.detach().cpu().item()),
            "beta_value": float(beta_gate.detach().cpu().item()),
        }
        if labels is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits_tensor, labels)
        return output


def build_wavlm_dual_branch_tim_ser_model(model_cfg: dict, embedding_dim: int) -> WavLMDualBranchTIMSerModel:
    config = WavLMDualBranchTIMConfig(
        embedding_dim=int(embedding_dim),
        num_labels=int(model_cfg.get("num_labels", 4)),
        temporal_feature_dim=int(model_cfg.get("temporal_feature_dim", 16)),
        temporal_emb_dim=int(model_cfg.get("temporal_emb_dim", 64)),
        memory_dim=int(model_cfg.get("memory_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        alpha_init=float(model_cfg.get("alpha_init", 0.0)),
        beta_init=float(model_cfg.get("beta_init", 0.0)),
    )
    return WavLMDualBranchTIMSerModel(config)
