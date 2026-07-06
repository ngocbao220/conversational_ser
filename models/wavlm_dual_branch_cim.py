from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class WavLMDualBranchCIMConfig:
    embedding_dim: int
    num_labels: int = 4
    temporal_feature_dim: int = 16
    temporal_emb_dim: int = 64
    temporal_hidden_dim: int = 64
    memory_dim: int = 128
    dropout: float = 0.2
    alpha_init: float = 0.0
    beta_init: float = 0.0
    fusion_mode: str = "residual_gated"
    dialogue_memory_ablation_mode: str = "normal"
    interaction_memory_ablation_mode: str = "normal"
    dialogue_memory_shuffle_seed: int = 0
    interaction_memory_shuffle_seed: int = 0


class TemporalInteractionEncoder(nn.Module):
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
            nn.LayerNorm(temporal_emb_dim),
            nn.GELU(),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.net(temporal_features)


class DialogueMemoryBranch(nn.Module):
    """CDM-style causal read-before-write branch that receives only acoustic embeddings."""

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


class WavLMDualBranchCIMSerModel(nn.Module):
    SUPPORTED_FUSION_MODES = {
        "residual_gated",
        "residual_sum",
        "branch_sum",
        "branch_concat",
        "dialogue_only",
        "temporal_residual_sum",
    }

    def __init__(self, config: WavLMDualBranchCIMConfig) -> None:
        super().__init__()
        self.config = config
        if config.fusion_mode not in self.SUPPORTED_FUSION_MODES:
            supported = ", ".join(sorted(self.SUPPORTED_FUSION_MODES))
            raise ValueError(f"Unsupported fusion_mode={config.fusion_mode!r}. Supported values: {supported}.")
        supported_memory_ablation_modes = {"normal", "zero_state", "shuffled_order"}
        for name, mode in {
            "dialogue_memory_ablation_mode": config.dialogue_memory_ablation_mode,
            "interaction_memory_ablation_mode": config.interaction_memory_ablation_mode,
        }.items():
            if mode not in supported_memory_ablation_modes:
                supported = ", ".join(sorted(supported_memory_ablation_modes))
                raise ValueError(f"Unsupported {name}={mode!r}. Supported values: {supported}.")
        self.temporal_encoder = TemporalInteractionEncoder(
            input_dim=config.temporal_feature_dim,
            temporal_emb_dim=config.temporal_emb_dim,
            hidden_dim=config.temporal_hidden_dim,
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
        classifier_input_dim = config.embedding_dim * 2 if config.fusion_mode == "branch_concat" else config.embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(classifier_input_dim, config.num_labels),
        )

    def fuse(
        self,
        embedding: torch.Tensor,
        dialogue_residual: torch.Tensor,
        temporal_residual: torch.Tensor,
        alpha_gate: torch.Tensor,
        beta_gate: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.config.fusion_mode
        if mode == "residual_gated":
            return embedding + alpha_gate * dialogue_residual + beta_gate * temporal_residual
        if mode == "residual_sum":
            return embedding + dialogue_residual + temporal_residual
        if mode == "branch_sum":
            return dialogue_residual + temporal_residual
        if mode == "branch_concat":
            return torch.cat([dialogue_residual, temporal_residual], dim=-1)
        if mode == "dialogue_only":
            return dialogue_residual
        if mode == "temporal_residual_sum":
            return embedding + beta_gate * temporal_residual
        raise AssertionError(f"Unhandled fusion mode: {mode}")

    def _memory_branch_forward(
        self,
        branch: DialogueMemoryBranch | TemporalMemoryBranch,
        inputs: torch.Tensor,
        mode: str,
        shuffle_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = branch.initial_state(inputs.device, inputs.dtype)
        if mode == "shuffled_order" and inputs.shape[0] > 1:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(shuffle_seed) + int(inputs.shape[0]))
            order = torch.randperm(inputs.shape[0], generator=generator).to(inputs.device)
            inverse_order = torch.empty_like(order)
            inverse_order[order] = torch.arange(inputs.shape[0], device=inputs.device)
            ordered_inputs = inputs[order]
        else:
            inverse_order = None
            ordered_inputs = inputs

        residuals: list[torch.Tensor] = []
        for branch_input in ordered_inputs:
            if mode == "zero_state":
                state = torch.zeros_like(state)
            residual, projected_input = branch.read(branch_input, state)
            residuals.append(residual)
            state = branch.update(projected_input, state)

        residual_tensor = torch.stack(residuals, dim=0)
        if inverse_order is not None:
            residual_tensor = residual_tensor[inverse_order]
        return residual_tensor, state

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
        dialogue_residual_tensor, dialogue_state = self._memory_branch_forward(
            self.dialogue_branch,
            embeddings,
            mode=self.config.dialogue_memory_ablation_mode,
            shuffle_seed=self.config.dialogue_memory_shuffle_seed,
        )
        temporal_residual_tensor, temporal_state = self._memory_branch_forward(
            self.temporal_branch,
            temporal_embeddings,
            mode=self.config.interaction_memory_ablation_mode,
            shuffle_seed=self.config.interaction_memory_shuffle_seed,
        )

        alpha_gate = torch.tanh(self.alpha)
        beta_gate = torch.tanh(self.beta)
        fused_tensor = self.fuse(
            embeddings,
            dialogue_residual_tensor,
            temporal_residual_tensor,
            alpha_gate,
            beta_gate,
        )
        logits_tensor = self.classifier(fused_tensor)
        output = {
            "logits": logits_tensor,
            "final_dialogue_state": dialogue_state,
            "final_temporal_state": temporal_state,
            "dialogue_residuals": dialogue_residual_tensor,
            "temporal_residuals": temporal_residual_tensor,
            "fused_embeddings": fused_tensor,
            "fusion_mode": self.config.fusion_mode,
            "alpha_value": float(alpha_gate.detach().cpu().item()),
            "beta_value": float(beta_gate.detach().cpu().item()),
        }
        if labels is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits_tensor, labels)
        return output


def build_wavlm_dual_branch_cim_ser_model(model_cfg: dict, embedding_dim: int) -> WavLMDualBranchCIMSerModel:
    config = WavLMDualBranchCIMConfig(
        embedding_dim=int(embedding_dim),
        num_labels=int(model_cfg.get("num_labels", 4)),
        temporal_feature_dim=int(model_cfg.get("temporal_feature_dim", 16)),
        temporal_emb_dim=int(model_cfg.get("temporal_emb_dim", 64)),
        temporal_hidden_dim=int(model_cfg.get("temporal_hidden_dim", 64)),
        memory_dim=int(model_cfg.get("memory_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        alpha_init=float(model_cfg.get("alpha_init", 0.0)),
        beta_init=float(model_cfg.get("beta_init", 0.0)),
        fusion_mode=str(model_cfg.get("fusion_mode", "residual_gated")),
        dialogue_memory_ablation_mode=str(model_cfg.get("dialogue_memory_ablation_mode", model_cfg.get("memory_ablation_mode", "normal"))),
        interaction_memory_ablation_mode=str(
            model_cfg.get("interaction_memory_ablation_mode", model_cfg.get("temporal_memory_ablation_mode", "normal"))
        ),
        dialogue_memory_shuffle_seed=int(model_cfg.get("dialogue_memory_shuffle_seed", model_cfg.get("memory_shuffle_seed", 0))),
        interaction_memory_shuffle_seed=int(
            model_cfg.get("interaction_memory_shuffle_seed", model_cfg.get("temporal_memory_shuffle_seed", 0))
        ),
    )
    return WavLMDualBranchCIMSerModel(config)
