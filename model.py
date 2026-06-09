from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.score(hidden_states).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return torch.sum(hidden_states * weights, dim=1)


class SERModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        pooling: str = "mean",
        freeze_encoder: bool = True,
        dropout: float = 0.2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_config = AutoConfig.from_pretrained(encoder_name)
        self.pooling = pooling
        self.hidden_size = int(getattr(encoder_config, "hidden_size"))

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        if pooling == "attention":
            self.attention_pooling = AttentionPooling(self.hidden_size)
        elif pooling != "mean":
            raise ValueError(f"Unsupported pooling={pooling!r}. Use 'mean' or 'attention'.")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def _feature_attention_mask(self, attention_mask: Optional[torch.Tensor], feature_length: int) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(dim=1)).to(torch.long)
            mask = torch.zeros((attention_mask.shape[0], feature_length), device=attention_mask.device, dtype=torch.long)
            for idx, length in enumerate(lengths):
                mask[idx, : min(int(length), feature_length)] = 1
            return mask
        return torch.nn.functional.interpolate(
            attention_mask[:, None].float(), size=feature_length, mode="nearest"
        ).squeeze(1).long()

    def _mean_pool(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / counts

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        feature_mask = self._feature_attention_mask(attention_mask, hidden_states.shape[1])
        if self.pooling == "attention":
            pooled = self.attention_pooling(hidden_states, feature_mask)
        else:
            pooled = self._mean_pool(hidden_states, feature_mask)
        return self.classifier(pooled)
