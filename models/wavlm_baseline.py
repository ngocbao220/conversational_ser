from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass(frozen=True)
class WavLMSERBaselineConfig:
    wavlm_model_name: str = "microsoft/wavlm-base"
    num_labels: int = 4
    pooling: str = "attentive_statistics"
    dropout: float = 0.2
    freeze_wavlm: bool = True
    unfreeze_last_n_layers: int = 0


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(hidden_states).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = torch.sum(hidden_states * weights, dim=1)
        variance = torch.sum(((hidden_states - mean.unsqueeze(1)) ** 2) * weights, dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-6))
        return torch.cat([mean, std], dim=-1)


class WavLMSERBaseline(nn.Module):
    def __init__(self, config: WavLMSERBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.wavlm = AutoModel.from_pretrained(config.wavlm_model_name)
        wavlm_config = AutoConfig.from_pretrained(config.wavlm_model_name)
        self.hidden_size = int(getattr(wavlm_config, "hidden_size"))

        if config.freeze_wavlm:
            for parameter in self.wavlm.parameters():
                parameter.requires_grad = False
        if config.unfreeze_last_n_layers > 0:
            self._unfreeze_last_layers(config.unfreeze_last_n_layers)
        self.wavlm_fully_frozen = not any(parameter.requires_grad for parameter in self.wavlm.parameters())

        if config.pooling == "attentive_statistics":
            self.pooler = AttentiveStatisticsPooling(self.hidden_size)
            pooled_size = self.hidden_size * 2
        elif config.pooling == "mean":
            self.pooler = None
            pooled_size = self.hidden_size
        else:
            raise ValueError(f"Unsupported pooling={config.pooling!r}.")

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(pooled_size, config.num_labels),
        )

    def _unfreeze_last_layers(self, num_layers: int) -> None:
        layers = getattr(getattr(self.wavlm, "encoder", None), "layers", None)
        if layers is None:
            raise ValueError("Cannot locate WavLM transformer layers for unfreezing.")
        for layer in layers[-int(num_layers) :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _feature_attention_mask(self, attention_mask: Optional[torch.Tensor], feature_length: int) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if hasattr(self.wavlm, "_get_feat_extract_output_lengths"):
            lengths = self.wavlm._get_feat_extract_output_lengths(attention_mask.sum(dim=1)).to(torch.long)
            mask = torch.zeros((attention_mask.shape[0], feature_length), device=attention_mask.device, dtype=torch.long)
            for index, length in enumerate(lengths):
                mask[index, : min(int(length), feature_length)] = 1
            return mask
        return torch.nn.functional.interpolate(
            attention_mask[:, None].float(), size=feature_length, mode="nearest"
        ).squeeze(1).long()

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **metadata,
    ) -> dict:
        del metadata
        if self.wavlm_fully_frozen:
            with torch.no_grad():
                outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        else:
            outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        feature_mask = self._feature_attention_mask(attention_mask, hidden_states.shape[1])
        if self.config.pooling == "attentive_statistics":
            pooled = self.pooler(hidden_states, feature_mask)
        else:
            pooled = self._mean_pool(hidden_states, feature_mask)
        logits = self.classifier(pooled)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = torch.nn.functional.cross_entropy(logits, labels)
        return result


def build_wavlm_ser_baseline(model_cfg: dict) -> WavLMSERBaseline:
    config = WavLMSERBaselineConfig(
        wavlm_model_name=str(model_cfg.get("wavlm_model_name", "microsoft/wavlm-base")),
        num_labels=int(model_cfg.get("num_labels", 4)),
        pooling=str(model_cfg.get("pooling", "attentive_statistics")),
        dropout=float(model_cfg.get("dropout", 0.2)),
        freeze_wavlm=bool(model_cfg.get("freeze_wavlm", True)),
        unfreeze_last_n_layers=int(model_cfg.get("unfreeze_last_n_layers", 0)),
    )
    return WavLMSERBaseline(config)
