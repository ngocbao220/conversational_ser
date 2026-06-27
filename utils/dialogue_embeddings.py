from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.iemocap_kaggle import (
    ConversationalSERCollator,
    ConversationalSERDataset,
    ConversationSERSample,
    load_audio_mono,
)


@dataclass(frozen=True)
class DialogueEmbedding:
    dialogue_id: str
    embeddings: torch.Tensor
    labels: torch.Tensor
    rows: List[Dict[str, Any]]


def sort_samples_for_dialogue(samples: Sequence[ConversationSERSample]) -> List[ConversationSERSample]:
    return sorted(samples, key=lambda sample: (sample.dialogue_id, sample.start_time, sample.end_time, sample.utterance_id))


def group_samples_by_dialogue(samples: Sequence[ConversationSERSample]) -> Dict[str, List[ConversationSERSample]]:
    grouped: Dict[str, List[ConversationSERSample]] = {}
    for sample in sort_samples_for_dialogue(samples):
        grouped.setdefault(sample.dialogue_id, []).append(sample)
    return grouped


def _feature_attention_mask(model: torch.nn.Module, attention_mask: Optional[torch.Tensor], feature_length: int) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if hasattr(model, "_get_feat_extract_output_lengths"):
        lengths = model._get_feat_extract_output_lengths(attention_mask.sum(dim=1)).to(torch.long)
        mask = torch.zeros((attention_mask.shape[0], feature_length), device=attention_mask.device, dtype=torch.long)
        for index, length in enumerate(lengths):
            mask[index, : min(int(length), feature_length)] = 1
        return mask
    return torch.nn.functional.interpolate(
        attention_mask[:, None].float(), size=feature_length, mode="nearest"
    ).squeeze(1).long()


def mean_pool_hidden_states(hidden_states: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return hidden_states.mean(dim=1)
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)


def precompute_wavlm_mean_embeddings(
    samples: Sequence[ConversationSERSample],
    wavlm_model_name: str,
    sampling_rate: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_duration_seconds: Optional[float] = None,
    progress: bool = True,
) -> Dict[str, Dict[str, Any]]:
    from transformers import AutoFeatureExtractor, AutoModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(wavlm_model_name)
    collator = ConversationalSERCollator(feature_extractor, sampling_rate=sampling_rate)
    dataset = ConversationalSERDataset(
        sort_samples_for_dialogue(samples),
        sampling_rate=sampling_rate,
        max_duration_seconds=max_duration_seconds,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    model = AutoModel.from_pretrained(wavlm_model_name).to(device)
    model.eval()

    rows_by_utterance: Dict[str, Dict[str, Any]] = {}
    iterator = tqdm(dataloader, desc="precompute fixed mean-pooled WavLM embeddings", disable=not progress, dynamic_ncols=True)
    with torch.no_grad():
        for batch in iterator:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            feature_mask = _feature_attention_mask(model, attention_mask, hidden_states.shape[1])
            embeddings = mean_pool_hidden_states(hidden_states, feature_mask).detach().cpu()
            for index, utterance_id in enumerate(batch["utterance_id"]):
                row = {
                    "embedding": embeddings[index],
                    "label": int(batch["labels"][index].item()),
                    "audio_path": batch["audio_path"][index],
                    "dialogue_id": batch["dialogue_id"][index],
                    "utterance_id": utterance_id,
                    "speaker_id": batch["speaker_id"][index],
                    "start_time": float(batch["start_time"][index].item()),
                    "end_time": float(batch["end_time"][index].item()),
                    "label_name": batch["label_name"][index],
                }
                rows_by_utterance[utterance_id] = row
    return rows_by_utterance


def build_dialogue_embeddings(
    samples: Sequence[ConversationSERSample],
    embeddings_by_utterance: Mapping[str, Mapping[str, Any]],
) -> List[DialogueEmbedding]:
    dialogues: List[DialogueEmbedding] = []
    for dialogue_id, dialogue_samples in group_samples_by_dialogue(samples).items():
        rows: List[Dict[str, Any]] = []
        embeddings: List[torch.Tensor] = []
        labels: List[int] = []
        for sample in dialogue_samples:
            row = dict(embeddings_by_utterance[sample.utterance_id])
            row.update(
                {
                    "audio_path": sample.audio_path,
                    "dialogue_id": sample.dialogue_id,
                    "utterance_id": sample.utterance_id,
                    "speaker_id": sample.speaker_id,
                    "start_time": sample.start_time,
                    "end_time": sample.end_time,
                    "label_name": sample.label_name,
                }
            )
            rows.append(row)
            embeddings.append(torch.as_tensor(row["embedding"], dtype=torch.float32))
            labels.append(int(sample.label))
        dialogues.append(
            DialogueEmbedding(
                dialogue_id=dialogue_id,
                embeddings=torch.stack(embeddings, dim=0),
                labels=torch.tensor(labels, dtype=torch.long),
                rows=rows,
            )
        )
    return sorted(dialogues, key=lambda dialogue: dialogue.dialogue_id)


def build_audio_dialogues(samples: Sequence[ConversationSERSample], embedding_dim: int) -> List[DialogueEmbedding]:
    dialogues: List[DialogueEmbedding] = []
    for dialogue_id, dialogue_samples in group_samples_by_dialogue(samples).items():
        rows: List[Dict[str, Any]] = []
        labels: List[int] = []
        for sample in dialogue_samples:
            rows.append(sample.to_dict())
            labels.append(int(sample.label))
        dialogues.append(
            DialogueEmbedding(
                dialogue_id=dialogue_id,
                embeddings=torch.zeros((len(rows), int(embedding_dim)), dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.long),
                rows=rows,
            )
        )
    return sorted(dialogues, key=lambda dialogue: dialogue.dialogue_id)


class TrainableWavLMMeanExtractor(torch.nn.Module):
    """Mean-pooled WavLM utterance encoder for end-to-end dialogue training."""

    def __init__(
        self,
        wavlm_model_name: str,
        sampling_rate: int = 16000,
        max_duration_seconds: float | None = None,
        freeze_wavlm: bool = True,
        unfreeze_last_n_layers: int = 0,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

        self.wavlm_model_name = str(wavlm_model_name)
        self.sampling_rate = int(sampling_rate)
        self.max_audio_length = (
            int(float(max_duration_seconds) * self.sampling_rate)
            if max_duration_seconds is not None and float(max_duration_seconds) > 0
            else None
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.wavlm_model_name)
        self.wavlm = AutoModel.from_pretrained(self.wavlm_model_name)
        self.hidden_size = int(getattr(AutoConfig.from_pretrained(self.wavlm_model_name), "hidden_size"))

        if freeze_wavlm:
            for parameter in self.wavlm.parameters():
                parameter.requires_grad = False
            self._unfreeze_last_layers(int(unfreeze_last_n_layers))
        self.wavlm_fully_frozen = not any(parameter.requires_grad for parameter in self.wavlm.parameters())

    def _unfreeze_last_layers(self, num_layers: int) -> None:
        if num_layers <= 0:
            return
        layers = getattr(getattr(self.wavlm, "encoder", None), "layers", None)
        if layers is None:
            raise ValueError(f"Cannot locate WavLM encoder layers for {self.wavlm_model_name!r}.")
        for layer in layers[-num_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _feature_attention_mask(self, attention_mask: torch.Tensor | None, feature_length: int) -> torch.Tensor | None:
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
    def _mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        return torch.sum(hidden_states * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)

    def _load_waveforms(self, rows: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        waveforms = []
        for row in rows:
            waveform = load_audio_mono(str(row["audio_path"]), self.sampling_rate)
            if self.max_audio_length is not None and waveform.shape[0] > self.max_audio_length:
                waveform = waveform[: self.max_audio_length]
            waveforms.append(np.asarray(waveform, dtype=np.float32))
        return waveforms

    def encode_rows(self, rows: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int = 4) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            encoded = self.feature_extractor(
                self._load_waveforms(batch_rows),
                sampling_rate=self.sampling_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_values = encoded["input_values"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if self.wavlm_fully_frozen:
                with torch.no_grad():
                    outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)
            else:
                outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            feature_mask = self._feature_attention_mask(attention_mask, hidden_states.shape[1])
            embeddings.append(self._mean_pool(hidden_states, feature_mask))
        return torch.cat(embeddings, dim=0)


def save_embedding_cache(path: str | Path, rows_by_utterance: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_rows = {
        utterance_id: {key: value for key, value in row.items()}
        for utterance_id, row in rows_by_utterance.items()
    }
    torch.save({"metadata": dict(metadata), "rows_by_utterance": serializable_rows}, path)


def load_embedding_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def sample_metadata(samples: Iterable[ConversationSERSample]) -> List[Dict[str, Any]]:
    return [asdict(sample) for sample in samples]
