from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.iemocap_kaggle import ConversationalSERCollator, ConversationalSERDataset, ConversationSERSample


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
