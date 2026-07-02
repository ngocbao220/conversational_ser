You are a Codex agent working on a PyTorch research codebase for conversational Speech Emotion Recognition (SER).

Implement Experiment 1: Baseline WavLM SER model with NO CDM and NO CIM.

Goal:
Build a clean utterance-level SER baseline:

- Input: utterance audio only.
- Backbone: WavLM from HuggingFace.
- No dialogue memory.
- No timestamp features.
- No temporal interaction modeling.
- Output: utterance-level emotion classification.

This experiment will be used as the control condition for later CDM and CIM experiments, so the implementation must be modular and reproducible.

Dataset assumptions:
Each sample should contain at least:

- audio path or waveform
- label
- dialogue_id
- utterance_id or turn_id
- speaker_id
- start_time
- end_time

For Experiment 1, load dialogue_id, speaker_id, start_time, and end_time into the dataset object, but DO NOT use them in the model. They are only needed for consistent data handling and later experiments.

Required implementation:

1. Dataset and dataloader

- Implement or adapt a dataset class named `ConversationalSERDataset`.
    The dataset is the Kaggle version of IEMOCAP:
    https://www.kaggle.com/datasets/sangayb/iemocap

    Each session contains separate dialogue transcript and sentences audio folders. I have downloaded this into 'iemocap' folder. Check it out.
    
    The transcript includes utterance timestamps in the format:

    [017.6000-020.6264]

    which correspond to:

    start_time = 17.6000
    end_time = 20.6264

    Parse these timestamps into floating-point values.
- Each dataset sample should be represented as:

{
    "audio_path": str,
    "label": int,
    "label_name": str,
    "session_id": int,
    "dialogue_id": str,
    "utterance_id": str,
    "speaker_id": str,
    "start_time": float,
    "end_time": float,
}

- Map labels as follows:
    ang -> angry
    hap -> happy
    exc -> happy
    sad -> sad
    neu -> neutral

- Discard all utterances with labels outside {ang, hap, exc, sad, neu}.
- Use Leave-One-Session-Out (LOSO) evaluation.

    The held-out session is specified by `TEST_SESSION`.

    Example:

    TEST_SESSION = 5

    Training sessions:
    1,2,3,4

    Validation:
    10% dialogue-level split from the training sessions. Do not split utterance from the same dialogue across train and validation. All utterances from one dialogue must belong to exactly one split.

    Test:
    Session 5.

    Allow changing the TEST_SESSION parameter.

- It should return:
  - `input_values`
  - `attention_mask`
  - `label`
  - `dialogue_id`
  - `utterance_id`
  - `speaker_id`
  - `start_time`
  - `end_time`
- Use HuggingFace `AutoFeatureExtractor` or `Wav2Vec2FeatureExtractor` compatible with WavLM. Resample audio to 16 kHz before feature extraction.
- Add a collate function that pads audio inputs properly.

2. Model
   Implement `WavLMSERBaseline`.

Architecture:

- WavLM encoder
- pooling layer over frame-level hidden states
- dropout
- linear classifier

Pooling options:

- default: attentive statistics pooling
- fallback: mean pooling over valid frames using attention mask

Forward signature:

```python
forward(input_values, attention_mask=None, labels=None, **metadata)
```

Do not use:

- dialogue_id
- speaker_id
- start_time
- end_time
- previous utterances
- memory state

Backbone fine-tuning options:
Add config flags:

- `freeze_wavlm: bool`
- `unfreeze_last_n_layers: int`

When freeze_wavlm=True,
all WavLM parameters except the classifier head must have requires_grad=False.

When unfreeze_last_n_layers>0,
only the last n Transformer encoder layers should be trainable.

Default for this experiment:

- `freeze_wavlm=True`
- `unfreeze_last_n_layers=0`

But the code should allow:

- frozen WavLM
- last-4-layer fine-tuning

3. Training
   Implement a training script or config entry for:
   `experiment_name = wavlm_baseline_no_cdm_no_cim`

Use yaml config.

Avoid hardcoded hyperparameters.

Default hyperparameters:

- learning rate classifier: 1e-4
- learning rate WavLM if unfrozen: 1e-5
- batch size: choose based on GPU memory (default is 16)
- optimizer: AdamW
- scheduler: linear warmup or cosine
- max epochs: 10
- gradient clipping: 1.0
- Use CrossEntropyLoss.
4. Metrics
   Report:

- WA: weighted accuracy / overall accuracy, computed as accuracy_score over all utterances.
- UA: unweighted accuracy, computed as macro-average recall / balanced_accuracy_score.
- Macro-F1: macro-average F1 over emotion classes.
- WF1: weighted-average F1 over emotion classes.
- per-class F1
- confusion matrix
    confusion_matrix.png - heatmap
    confusion_matrix.csv

Save outputs:

- `results/wavlm_baseline_no_cdm_no_cim/metrics.json`
    metrics.json should include

    epoch
    loss
    WA
    UA
    WF1
    Macro-F1
    per-class precision
    per-class recall
    per-class F1
- `results/wavlm_baseline_no_cdm_no_cim/predictions.csv`
- `results/wavlm_baseline_no_cdm_no_cim/config.json`
-  Save:
    best.pth according to the best validation UA
    last.pth

`predictions.csv` should contain:

- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time
- gold_label
- pred_label
- softmax probability for every class

5. Reproducibility

- Add seed setting for Python, NumPy, and PyTorch.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
- Log model parameter count.
- Log number of trainable parameters.
- Log whether WavLM is frozen or last-n layers are unfrozen.
- Using wandb and tqdm progress bar (config contain use_wandb=true for default)
    Log:
    training loss
    validation loss
    learning rate
    WA
    UA
    Macro-F1
    WF1

6. Acceptance criteria
   The code is correct if:

- Training runs end-to-end on the selected dataset.
- Evaluation produces WA, UA, WF1, Macro-F1.
- The model does not use timestamps, speakers, dialogue history, CDM, or CIM.
- The dataset and metadata are loaded consistently for later experiments.
- Results are saved in the specified directory.

7. Code structure

The implementation should be modular because later experiments will introduce:

- CDM
- CIM
- Test-time adaptation

Therefore:

Avoid experiment-specific assumptions inside shared modules.

Dataset should expose metadata even if unused.

Model forward should accept **metadata.

Training pipeline should not assume a specific model implementation.

The baseline should be implemented as the simplest special case of the future framework.