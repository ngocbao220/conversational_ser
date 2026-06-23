B0 — Utterance-Level Speech Emotion Baseline

Goal:
- Build the required utterance-level baseline for IEMOCAP.
- Do not use the SLLM/ADEPT pipeline in this baseline.
- Keep the code modular so later baselines can be added without rewriting B0.

Dataset:
- Load `AbstractTTS/IEMOCAP`.
- Each sample should contain utterance audio, transcript if available, original label, and mapped label.
- Map labels to 4 classes:
  - neutral -> neutral
  - happy, excited, surprise -> happy
  - sad, fear -> sad
  - angry, frustrated, disgust -> angry
  - `other` and `xxx`/unclear labels -> no supervised target

Baseline B0:
audio utterance
-> frozen WavLM/Wav2Vec2 encoder
-> pooling
-> classifier
-> emotion

Default B0 configuration:
- Encoder: `microsoft/wavlm-base`
- Encoder is frozen by default.
- Pooling: mean by default, attention pooling supported.
- Classifier: small MLP.
- Output classes: neutral, happy, sad, angry.

Evaluation:
- Compare predicted emotion with mapped gold label.
- Compute WA, UA, macro F1, WF1, and confusion matrix.
- Save metrics under `outputs/b0_utterance/`.

Code structure:
- `models/b0.py`: B0 model definition.
- `utils/dataset.py`: dataset loading, label mapping, collator.
- `utils/config.py`: CLI/config helpers.
- `utils/metrics.py`: reusable metrics for future baselines.
- `scripts/train_b0.py`: B0 training loop.
- `scripts/evaluate_b0.py`: B0 evaluation.
- `scripts/infer_b0.py`: single-audio inference for B0.
- `scripts/train_b0.sh`, `scripts/evaluate_b0.sh`, `scripts/infer_b0.sh`: script entrypoints with editable parameters at the top.

Keep future baselines isolated:
- Add new model/training files for B1, B2, etc.
- Keep B0 stable as the mandatory reference baseline.

B01:
- Same architecture family as B0.
- Unfreeze the last 4 SSL encoder transformer layers.
- Use LOSO split: train Ses01-Ses04, test Ses05.
- Entry points: `scripts/train_b01.sh`, `scripts/evaluate_b01.sh`.
