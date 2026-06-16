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
  - happy, excited -> happy
  - sad -> sad
  - angry, frustrated -> angry
  - other/minor/unclear/tie labels -> drop

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
- `dataset.py`: dataset loading, label mapping, collator.
- `b0_model.py`: B0 model definition.
- `train_b0.py`: B0 training loop.
- `evaluate_b0.py`: B0 evaluation.
- `infer_b0.py`: single-audio inference for B0.
- `metrics.py`: reusable metrics for future baselines.
- `train_b0.sh`, `evaluate_b0.sh`, `infer_b0.sh`: script entrypoints with editable parameters at the top.

Keep future baselines isolated:
- Add new model/training files for B1, B2, etc.
- Keep B0 stable as the mandatory reference baseline.
