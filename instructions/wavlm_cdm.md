You are a Codex agent working on a PyTorch research codebase for conversational Speech Emotion Recognition (SER).

Implement Experiment 2: WavLM + CDM, with NO CIM.

Goal:
Build a dialogue-memory model that adds a Memory-as-a-Layer inspired residual memory module on top of WavLM utterance embeddings.

This experiment should test whether dialogue memory improves SER without using explicit timestamp or temporal interaction features.

Important comparison requirement:
Experiment 2 must be architecturally as close as possible to Experiment 3. The only difference should be:

* Experiment 2 receives no real temporal interaction features.
* Experiment 3 receives real temporal interaction features.

For fairness, implement the model so that Experiment 2 can pass a zero temporal feature vector of the same dimension used by Experiment 3. This keeps the interface stable and reduces confounds.

Dataset:
Use the same `ConversationalSERDataset` and collate function from Experiment 1.

Each sample contains:

* audio
* label
* dialogue_id
* utterance_id
* speaker_id
* start_time
* end_time

For Experiment 2:

* Use dialogue_id and utterance_id only to process utterances in the correct dialogue order.
* Use speaker_id only as metadata for logging; do not feed it into the model unless the baseline code already requires it.
* Do not use start_time or end_time as model input.
* Do not compute overlap, gap, duration, interruption, or other timing features.

Model:
Implement `WavLM_CDMSerModel`.

Pipeline:

1. Encode each utterance with WavLM.
2. Pool frame-level hidden states into an utterance embedding:
   `h_i`.
3. Pass `h_i` through a CDM-style dialogue memory module:
   `m_i, S_i = CDM(h_i, S_{i-1})`.
4. Apply residual update:
   `h_tilde_i = h_i + tanh(alpha) * m_i`.
5. Classify:
   `y_hat_i = classifier(h_tilde_i)`.

CDM module requirements:

* Name: `CDMMemoryModule`.
* It must maintain a dialogue-specific state.
* Reset memory state at the beginning of each dialogue.
* The memory update must be causal: prediction for utterance i can only use memory from utterances 1...i-1 and current utterance representation.
* Do not use future utterances.
* Implement a simple, stable CDM-lite memory first. Acceptable choices:

  * GRUCell-based memory
  * small recurrent MLP state
  * gated residual memory
* Do not implement full Titans test-time gradient memory unless it is already available in the repo.

Recommended implementation:

* Project utterance embedding to memory dimension:
  `z_i = W_in(h_i)`.
* Update memory:
  `S_i = GRUCell(z_i, S_{i-1})`.
* Read memory:
  `m_i = W_out(S_i)` or `m_i = W_out(concat(z_i, S_{i-1}))`.
* Residual gate:
  `h_tilde_i = h_i + tanh(alpha) * m_i`.
* Initialize `alpha = 0.0`.

Suggested dimensions:

* WavLM hidden dim: infer from backbone
* memory dim: 128 or 256
* dropout: 0.1 to 0.3

Training mode:
Because CDM state depends on dialogue order, implement one of the following:
Option A:

* Batch utterances by dialogue.
* Process utterances sequentially within each dialogue.
  Option B:
* Use a custom batch sampler that keeps dialogue order.
  Option C:
* Precompute WavLM embeddings first, then train the CDM model dialogue-by-dialogue.

If using standard shuffled utterance batches, do not maintain memory across shuffled samples. That would be incorrect.

Recommended for simplicity:

* Add a mode `precompute_wavlm_embeddings=True`.
* Save utterance embeddings from the frozen WavLM baseline.
* Train CDM on precomputed embeddings sorted by dialogue_id and utterance_id.
* Keep an option to fine-tune last 4 WavLM layers later, but default to frozen WavLM.

Loss:

* Cross entropy.
* Optional class weights if dataset is imbalanced.

Metrics:
Report the same metrics as Experiment 1:

* WA
* UA
* WF1
* Macro-F1
* per-class F1
* confusion matrix

Save outputs:

* `results/wavlm_cdm_no_cim/metrics.json`
* `results/wavlm_cdm_no_cim/predictions.csv`
* `results/wavlm_cdm_no_cim/config.json`
* best checkpoint

`predictions.csv` should contain:

* dialogue_id
* utterance_id
* speaker_id
* start_time
* end_time
* gold_label
* pred_label
* probability for each class

Ablation protection:
Add a config field:

```yaml
use_temporal_features: false
temporal_feature_mode: zero
```

When `temporal_feature_mode = zero`, create a zero vector with the same dimensionality that Experiment 3 will use, but do not put timestamp information in it.

Acceptance criteria:
The code is correct if:

* The model uses WavLM utterance embeddings plus dialogue memory.
* The model does not use start_time, end_time, duration, gap, overlap, or any timing feature.
* Memory state resets at dialogue boundaries.
* Prediction is causal and does not use future utterances.
* Results can be compared directly with Experiment 1 and Experiment 3.
