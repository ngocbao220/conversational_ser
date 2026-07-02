You are a Codex agent working on a PyTorch research codebase for conversational Speech Emotion Recognition (SER).

Implement Experiment 3: WavLM + CIM.

Goal:
Build the proposed Temporal Interaction Memory model. This model should be identical to Experiment 2 as much as possible, except that it uses explicit temporal interaction features derived from utterance start/end times and dialogue turn structure.

This is the main contribution experiment.

Research hypothesis:
Utterance boundary relations encode conversational dynamics such as overlap, interruption, response latency, speaker switching, and speaker-specific turn-taking habits. These temporal interaction cues provide complementary information for conversational SER beyond ordinary dialogue memory.

Important comparison requirement:
Experiment 3 must reuse the same WavLM encoder, pooling, classifier structure, CDM-style residual gate, memory dimension, training setup, and evaluation pipeline as Experiment 2.

The key difference:

* Experiment 2 uses zero temporal vectors.
* Experiment 3 uses real temporal interaction vectors.

Dataset:
Each sample contains:

* audio
* label
* dialogue_id
* utterance_id
* speaker_id
* start_time
* end_time

The dataset must be sorted by:

1. dialogue_id
2. start_time or utterance_id

Temporal feature extraction:
Implement a feature builder named `TemporalInteractionFeatureBuilder`.

For each utterance i, compute only causal features. Do not use future utterances.

Required features:

1. `duration_i = end_time_i - start_time_i`
2. `gap_prev_i = start_time_i - end_time_{i-1}` within the same dialogue
3. `overlap_prev_i = max(0, end_time_{i-1} - start_time_i)`
4. `overlap_ratio_i = overlap_prev_i / max(duration_i, eps)`
5. `is_overlap_i = 1 if overlap_prev_i > threshold else 0`
6. `is_interrupting_prev_i = 1 if speaker_i != speaker_{i-1} and start_time_i < end_time_{i-1} else 0`
7. `speaker_switch_i = 1 if speaker_i != speaker_{i-1} else 0`
8. `same_speaker_i = 1 if speaker_i == speaker_{i-1} else 0`
9. `turn_index_norm_i = turn_index_i / dialogue_length`, if dialogue_length is allowed as metadata; otherwise use turn_index_i normalized by a fixed max length.
10. `prev_gap_abs_i = abs(gap_prev_i)`
11. `short_response_i = 1 if 0 <= gap_prev_i < short_gap_threshold else 0`
12. `long_pause_i = 1 if gap_prev_i > long_gap_threshold else 0`

Speaker-habit features:
Compute these causally from previous utterances only:
13. `speaker_prev_overlap_rate`
14. `speaker_prev_mean_gap`
15. `speaker_prev_mean_duration`
16. `speaker_prev_turn_count_norm`

Do not compute speaker statistics using the entire dialogue before prediction, because that leaks future information.

Feature normalization:

* Normalize continuous features using statistics from the training split only.
* Save the normalization stats to:
  `results/wavlm_cim/temporal_feature_stats.json`
* Apply the same stats to dev/test.

Temporal encoder:
Implement `TemporalFeatureEncoder`.

* Input: temporal feature vector.
* Architecture: small MLP.
* Suggested:

  * Linear(input_dim, 64)
  * LayerNorm
  * GELU/ReLU
  * Dropout
  * Linear(64, temporal_emb_dim)
* Suggested temporal_emb_dim: 64 or 128.

CIM model:
Implement `WavLMCIMSerModel`.

Pipeline:

1. Encode utterance audio with WavLM.
2. Pool frame-level hidden states into utterance embedding:
   `h_i`.
3. Encode temporal features:
   `t_i = TemporalFeatureEncoder(tau_i)`.
4. Combine:
   `z_i = concat(h_i, t_i)`.
5. Feed into temporal interaction memory:
   `m_i, S_i = CIM(z_i, S_{i-1})`.
6. Apply residual update:
   `h_tilde_i = h_i + tanh(alpha) * m_i`.
7. Classify:
   `y_hat_i = classifier(h_tilde_i)`.

CIM memory:
Use the same underlying memory mechanism as Experiment 2:

* GRUCell-based memory or recurrent MLP state.
* Same memory dimension.
* Same residual gate initialization: `alpha = 0.0`.
* Same classifier.
* Same dropout if possible.

The only real difference from Experiment 2 should be the non-zero temporal feature vector.

Causality requirement:
Prediction for utterance i may use:

* current audio
* current utterance start/end time
* previous utterance metadata
* previous speaker-habit statistics
* previous memory state

Prediction for utterance i must NOT use:

* future utterance labels
* future utterance audio
* future start/end times for determining whether current utterance is interrupted later
* full-dialogue speaker statistics that include future utterances

Training:
Use the same training setup as Experiment 2:

* Cross entropy
* AdamW
* early stopping on dev Macro-F1 or WF1
* gradient clipping 1.0
* mixed precision if available
* default WavLM frozen
* support optional last-4-layer fine-tuning

Config:
Add:

```yaml
experiment_name: wavlm_cim
use_cdm_memory: true
use_temporal_features: true
temporal_feature_mode: real
temporal_emb_dim: 64
memory_dim: 128
residual_gate_init: 0.0
short_gap_threshold: 0.3
long_gap_threshold: 1.0
overlap_threshold: 0.05
```

Metrics:
Report the same metrics as Experiments 1 and 2:

* WA
* UA
* WF1
* Macro-F1
* per-class F1
* confusion matrix

Save outputs:

* `results/wavlm_cim/metrics.json`
* `results/wavlm_cim/predictions.csv`
* `results/wavlm_cim/config.json`
* `results/wavlm_cim/temporal_feature_stats.json`
* best checkpoint

`predictions.csv` should contain:

* dialogue_id
* utterance_id
* speaker_id
* start_time
* end_time
* duration
* gap_prev
* overlap_prev
* overlap_ratio
* is_interrupting_prev
* speaker_switch
* gold_label
* pred_label
* probability for each class

Additional required analysis:
Create a subset evaluation script:
`evaluate_temporal_subsets.py`

It should report metrics for:

1. no-overlap utterances
2. overlap utterances
3. strong-overlap utterances
4. short-response utterances
5. long-pause utterances
6. speaker-switch utterances

Save:

* `results/wavlm_cim/subset_metrics.json`

Acceptance criteria:
The code is correct if:

* CIM uses real temporal interaction features.
* CIM keeps the same CDM-style residual memory interface as Experiment 2.
* No future information leaks into temporal features.
* The model can be compared directly with:

  * WavLM baseline
  * WavLM + CDM without CIM
* The output includes overall metrics and temporal subset metrics.
