You are a Codex agent working on a PyTorch research codebase for conversational Speech Emotion Recognition.

Implement a new experiment with:

run_name: dual_branch
experiment_name: wavlm_dual_branch_tim
output_dir: results/dual_branch

Goal:
Implement a Dual-Branch Temporal Dialogue Memory model.

Motivation:
The current TIM injects temporal features directly into the same GRU memory input as audio embeddings. This may make temporal features noisy and can hurt MAL. The new design must decouple:
1. acoustic dialogue memory branch
2. temporal interaction branch

Then fuse both branches with gated residual addition.

==================================================
Existing experiments
==================================================

Use the existing codebase and reuse as much as possible from:

- wavlm_baseline_no_mal_no_tim
- wavlm_mal_no_tim
- wavlm_tim

Do not break existing experiments.

==================================================
Data
==================================================

Reuse existing IEMOCAP dataset parser, LOSO split, and precomputed/frozen WavLM embeddings.

Each utterance must provide:

- wavlm_embedding h_i
- label
- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time

Do not use transcript text.

Validation/test must remain dialogue-sequential:
- group by dialogue_id
- sort utterances by start_time, end_time, utterance_id
- reset memory at dialogue boundary
- process utterances sequentially
- do not shuffle utterances inside dialogue
- do not update model weights during validation/test

==================================================
Model
==================================================

Create:

models/wavlm_dual_branch_tim.py

Implement:

1. TemporalFeatureBuilder or reuse existing temporal feature builder.
2. TemporalInteractionEncoder.
3. DialogueMemoryBranch.
4. TemporalMemoryBranch.
5. WavLMDualBranchTIMSerModel.

Input per utterance:

- h_i: WavLM utterance embedding
- tau_i: causal temporal/interaction features

Do not feed future information.

==================================================
Architecture
==================================================

For utterance i:

1. Acoustic embedding:

h_i = WavLM embedding or precomputed WavLM embedding.

2. Dialogue memory branch:

This branch is MAL-style and only receives h_i.

Read-before-write:

d_i = DialogueRead(h_i, S_d_{i-1})

S_d_i = DialogueUpdate(S_d_{i-1}, h_i)

Suggested implementation:

z_d = W_d_in(h_i)
d_i = W_d_out(readout_d(concat(z_d, S_d_{i-1})))
S_d_i = GRUCell_d(z_d, S_d_{i-1})

3. Temporal interaction branch:

This branch only receives temporal/interaction features.

tau_i = temporal features from start_time/end_time/speaker history

t_i = TemporalInteractionEncoder(tau_i)

q_i = TemporalRead(t_i, S_t_{i-1})

S_t_i = TemporalUpdate(S_t_{i-1}, t_i)

Suggested implementation:

t_i = TemporalInteractionEncoder(tau_i)
q_i = W_t_out(readout_t(concat(t_i, S_t_{i-1})))
S_t_i = GRUCell_t(t_i, S_t_{i-1})

4. Project both branches to the same dimension as h_i:

d_proj = ProjectDialogue(d_i)
q_proj = ProjectTemporal(q_i)

Both must have shape [hidden_dim_of_wavlm_embedding].

5. Gated residual fusion:

h_tilde_i =
    h_i
    + tanh(alpha) * d_proj
    + tanh(beta) * q_proj

Initialize:
- alpha = 0.0
- beta = 0.0

Important:
beta must start at 0 so temporal branch cannot damage the already useful acoustic/dialogue branch at the beginning of training.

6. Classifier:

logits_i = classifier(h_tilde_i)

7. Update states after prediction:

After logits_i is produced:
- update S_d_i using h_i branch
- update S_t_i using t_i branch

Prediction for utterance i must use:
- h_i
- tau_i
- S_d_{i-1}
- S_t_{i-1}

It must not use:
- future utterances
- future labels
- future temporal features

==================================================
Temporal features
==================================================

Use the same causal temporal features as existing TIM, but allow grouping later.

Default temporal features:

1. duration
2. gap_prev
3. overlap_prev
4. overlap_ratio
5. is_overlap
6. is_interrupting_prev
7. speaker_switch
8. same_speaker
9. turn_index_norm
10. abs_gap
11. short_response
12. long_pause
13. speaker_prev_overlap_rate
14. speaker_prev_mean_gap
15. speaker_prev_mean_duration
16. speaker_prev_turn_count_norm

Continuous features:
- normalize using train split statistics only
- save stats to results/dual_branch/temporal_feature_stats.json
- apply same stats to val/test

Binary flags:
- do not normalize

Clip continuous values:
- duration: [0.05, 20.0]
- gap_prev: [-5.0, 5.0]
- overlap_prev: [0.0, 10.0]
- overlap_ratio: [0.0, 1.0]
- abs_gap: [0.0, 5.0]
- relative/speaker stats: reasonable finite clipping

==================================================
Training strategy
==================================================

Implement script:

scripts/train_dual_branch.py

Config:

configs/dual_branch.yaml

Default config:

run_name: dual_branch
experiment_name: wavlm_dual_branch_tim
output_dir: results/dual_branch

use_precomputed_wavlm_embeddings: true
freeze_wavlm: true
test_session: 5
validation_split: 0.10
split_level: dialogue

selection_metric: UA
max_epochs: 10
batch_mode: dialogue
learning_rate: 1e-4
weight_decay: 1e-4
gradient_clip: 1.0
dropout: 0.2
memory_dim: 128
temporal_emb_dim: 64
seed: 42
use_wandb: true

==================================================
Optional staged training
==================================================

Add config options for staged training, but default may be end-to-end:

training_stage:
  mode: end_to_end

Also support:

training_stage:
  mode: staged
  stage_1_train_dialogue_branch: true
  stage_2_freeze_dialogue_train_temporal: true
  stage_3_finetune_fusion: true

If staged is implemented:

Stage 1:
- train dialogue branch + classifier
- temporal beta fixed at 0

Stage 2:
- freeze dialogue branch
- train temporal branch + beta + classifier or temporal projection
- alpha can remain fixed or trainable

Stage 3:
- unfreeze fusion gates alpha/beta and classifier
- fine-tune lightly

If staged is too much, implement the config hooks and run end_to_end by default.

==================================================
Baselines and comparisons
==================================================

The new model must be directly comparable with:

- results/wavlm_baseline_no_mal_no_tim
- results/wavlm_mal_no_tim
- results/wavlm_tim

Add evaluation script or extend existing one to compare:

- Baseline
- MAL
- TIM
- dual_branch

==================================================
Ablations
==================================================

Implement or prepare config variants:

1. dual_branch_full
2. dual_branch_no_temporal
   - beta fixed to 0
   - should behave like dialogue branch only
3. dual_branch_temporal_only
   - alpha fixed to 0
   - use h_i + temporal branch only
4. dual_branch_zero_temporal
   - tau_i = zero vector
5. dual_branch_shuffled_temporal
   - shuffle temporal features within split/dialogues for sanity check
6. dual_branch_no_overlap
7. dual_branch_no_gap
8. dual_branch_no_duration
9. dual_branch_no_speaker_switch
10. dual_branch_no_turn_position

Save ablation results to:

results/dual_branch/ablation_metrics.csv
results/dual_branch/ablation_metrics.json

==================================================
Outputs
==================================================

Save all outputs under:

results/dual_branch/

Required files:

- metrics.json
- predictions.csv
- config.json
- temporal_feature_stats.json
- confusion_matrix.csv
- confusion_matrix.png
- best.pth
- last.pth
- ablation_metrics.csv
- temporal_subset_metrics.json
- branch_gate_stats.json

predictions.csv must contain:

- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time
- gold_label
- pred_label
- probability columns for every class
- temporal feature columns:
  - duration
  - gap_prev
  - overlap_prev
  - overlap_ratio
  - is_overlap
  - is_interrupting_prev
  - speaker_switch
  - short_response
  - long_pause
- fusion values:
  - alpha_value
  - beta_value
  - dialogue_residual_norm
  - temporal_residual_norm

==================================================
Metrics
==================================================

Use same definitions as previous experiments:

- WA = overall accuracy
- UA = unweighted accuracy / macro recall / balanced accuracy
- WF1 = weighted F1
- Macro-F1 = macro F1
- per-class precision/recall/F1
- confusion matrix

Select best.pth by validation UA only.
Never use test metrics for checkpoint selection.

==================================================
Temporal subset analysis
==================================================

Evaluate dual_branch on:

- all
- no_overlap
- any_overlap
- strong_overlap
- interrupting_prev
- short_response
- long_pause
- high_temporal_interaction
- low_temporal_interaction

Save:

results/dual_branch/temporal_subset_metrics.json

Also compare against MAL and old TIM if their predictions exist.

==================================================
Branch diagnostics
==================================================

Save:

results/dual_branch/branch_gate_stats.json

Include:

- alpha value after training
- beta value after training
- mean dialogue residual norm
- mean temporal residual norm
- residual norm by emotion class
- residual norm by subset
- whether beta remains near zero
- whether temporal branch contributes more on high-temporal-interaction subsets

This is important for interpretation.

==================================================
Acceptance criteria
==================================================

The implementation is correct if:

1. Existing experiments still run.
2. dual_branch has separate dialogue and temporal branches.
3. Dialogue branch receives only h_i.
4. Temporal branch receives only temporal features tau_i.
5. Fusion occurs after both branches are projected to the same size.
6. Fusion uses gated residual addition:
   h_i + tanh(alpha) * dialogue + tanh(beta) * temporal
7. beta initializes to 0.
8. Validation/test are dialogue-sequential.
9. Memory resets at dialogue boundaries.
10. No future utterance information is used.
11. Continuous temporal features are normalized with train split only.
12. Binary temporal flags are not normalized.
13. Outputs and diagnostics are saved under results/dual_branch.
14. The model can be compared directly with MAL and old TIM.

==================================================
Important note
==================================================

Do not implement this as:
    concat(h_i, temporal_features) -> one GRUCell

That is the old TIM.

The new model must have two separate branches:
    dialogue memory branch
    temporal interaction branch

Then fuse them with gated residual addition.