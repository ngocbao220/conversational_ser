You are a Codex agent working on a PyTorch research codebase for conversational Speech Emotion Recognition.

Create a Jupyter notebook:

notebooks/feature_engineer.ipynb

This notebook is analysis-only. Do not train the main SER models.

Goal:
Build a data-driven temporal feature qualification pipeline for IEMOCAP and MELD.

We want to decide which temporal interaction features should be included in future TIM / Dual-Branch TIM models based on:

1. Literature-motivated interaction phenomena
2. Dataset feasibility
3. Feature distribution
4. Statistical association with emotion labels
5. Complementarity beyond WavLM embeddings
6. Relationship with model error patterns

==================================================
Datasets
==================================================

Analyze both datasets if available:

1. IEMOCAP
2. MELD

The notebook should automatically detect available local dataset files / metadata caches.

For each utterance, try to load:

- dataset_name
- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time
- label / emotion
- audio_path if available
- transcript_text if available

Transcript text is only for analysis display, not model training.

If MELD does not contain precise start_time/end_time but has utterance order only, mark unsupported timestamp-based features clearly.

==================================================
Notebook structure
==================================================

Create these sections:

1. Setup
2. Load metadata
3. Dataset feasibility table
4. Temporal feature construction
5. Distribution analysis
6. Emotion association analysis
7. Redundancy analysis
8. Complementarity with WavLM embeddings
9. Relationship with model errors
10. Feature qualification table
11. Final recommendation for TIM v2
12. Save all outputs

==================================================
1. Setup
==================================================

Import:

- pandas
- numpy
- matplotlib
- pathlib
- json
- warnings
- sklearn.metrics
- sklearn.feature_selection
- sklearn.preprocessing
- sklearn.linear_model
- sklearn.ensemble
- sklearn.model_selection

Optional:
- scipy.stats
- seaborn if already installed
- umap if installed

Set random seed = 42.

Create output directory:

results/feature_engineering/

==================================================
2. Load metadata
==================================================

Implement robust loaders.

Try to load IEMOCAP metadata from existing project utilities if available.
Otherwise, parse from existing cached files or prediction files.

Try paths:
- iemocap/
- data/iemocap/
- results/wavlm_shared/
- results/*/predictions.csv

Try to load MELD metadata from:
- meld/
- data/meld/
- MELD/
- results/meld/
- results/*meld*/predictions.csv

The loader should return one standardized dataframe:

df_all

Columns:

- dataset
- dialogue_id
- utterance_id
- speaker_id
- turn_index
- start_time
- end_time
- duration
- label
- audio_path
- transcript_text

For each dataset:
- sort by dialogue_id, start_time if available, otherwise turn_index
- ensure turn_index exists
- preserve utterance order

Save:

results/feature_engineering/metadata_all.csv

==================================================
3. Dataset feasibility table
==================================================

Create a feature feasibility matrix.

For each proposed interaction phenomenon, check whether required fields exist.

Phenomena/features:

A. Response Dynamics:
- response_latency
- relative_response_latency
- latency_trend
- latency_variance
- immediate_response
- short_response
- long_pause

B. Turn-taking:
- speaker_switch
- consecutive_turn_count
- turn_holding_duration
- turn_yielding
- speaker_switch_frequency_window

C. Overlap / Interruption:
- overlap_duration
- overlap_ratio
- overlap_flag
- strong_overlap
- interruption_flag
- consecutive_overlap_count
- overlap_frequency_window
- competitive_overlap_proxy
- cooperative_overlap_proxy

D. Dialogue Rhythm:
- interaction_density
- silence_density
- burstiness
- rhythm_variance

E. Speaker Behavior:
- speaker_dominance_time
- speaker_dominance_turns
- speaker_interruption_tendency
- speaker_yield_tendency
- speaker_response_habit
- speaker_persistence

F. Dialogue State:
- rapid_exchange_state
- conflict_like_state
- hesitation_state
- calm_state
- floor_competition_state

For every feature, report:

- required columns
- computable_on_IEMOCAP: yes/no/partial
- computable_on_MELD: yes/no/partial
- reason if not computable
- whether feature is causal
- whether feature uses future information
- whether suitable for online inference

Save:

results/feature_engineering/feature_feasibility.csv

==================================================
4. Temporal feature construction
==================================================

Implement causal feature construction.

Create:

build_temporal_features(df_dataset, config)

Use only current and previous utterances in the same dialogue.

Default thresholds:
- overlap_threshold = 0.05
- strong_overlap_ratio_threshold = 0.30
- immediate_gap_threshold = 0.10
- short_gap_threshold = 0.30
- long_gap_threshold = 1.00
- rapid_exchange_window = 3
- rhythm_window = 5
- density_window_seconds = 10.0
- eps = 1e-8

Compute utterance-level features:

Basic:
- duration
- gap_prev
- overlap_duration
- overlap_ratio
- abs_gap
- log_turn_index

Response:
- immediate_response
- short_response
- long_pause
- relative_gap_to_speaker_mean
- previous_mean_gap
- window3_average_gap
- window5_average_gap
- window3_gap_variance
- window5_gap_variance

Turn-taking:
- speaker_switch
- same_speaker_continuation
- consecutive_same_speaker_turns
- turn_holding_duration_so_far
- speaker_switch_frequency_window3
- speaker_switch_frequency_window5

Overlap:
- overlap_flag
- strong_overlap
- interruption_flag
- consecutive_overlap_count
- overlap_frequency_window3
- overlap_frequency_window5
- interruption_frequency_window3
- interruption_frequency_window5
- competitive_overlap_proxy
- cooperative_overlap_proxy

Dialogue rhythm:
- interaction_density_10s
- silence_density_10s
- burstiness_window5
- rhythm_variance_window5

Speaker behavior:
- speaker_prev_turn_count
- speaker_prev_total_speaking_time
- speaker_dominance_time_so_far
- speaker_dominance_turns_so_far
- speaker_prev_overlap_rate
- speaker_prev_interruption_rate
- speaker_prev_mean_gap
- speaker_prev_mean_duration
- speaker_persistence_so_far

Dialogue states:
- rapid_exchange_state
- conflict_like_state
- hesitation_state
- calm_state
- floor_competition_state

Definitions:

competitive_overlap_proxy:
speaker_switch == 1 AND overlap_duration > overlap_threshold AND overlap_ratio >= 0.30

cooperative_overlap_proxy:
speaker_switch == 1 AND overlap_duration > overlap_threshold AND overlap_ratio < 0.30

rapid_exchange_state:
window5_average_gap < short_gap_threshold AND speaker_switch_frequency_window5 high

conflict_like_state:
overlap_frequency_window5 high OR interruption_frequency_window5 high

hesitation_state:
long_pause == 1 OR window5_average_gap > long_gap_threshold

calm_state:
low overlap AND normal gap AND low rhythm variance

floor_competition_state:
consecutive overlaps or repeated speaker_switch with negative gaps

Important:
If timestamp fields are unavailable or unreliable, set timestamp-dependent features to NaN and mark unsupported.

Save:

results/feature_engineering/features_all.csv

==================================================
5. Distribution analysis
==================================================

For every dataset and every feature:

Compute:
- count
- missing_rate
- mean
- std
- median
- min
- max
- q05
- q25
- q75
- q95
- unique_count
- zero_rate
- binary_rate if binary

Flag:
- near_constant if zero_rate > 0.95 or unique_count <= 2 with imbalance > 0.95
- sparse_event if event rate < 0.05
- high_missing if missing_rate > 0.30

Save:

results/feature_engineering/feature_distribution_stats.csv

Plots:
- histogram for continuous features
- bar chart for binary/event features
- save under:
  results/feature_engineering/plots/distributions/

==================================================
6. Emotion association analysis
==================================================

For every dataset and every feature:

Analyze relationship with emotion labels.

For continuous features:
- Kruskal-Wallis test across emotion classes
- ANOVA if scipy available
- eta squared or epsilon squared effect size
- Spearman correlation with label_id as rough diagnostic only
- mutual information with label

For binary features:
- chi-square test if scipy available
- Cramer's V
- event rate by emotion
- mutual information with label

For each feature, compute:
- p_value
- effect_size
- mutual_information
- classwise_mean_or_rate
- best_separating_emotion_pair if possible

Save:

results/feature_engineering/emotion_association.csv
results/feature_engineering/emotion_classwise_stats.csv

Plots:
- boxplots by emotion for continuous features
- event rate by emotion for binary features
- save under:
  results/feature_engineering/plots/emotion_association/

==================================================
7. Redundancy analysis
==================================================

For each dataset:

Compute feature-feature redundancy.

Use:
- Spearman correlation matrix
- Pearson correlation matrix for continuous features
- Cramer's V / phi coefficient for binary pairs if feasible
- mutual information between features if feasible

Identify redundant clusters:
- absolute correlation > 0.90
- or MI very high

Save:
- results/feature_engineering/redundancy_matrix_spearman_{dataset}.csv
- results/feature_engineering/redundant_feature_clusters_{dataset}.csv

Create heatmap plots.

==================================================
8. Complementarity with WavLM embeddings
==================================================

Goal:
Determine whether temporal features provide information beyond frozen WavLM embeddings.

Try to load WavLM embeddings from:
- results/wavlm_shared/cache/wavlm_mean_embeddings.pt
- results/*/cache/*.pt

If embeddings are unavailable, skip this section with a clear warning.

For each dataset:

Create three simple probes using train/val split or cross-validation:

A. WavLM only:
- input = WavLM embedding
- classifier = LogisticRegression or LinearSVM
- metrics = WA, UA, Macro-F1, WF1

B. Temporal only:
- input = selected temporal features
- classifier = LogisticRegression or RandomForest
- metrics same

C. WavLM + Temporal:
- input = concat(WavLM embedding, temporal features)
- classifier = LogisticRegression
- metrics same

Also compute:
- delta_macro_f1 = MacroF1(WavLM+Temporal) - MacroF1(WavLM)
- delta_UA
- which features improve most using permutation importance if feasible

Important:
This is a lightweight probe, not main model training.

Save:
- results/feature_engineering/complementarity_probe_metrics.csv
- results/feature_engineering/temporal_probe_feature_importance.csv

==================================================
9. Relationship with model errors
==================================================

If prediction files are available, analyze whether temporal features explain improvements/errors.

Try to load:
- results/wavlm_mal_no_tim/predictions.csv
- results/wavlm_tim/predictions.csv
- results/dual_branch/predictions.csv

Merge by:
- dialogue_id
- utterance_id

Create:
- mal_correct
- tim_correct
- dual_correct
- dual_improves_over_mal
- dual_hurts_vs_mal

For each feature:
- compare distribution in:
  - MAL wrong / Dual correct
  - MAL correct / Dual wrong
  - both correct
  - both wrong
- compute association with improvement
- compute association with hurt
- compute effect size

Save:
- results/feature_engineering/error_feature_analysis.csv
- results/feature_engineering/error_feature_classwise.csv

This section is important because a feature may not correlate strongly with emotion overall, but may explain where MAL fails.

==================================================
10. Feature qualification table
==================================================

Create final feature qualification table.

For each feature, include:

- feature_name
- phenomenon_group
- description
- literature_motivation_short
- computable_iemocap
- computable_meld
- causal
- online_suitable
- distribution_quality
- missing_rate
- event_rate_or_variance
- emotion_association_score
- effect_size
- mutual_information
- redundancy_score
- complementarity_score
- error_explanation_score
- final_recommendation:
  - Highly Recommended
  - Useful
  - Weak
  - Redundant
  - Noisy
  - Not computable
- recommendation_reason

Scoring suggestion:

distribution_quality:
- low if near_constant or high_missing
- medium if sparse but meaningful
- high otherwise

emotion_association_score:
- based on effect_size + mutual_information

redundancy_score:
- high redundancy if max abs corr > 0.90

complementarity_score:
- based on temporal probe and WavLM+Temporal gain

error_explanation_score:
- based on association with dual_improves_over_mal

Final recommendation rules:
- Highly Recommended:
  computable + causal + good distribution + nontrivial emotion association or error explanation + not extremely redundant
- Useful:
  computable + causal + moderate evidence
- Weak:
  computable but weak association
- Redundant:
  useful but highly redundant with stronger feature
- Noisy:
  unstable, sparse, or associated with errors/hurts
- Not computable:
  required metadata unavailable

Save:
results/feature_engineering/feature_qualification_table.csv
results/feature_engineering/recommended_features.csv

==================================================
11. Final recommendation for TIM v2
==================================================

Generate a Markdown report:

results/feature_engineering/feature_engineering_report.md

Report must include:

1. Executive summary
2. Dataset feasibility comparison: IEMOCAP vs MELD
3. Which features are actually measurable
4. Which interaction phenomena appear frequently enough
5. Which features are statistically associated with emotion
6. Which features are complementary to WavLM
7. Which features explain MAL/Dual errors
8. Recommended feature groups for TIM v2
9. Features to remove or avoid
10. Proposed TIM v2 design based on the evidence

Recommended TIM v2 design should include:
- response dynamics group
- turn-taking group
- overlap/interruption group
- speaker behavior group
- dialogue rhythm/state group
- group-wise encoders
- adaptive gates
- dual-branch fusion

==================================================
12. Save all outputs
==================================================

All outputs must go to:

results/feature_engineering/

Do not modify existing training files.
Do not overwrite existing model results.

==================================================
Acceptance criteria
==================================================

The notebook is correct if:

1. It creates notebooks/feature_engineer.ipynb.
2. It loads or attempts to load IEMOCAP and MELD metadata.
3. It builds a standardized metadata dataframe.
4. It creates a feasibility table.
5. It computes causal temporal/interaction features when possible.
6. It saves feature distributions.
7. It tests association with emotion labels.
8. It analyzes redundancy.
9. It performs complementarity probing with WavLM embeddings if available.
10. It analyzes relation to MAL/TIM/DualBranch errors if predictions exist.
11. It produces a feature qualification table.
12. It produces recommended_features.csv.
13. It writes feature_engineering_report.md.
14. It does not train the main SER models.
15. It does not use transcript text as model input.