You are a Codex agent working on a PyTorch research codebase for conversational Speech Emotion Recognition.

Create a Jupyter notebook for analyzing when and why the dual-branch temporal dialogue memory model helps or hurts.

Notebook name:
notebooks/analyze_dual_branch_errors.ipynb

Do not train any new model in this notebook.
This is an analysis-only notebook.

==================================================
Goal
==================================================

Analyze prediction behavior of:

1. MAL / dialogue-only model
2. old TIM if available
3. dual_branch model

The goal is to answer:

- In which cases does dual_branch improve over MAL?
- In which cases does dual_branch hurt?
- Which emotion classes benefit most?
- Which temporal interaction patterns are associated with improvement?
- Does the temporal branch contribute more on high-interaction utterances?
- Are errors caused by noisy temporal cues such as overlap or duration?
- Can we identify representative case studies with timeline + transcript/audio metadata?

==================================================
Input files
==================================================

The notebook should automatically look for these files:

MAL predictions:
results/wavlm_mal_no_tim/predictions.csv

Old TIM predictions:
results/wavlm_tim/predictions.csv

Dual Branch predictions:
results/dual_branch/predictions.csv

Dual Branch temporal subset:
results/dual_branch/temporal_subset_metrics.json

Dual Branch diagnostics:
results/dual_branch/branch_gate_stats.json

Optional transcript metadata:
If transcript text is available in the dataset parser or metadata cache, load it.
Transcript text is only for analysis and visualization, not for model input.

If a file is missing, the notebook should warn clearly and continue with available files.

==================================================
Notebook sections
==================================================

Create the notebook with the following sections.

--------------------------------------------------
1. Setup
--------------------------------------------------

Import:
- pandas
- numpy
- matplotlib
- sklearn.metrics
- pathlib
- json
- warnings

Optional:
- seaborn only if already available
- scipy if available

Set display options.

Define paths.

--------------------------------------------------
2. Load predictions
--------------------------------------------------

Load available prediction files.

Standardize column names.

Required columns:
- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time
- gold_label
- pred_label

For dual_branch predictions, also load if available:
- duration
- gap_prev
- overlap_prev
- overlap_ratio
- is_overlap
- is_interrupting_prev
- speaker_switch
- short_response
- long_pause
- alpha_value
- beta_value
- dialogue_residual_norm
- temporal_residual_norm

If temporal columns are missing, recompute them causally from:
dialogue_id, speaker_id, start_time, end_time.

--------------------------------------------------
3. Merge model predictions
--------------------------------------------------

Merge MAL, TIM, and dual_branch by:
- dialogue_id
- utterance_id

Verify:
- gold labels match across models
- no duplicate utterance ids after merge
- no missing critical fields

Create columns:
- mal_correct
- tim_correct if TIM exists
- dual_correct
- dual_improves_over_mal:
    MAL wrong, Dual correct
- dual_hurts_vs_mal:
    MAL correct, Dual wrong
- both_correct
- both_wrong

Save merged analysis table:
results/dual_branch/analysis/merged_predictions.csv

--------------------------------------------------
4. Overall metrics
--------------------------------------------------

Compute metrics for each available model:
- WA = accuracy
- UA = balanced accuracy / macro recall
- Macro-F1
- WF1
- per-class precision/recall/F1

Create a summary table:
results/dual_branch/analysis/overall_metrics.csv

Plot bar chart for:
- WA
- UA
- Macro-F1
- WF1

--------------------------------------------------
5. Error taxonomy
--------------------------------------------------

Create error categories comparing MAL vs Dual:

A. MAL wrong, Dual correct
B. MAL correct, Dual wrong
C. both correct
D. both wrong

Report:
- count
- percentage
- emotion distribution
- prediction transition matrix

Save:
results/dual_branch/analysis/error_taxonomy.csv
results/dual_branch/analysis/error_taxonomy_by_emotion.csv

Plots:
- stacked bar by emotion
- confusion transition heatmap:
    gold_label x category
- MAL pred -> Dual pred transitions for cases where they differ

--------------------------------------------------
6. Temporal interaction score
--------------------------------------------------

Create an interpretable interaction score.

Default:

interaction_score =
    1.0 * is_overlap
  + 1.0 * is_interrupting_prev
  + 0.5 * speaker_switch
  + 0.75 * short_response
  + 0.75 * long_pause
  + min(overlap_ratio, 1.0)

If columns missing, use available features.

Create bins:
- low interaction
- medium interaction
- high interaction

Default:
- low: bottom 33%
- medium: middle 33%
- high: top 33%

Also support fixed-rule bins:
- high_temporal_interaction = any of:
  is_overlap, is_interrupting_prev, short_response, long_pause

Analyze:
- MAL vs Dual metrics by interaction bin
- dual improvement rate by interaction bin
- dual hurt rate by interaction bin

Save:
results/dual_branch/analysis/interaction_score_analysis.csv

Plots:
- histogram of interaction_score
- improvement/hurt rate by interaction bin
- Macro-F1 by interaction bin

--------------------------------------------------
7. Temporal feature distribution by outcome
--------------------------------------------------

Compare feature distributions across:

- MAL wrong / Dual correct
- MAL correct / Dual wrong
- both correct
- both wrong

Features:
- duration
- gap_prev
- overlap_prev
- overlap_ratio
- is_overlap
- is_interrupting_prev
- speaker_switch
- short_response
- long_pause
- interaction_score
- dialogue_residual_norm
- temporal_residual_norm
- beta_value if available

For continuous features:
- mean
- std
- median
- 25/75 percentile

For binary features:
- rate per category

Save:
results/dual_branch/analysis/feature_by_error_category.csv

Plots:
- boxplots for continuous features
- bar charts for binary feature rates

--------------------------------------------------
8. Residual contribution analysis
--------------------------------------------------

If available, analyze:

- dialogue_residual_norm
- temporal_residual_norm
- ratio = temporal_residual_norm / (dialogue_residual_norm + eps)
- alpha_value
- beta_value

Questions:
- Does temporal residual become larger in high-interaction cases?
- Is temporal residual larger when Dual improves over MAL?
- Is temporal residual too large when Dual hurts MAL?
- Which emotion classes rely more on temporal branch?

Save:
results/dual_branch/analysis/residual_analysis.csv

Plots:
- temporal_residual_norm by interaction bin
- residual ratio by outcome category
- residual norm by emotion
- scatter: interaction_score vs temporal_residual_norm
- scatter: temporal_residual_norm vs correctness

--------------------------------------------------
9. Emotion-wise gain analysis
--------------------------------------------------

For each emotion class:
- MAL recall
- Dual recall
- gain in recall
- MAL F1
- Dual F1
- gain in F1
- number of MAL wrong / Dual correct
- number of MAL correct / Dual wrong

Save:
results/dual_branch/analysis/emotion_wise_gain.csv

Plot:
- recall gain by emotion
- F1 gain by emotion

--------------------------------------------------
10. Confusion matrix comparison
--------------------------------------------------

Create confusion matrices for:
- MAL
- Dual
- TIM if available

Also create delta confusion:

delta = confusion_dual - confusion_mal

Interpretation:
- Negative off-diagonal values mean Dual reduces that error type.
- Positive off-diagonal values mean Dual increases that error type.

Save:
results/dual_branch/analysis/confusion_mal.csv
results/dual_branch/analysis/confusion_dual.csv
results/dual_branch/analysis/confusion_delta_dual_minus_mal.csv

Plots:
- confusion matrix heatmaps
- delta confusion heatmap

--------------------------------------------------
11. Case study selection
--------------------------------------------------

Automatically select representative examples.

Create four CSVs:

1. dual_improves_cases.csv
Cases where:
- MAL wrong
- Dual correct
- high interaction_score
Sort by:
- high temporal_residual_norm
- high interaction_score

2. dual_hurts_cases.csv
Cases where:
- MAL correct
- Dual wrong
Sort by:
- high temporal_residual_norm
- high interaction_score

3. both_wrong_high_interaction_cases.csv
Cases where:
- MAL wrong
- Dual wrong
- high interaction_score

4. low_interaction_no_gain_cases.csv
Cases where:
- low interaction_score
- MAL and Dual predictions are the same or both correct

Columns should include:
- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time
- gold_label
- mal_pred_label
- dual_pred_label
- tim_pred_label if available
- duration
- gap_prev
- overlap_prev
- overlap_ratio
- is_overlap
- is_interrupting_prev
- speaker_switch
- short_response
- long_pause
- interaction_score
- dialogue_residual_norm
- temporal_residual_norm
- transcript_text if available
- audio_path if available

Save under:
results/dual_branch/analysis/case_studies/

--------------------------------------------------
12. Dialogue timeline visualization
--------------------------------------------------

Create a function:

plot_dialogue_timeline(dialogue_id, center_utterance_id=None, window=5)

It should show:
- utterances as horizontal bars
- x-axis: time in seconds
- y-axis: speaker_id
- highlight center utterance
- color by:
  - gold emotion
  - or correctness category
- annotate:
  - utterance_id
  - gold label
  - MAL pred
  - Dual pred
  - overlap regions if possible

Generate timeline plots for top 10:
- dual_improves_cases
- dual_hurts_cases

Save:
results/dual_branch/analysis/timelines/

--------------------------------------------------
13. Feature importance sanity checks
--------------------------------------------------

If possible, implement simple non-training analyses:

A. Permutation importance on existing predictions:
This may be limited because we are not rerunning the model.
If model cannot be rerun, skip.

B. Statistical association:
For each temporal feature, compute:
- correlation with dual_improves_over_mal
- correlation with dual_hurts_vs_mal
- mutual information if available

Save:
results/dual_branch/analysis/feature_association_with_improvement.csv

--------------------------------------------------
14. Summary report
--------------------------------------------------

Generate a Markdown report:

results/dual_branch/analysis/dual_branch_error_analysis_report.md

The report should include:

1. Overall finding
2. Whether Dual improves over MAL
3. Which emotions benefit most
4. Which temporal interaction subsets benefit most
5. Which cases Dual hurts
6. Whether temporal residual is larger in high-interaction cases
7. Recommended next fixes

Possible recommendations:
- reduce temporal branch when low interaction
- add adaptive beta per utterance
- replace global beta with sample-wise gate
- use feature-group gates
- suppress overlap if it causes false positives
- use transcript only for analysis, not as model input

==================================================
Output directory
==================================================

Create:

results/dual_branch/analysis/

Do not overwrite original prediction files.

==================================================
Acceptance criteria
==================================================

The notebook is correct if:

1. It loads available prediction files.
2. It merges predictions by dialogue_id and utterance_id.
3. It computes error taxonomy comparing MAL and Dual.
4. It computes temporal interaction score.
5. It analyzes feature distributions by error category.
6. It analyzes residual/gate contribution if available.
7. It produces emotion-wise gain analysis.
8. It creates confusion matrix comparison.
9. It exports representative case study CSVs.
10. It produces dialogue timeline visualizations.
11. It saves all outputs under results/dual_branch/analysis/.
12. It generates a final Markdown report.
13. It does not train any model.
14. It does not modify existing checkpoints or predictions.