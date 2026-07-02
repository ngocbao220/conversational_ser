You are a senior researcher in conversational speech emotion recognition (SER), conversation analysis, and statistical data analysis.

Your task is NOT to train a model.

Instead, perform a comprehensive statistical study on temporal interaction cues using BOTH datasets:

- IEMOCAP
- MELD

The purpose is to verify whether temporal interaction behaviors are actually correlated with emotion labels and whether they provide useful complementary information beyond acoustic features.

==================================================
Goal
==================================================

Investigate whether conversational temporal interaction cues carry emotion-related information.

Focus on interaction behaviors rather than speech content.

The final report should answer:

1. Which temporal interaction cues are significantly correlated with emotion?
2. Which cues are dataset-specific?
3. Which cues are consistent across IEMOCAP and MELD?
4. Which cues are likely useful for conversational SER?
5. Which cues are noisy or redundant?

==================================================
Temporal Interaction Cues
==================================================

For every utterance, compute the following features whenever possible:

Basic:

- duration
- gap_to_previous
- gap_to_next
- overlap_duration
- overlap_ratio
- overlap_flag
- interruption_flag
- speaker_switch
- same_speaker_continuation
- turn_index
- normalized_turn_position

Response dynamics:

- immediate_response
- short_response
- long_pause

Speaker statistics (causal):

- speaker_previous_mean_gap
- speaker_previous_mean_duration
- speaker_previous_overlap_rate
- speaker_previous_turn_count

Dialogue statistics (causal):

- previous_overlap_rate
- previous_speaker_switch_rate
- previous_mean_gap
- previous_mean_duration

Window statistics:

For previous N={3,5} utterances compute:

- overlap frequency
- interruption frequency
- average gap
- average duration
- speaker switching frequency

==================================================
Dataset Analysis
==================================================

For BOTH datasets separately:

Produce:

1. Feature distributions

Histogram

Boxplot

Violin plot

Mean

Std

Median

95% interval

2. Per-emotion statistics

For each emotion:

mean ± std

distribution

sample count

3. Correlation analysis

Pearson

Spearman

Mutual Information

Distance Correlation (if available)

between every temporal feature and emotion labels.

4. Statistical significance

ANOVA

Kruskal-Wallis

Mann-Whitney

Effect size

Report p-values and effect sizes.

5. Redundancy analysis

Correlation matrix

Variance Inflation Factor

Feature clustering

Detect highly redundant temporal features.

==================================================
Interaction Pattern Analysis
==================================================

Investigate whether some interaction patterns occur more frequently under specific emotions.

Examples:

Does angry contain:

- shorter response latency?
- higher interruption frequency?
- more overlap?

Does sad contain:

- longer pauses?
- longer duration?

Does happy contain:

- more positive overlap?
- faster turn taking?

Quantify these observations.

==================================================
Speaker Behavior Analysis
==================================================

For each speaker:

Analyze:

average response latency

average overlap

average interruption

average duration

Determine whether relative features are more informative than absolute features.

Example:

absolute_duration

vs

duration - speaker_mean_duration

==================================================
Dialogue Progression Analysis
==================================================

Analyze how temporal interaction changes during dialogue.

Split dialogue into:

Beginning

Middle

End

Investigate:

emotion distribution

overlap

pause

speaker switching

==================================================
Subset Analysis
==================================================

Create subsets:

High-overlap dialogues

Low-overlap dialogues

High interruption

Low interruption

Fast response

Slow response

Many speaker switches

Few speaker switches

Report emotion distributions for each subset.

==================================================
Cross-Dataset Comparison
==================================================

Compare IEMOCAP vs MELD.

Answer:

Which temporal features behave similarly?

Which are dataset-specific?

Which features are robust enough to transfer across datasets?

==================================================
Feature Recommendation
==================================================

Based on all statistical evidence:

Rank temporal interaction features into:

Highly Recommended

Useful

Weak

Redundant

Noisy

For every recommendation, provide justification using statistical evidence.

==================================================
Design Recommendation for CIM
==================================================

Finally propose an improved CIM architecture.

Do NOT simply concatenate all temporal features.

Recommend:

feature grouping

feature gating

feature attention

feature selection

adaptive temporal weighting

Explain why these design choices are supported by the statistical analysis.

==================================================
Output
==================================================

Produce:

analysis_report.md

feature_statistics.csv

emotion_statistics.csv

feature_importance.csv

correlation_matrix.csv

redundancy_matrix.csv

recommended_temporal_features.csv

plots/

histograms

boxplots

violin plots

correlation heatmaps

subset comparison figures

cross_dataset_comparison.md

cim_design_recommendation.md

The report should be written as if preparing the "Motivation" and "Method" sections of a top-tier speech emotion recognition paper.

==================================================
Complementarity Analysis
==================================================

Using frozen WavLM embeddings:

1. Train a simple classifier using:
   - WavLM embedding only
   - Temporal features only
   - WavLM + Temporal features

Compare Macro-F1.

2. Estimate complementary information by:

- Mutual Information
- Canonical Correlation Analysis (CCA)
- HSIC (if available)

Determine whether temporal interaction cues provide complementary information beyond acoustic representations.

If a temporal feature can already be linearly predicted from WavLM embeddings, mark it as redundant.

Otherwise, mark it as complementary.

This section is critical for motivating CIM.