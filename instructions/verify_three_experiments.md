Verify the full implementation of the three SER experiments.

Important: This is a verification task, not a new implementation task. Do not rewrite the project unless a scdml fix is absolutely necessary. First inspect the codebase, run checks, and produce a clear verification report.

Experiments to verify:

1. Experiment 1:
   - name: `wavlm_baseline_no_cdm_no_cim`
   - model: WavLM utterance-level SER baseline
   - no CDM
   - no CIM
   - no dialogue memory
   - no timestamp feature
   - no speaker feature as model input

2. Experiment 2:
   - name: `wavlm_cdm_no_cim`
   - model: WavLM + CDM
   - uses dialogue memory
   - no real temporal feature
   - temporal_feature_mode should be `zero`
   - start_time/end_time may only be used for dialogue ordering, not as model input
   - no speaker feature as model input
   - memory must reset at dialogue boundary
   - memory must be causal / read-before-write if implemented that way

3. Experiment 3:
   - name: `wavlm_cim`
   - model: WavLM + CIM
   - uses real temporal interaction features
   - must not use future utterances
   - must norcdmize continuous temporal features using train split only
   - must not norcdmize binary flags
   - must save temporal feature statistics
   - must be comparable to Exp 2

Please verify the following items:

A. File structure
Check that the expected files exist or equivalent files exist:

- `configs/wavlm_baseline_no_cdm_no_cim.yaml`
- `configs/wavlm_cdm_no_cim.yaml`
- `configs/wavlm_cim.yaml`
- dataset/parser module for Kaggle IEMOCAP
- WavLM baseline model file
- CDM model file
- CIM model file
- training scripts for Exp 1/2/3
- temporal feature utility for Exp 3
- subset evaluation script if implemented

B. Dataset parser
Verify that the Kaggle IEMOCAP parser:

- reads local `iemocap/`
- parses session_id
- parses dialogue_id
- parses utterance_id
- parses speaker_id
- parses start_time and end_time as float
- resolves correct audio_path
- parses emotion labels from annotation/evaluation files, not transcript text alone
- matches transcript/annotation/audio by utterance_id
- maps labels exactly:
  - `ang -> angry`
  - `hap -> happy`
  - `exc -> happy`
  - `sad -> sad`
  - `neu -> neutral`

- discards labels outside `{ang, hap, exc, sad, neu}`
- final class ids are:
  - angry: 0
  - happy: 1
  - neutral: 2
  - sad: 3

C. LOSO split
Verify that:

- `TEST_SESSION` is configurable
- default test session is 5
- train sessions exclude TEST_SESSION
- test split contains only TEST_SESSION
- validation is a 10% dialogue-level split from train sessions
- no dialogue_id appears in both train and validation
- no dialogue_id appears in both train and test
- no dialogue_id appears in both validation and test

D. Experiment 1 model behavior
Verify that `WavLMSERBaseline`:

- accepts `forward(input_values, attention_mask=None, labels=None, **metadata)`
- does not use dialogue_id, utterance_id, speaker_id, start_time, end_time
- does not use memory state
- does not use temporal features
- supports:
  - `freeze_wavlm: true`
  - `unfreeze_last_n_layers: 0`
  - optional last-n-layer unfreezing

- freezes all WavLM parameters by default
- trains only pooling/classifier parameters by default

E. Experiment 2 model behavior
Verify that WavLM + CDM:

- uses fixed WavLM embeddings or frozen WavLM as configured
- uses dialogue ordering correctly
- may use start_time only for sorting, not as input feature
- does not compute duration/gap/overlap/interruption
- does not feed speaker_id into the model
- uses zero temporal vector if temporal interface exists
- has `use_temporal_features: false`
- has `temporal_feature_mode: zero`
- resets memory state at every dialogue boundary
- does not use future utterances
- uses the same metric/checkpoint logic as Exp 1

F. Experiment 3 model behavior
Verify that WavLM + CIM:

- uses real temporal interaction features
- uses only causal metadata:
  - current utterance metadata
  - previous utterance metadata
  - previous speaker-history statistics

- does not use future utterance audio
- does not use future utterance labels
- does not use future start_time/end_time to compute current features
- computes expected temporal features:
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

- norcdmizes continuous features using train split statistics only
- does not norcdmize binary flags
- saves `temporal_feature_stats.json`
- uses the same checkpoint selection metric as Exp 1/2, preferably validation UA

G. Metrics
Verify metric definitions:

- WA = overall accuracy
- UA = unweighted accuracy / macro recall / balanced accuracy
- WF1 = weighted F1
- Macro-F1 = macro F1

Verify every experiment reports:

- loss
- WA
- UA
- WF1
- Macro-F1
- per-class precision
- per-class recall
- per-class F1
- confusion_matrix.csv
- confusion_matrix.png
- predictions.csv

H. Checkpointing
Verify:

- `best.pth` is selected by validation UA, not test UA
- `last.pth` is saved
- test set is not used for model selection

I. Output schema
For each of:

- `results/wavlm_baseline_no_cdm_no_cim/`
- `results/wavlm_cdm_no_cim/`
- `results/wavlm_cim/`

Verify expected output files if training has been run:

- `metrics.json`
- `predictions.csv`
- `config.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `best.pth`
- `last.pth`

For `results/wavlm_cim/`, also verify:

- `temporal_feature_stats.json`
- `subset_metrics.json` if subset analysis is implemented

Verify `predictions.csv` contains:

- dialogue_id
- utterance_id
- speaker_id
- start_time
- end_time
- gold_label
- pred_label
- probability columns for every class

J. Reproducibility
Verify:

- Python, NumPy, and PyTorch seeds are set
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- trainable and total parameter counts are logged
- wandb is config-controlled and can be disabled
- tqdm progress bars are used or available

K. Dry-run
Run a minicdm dry-run if possible:

- load dataset
- build train/val/test split
- instantiate Exp 1 model
- instantiate Exp 2 model
- instantiate Exp 3 model
- run one forward pass for each model using a tiny batch or one dialogue
- confirm tensor shapes are valid
- confirm loss can be computed
- confirm no NaN/inf in logits/loss
- confirm metadata is preserved but not incorrectly used in Exp 1/2

L. Produce report
Save a verification report to:

`results/verification_report.md`

The report must include:

- PASS/FAIL table for each section A–K
- exact files checked
- exact commands run
- any missing files
- any implementation mismatches
- any risk of leakage
- any metric definition mismatch
- any fairness issue between Exp 2 and Exp 3
- recommended fixes, if needed

Also print a final summary in the terminal:

- `VERIFICATION PASSED` if all critical checks pass
- `VERIFICATION FAILED` if any critical check fails

Critical failures include:

- wrong label mapping
- utterance-level split leakage
- using test metrics for checkpoint selection
- Exp 1 using metadata/ciming/memory
- Exp 2 using real ciming features
- Exp 3 using future utterance information
- wrong WA/UA definitions
- missing predictions schema
