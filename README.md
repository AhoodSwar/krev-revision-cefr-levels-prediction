# KUPA-KEYS Revision Analysis: Revision-Process Feature Extraction and Ordinal CEFR Modelling

This repository contains the reproducible analysis script used to
extract 13 revision-process features from KUPA-KEYS Task 2 keystroke
data and evaluate their association with CEFR writing proficiency using
ordinal logistic regression.


## Requirements

The analysis was conducted using Python 3.12.0.

Install the required Python packages using:

    pip install -r requirements.txt

## Repository contents

-   `kupakeys_revision13.py` ---
    End-to-end analysis script.

The script performs Task-2 text reconstruction and validation, obtains
K-REV revision episodes, extracts the 13 writing-level revision
features, constructs the composite CEFR outcome, retains writings that pass
Task-2 reconstruction validation (without Task-1 or cross-task exclusions),
and evaluates the primary B1--C2 ordinal model using 10-fold cross-validation.


## Data availability

The analysis uses `Task2.csv` and `DLLA.csv` from the original
KUPA-KEYS dataset. Instructions for providing the required input files
are given below.

The script uses the publicly available KUPA-KEYS dataset:

`https://huggingface.co/datasets/ALTACambridge/KUPA-KEYS`

Required input files:

-   `Task2.csv`
-   `DLLA.csv`

The data are not available in this repository due to their large size.

## Revision-process features

The final feature set contains 13 writing-level measures:

1.  `revisions_per_100_words`
2.  `total_revision_time_ratio`
3.  `revision_regularity_sec_sd`
4.  `ratio_backspace_seq_gt3_in_revisions`
5.  `n_backspace_sequences_with_gt1000ms_intervals`
6.  `mean_time_to_edit_ms_in_revisions`
7.  `median_backspaces_per_revision`
8.  `median_chars_deleted_per_revision`
9.  `median_chars_inserted_per_revision`
10. `median_time_from_start_process_sec`
11. `median_pause_before_revision_sec`
12. `n_backspacing_sequences_in_revisions`
13. `mean_backspaces_per_sequence`

The measures are derived from different underlying units of revision
activity. The `revisions_per_100_words` predictor replaces the raw
`n_revision_episodes` count in the 13-feature model. It is calculated as
the number of K-REV revision episodes divided by the final essay word
count from `DLLA.csv` (`answer` column), multiplied by 100. The raw
`n_revision_episodes` count remains available in the extracted outputs.
Writings with missing or zero final essay word counts are flagged and
prevent model fitting.

 K-REV is used to identify revision episodes. Some measures
summarise revision episodes, whereas the Backspace-sequence measures
summarise physical consecutive Backspace sequences associated with
revision activity.

### Backspace sequences with \>1,000-ms intervals

`n_backspace_sequences_with_gt1000ms_intervals` is a study-specific
temporal extension of Backspace-sequence analysis. It counts physical
consecutive Backspace sequences occurring within K-REV-defined revision
activity for which at least one Backspace press has an associated
preceding key-down (press-to-press) interval greater than 1,000 ms.

The feature is descriptive rather than proficiency-specific: it can be
related to proficiency in the present analysis, but it could also be
used to examine temporal disruption in revision under different tasks or
experimental conditions.

### Unique physical Backspace sequences

Physical Backspace sequences are uniquely identified within each writing
and counted once, even when the same sequence is associated with more
than one K-REV revision episode. This prevents the same physical
sequence from being counted multiple times because of overlapping or
nested episode associations.

## Model

The final model uses:

-   `mord.LogisticIT(alpha=1.0)`
-   median imputation
-   standardization
-   10-fold `StratifiedKFold`
-   shuffling with random state 42

Imputation and standardisation are performed inside the cross-validation
pipeline.

The primary evaluation is based on out-of-fold cross-validation
predictions rather than a separate holdout result.

Reported evaluation measures include:

-   accuracy
-   balanced accuracy
-   mean absolute error (MAE)
-   adjacent accuracy
-   pooled out-of-fold confusion matrix
-   McKelvey--Zavoina pseudo-R² for the full fitted model

## Requirements

The script requires Python and the following packages:

-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `mord`
-   `krev`
-   `matplotlib`

Install the required packages in an appropriate Python environment
before running the analysis.

## Running the analysis

From the repository directory, run:

``` bash
python K-REV_KUPA_KEYS_CLEAN.py \
  --task2 /path/to/Task2.csv \
  --dlla /path/to/DLLA.csv \
  --output-dir KUPAKEYS_Revision13_updated
```

If `--output-dir` is omitted, the script uses
`KUPAKEYS_Revision13_updated`.

## Main outputs

The output directory includes intermediate validation and
feature-extraction files as well as the final model results. Key outputs
include:

-   `00_Task2_reconstruction_validation.csv`
-   `02_KREV_revision_episodes.csv`
-   `05a_Revision13_all_validated_writings.csv`
-   `05_Revision13_writing_level_features.csv`
-   `07_model_ready_common_population.csv`
-   `Primary_4_levels/primary_CV_fold_results.csv`
-   `Primary_4_levels/primary_CV_summary.csv`
-   `Primary_4_levels/primary_OOF_predictions.csv`
-   `Primary_4_levels/primary_confusion_matrix.csv`
-   `Primary_4_levels/primary_standardized_coefficients.csv`
-   `Primary_4_levels/primary_pseudo_R2.csv` (full-fit McKelvey--Zavoina pseudo-R²)

The script also generates PNG figures in `Primary_4_levels/`, including
the confusion matrix, feature-coefficient plots, and feature boxplots.

## Reproducibility notes

The analysis begins with raw Task-2 event data and reconstructs the
writing process before revision-feature extraction. Writings that do not
pass the required Task-2 reconstruction validation are excluded. No Task-1 validation or cross-task exclusions are applied in the
current script. The final ordinal model is restricted to the B1--C2
proficiency range.

Missing predictor values are handled through median imputation within
each cross-validation training fold rather than through casewise
deletion.

## Anonymous-review note

This repository is provided for anonymous peer review and therefore does
not include author-identifying information. Author names, affiliations,
and permanent repository metadata can be added after the review process.
