# KUPA-KEYS: Revision Features and CEFR Prediction

Reproducible analysis of KUPA-KEYS Task 2 keystroke data using K-REV revision episodes and ordinal logistic regression. This repository runs **only the primary four-level analysis (B1, B2, C1, C2)**. 

## Data

Obtain `Task2.csv` and `DLLA.csv` from the original KUPA-KEYS dataset (ALTACambridge/KUPA-KEYS on Hugging Face). The original data are **not redistributed** in this repository. Follow the source dataset's terms of use.

## Setup

Python 3.12 is recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python kupakeys_revision13.py \
  --task2 /path/to/Task2.csv \
  --dlla /path/to/DLLA.csv \
  --output-dir results
```

The primary model results are written to `results/Primary_4_levels/`, alongside intermediate extraction and quality-control outputs in `results/`.

## Analysis

The script reconstructs Task 2 writing, validates the reconstructed text, extracts K-REV revision episodes, calculates 13 writing-level features, and predicts CEFR levels B1–C2. 

The ordinal model uses `mord.LogisticIT(alpha=1.0)`, median imputation, standardisation, and shuffled stratified 10-fold cross-validation (`random_state=42`). Imputation and standardization are fit inside each training fold. Evaluation includes accuracy, balanced accuracy, mean absolute error, adjacent accuracy, out-of-fold predictions/confusion matrix, and full-fit McKelvey–Zavoina pseudo-R². The script also generates plots.

