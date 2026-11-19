# BT4012 Fraud Analytics Group Project - Group 10

Detecting bank account opening fraud on the [Bank Account Fraud (BAF) dataset suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (NeurIPS 2022), with a human review stage tested on the [FiFAR](https://springernature.figshare.com/articles/dataset/Financial_Fraud_Alert_Review_Dataset/28351172) dataset. The full write-up is in the report; this README covers what is in the repo and how to run it.

## Notebooks

Each notebook builds on the ones before it, and notebook 07 brings them together.

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Explores the data and lists what the modelling has to deal with |
| `02_feature_engineering_and_models.ipynb` | Builds the features, compares and tunes three models, and freezes the final XGBoost model |
| `03_human_ai_review.ipynb` | Tests which fraud alerts should go to a human analyst (FiFAR) |
| `04_evaluation.ipynb` | Confidence intervals, month-by-month results, calibration, and threshold rules |
| `05_interpretability.ipynb` | Explains what the model relies on (SHAP, permutation importance, reason codes) |
| `06_fairness_accountability.ipynb` | Measures the age gap, tests ways to reduce it, and sets up accountability |
| `07_final_system.ipynb` | Combines the recommendations into one system and reports its final results |

## Main result

The final system (notebook 07) is the tuned XGBoost model retrained with reweighing, a fixed alert rate instead of a fixed threshold, and a review stage that sends the alerts the model is least sure about to a fairer pool of analysts. On the Base test months (6-7):

| | Frozen model (notebook 02) | Final system (notebook 07) |
|---|---|---|
| Recall | 0.56 | 0.52 |
| False positive rate | 5.3% | 4.9% |
| False positive rate, age 50+ / under 50 | 12.7% / 3.9% | 5.3% / 4.8% |
| Predictive equality (1 = equal) | 0.31 | 0.92 |

The age gap is closed for about 4 points of recall, and the fairness gain holds on all five BAF variants.

## Repository layout

```
data/            the BAF CSVs and FiFAR (not committed, see below)
notebooks/       the seven notebooks, and model_loading.py
configs/         final_model.json (notebook 02) and calibration.json (notebook 04)
outputs/
  models/        the frozen model, one file per seed (notebook 02)
  predictions/   the frozen model's test-month scores (notebook 02)
  interpretability/feature_dictionary.csv (notebook 05)
  final_results.csv  the final system's main results, with 95% intervals (notebook 07)
requirements.txt
```

Notebooks 03-07 load the saved model from `outputs/models/` instead of retraining it, so everyone uses exactly the same model.

## Setup

Python 3.12. From the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Select the `.venv` kernel in VS Code or Jupyter. Please run the notebooks in this environment: XGBoost can train slightly different trees on other Python or library versions.

- macOS: XGBoost needs OpenMP (`brew install libomp`).
- Windows: load the saved models with `load_model` from `notebooks/model_loading.py`, since a plain `joblib.load` fails there.
- Optional: `fairlearn` (a cross-check in notebook 06) and `interpret` (an extra model in notebook 05). Both are skipped if not installed.

## Data

The data is not in the repo (about 1.6 GB in total).

1. **BAF:** download **version 1** from [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/versions/1) and put the six CSVs in `data/` with their original names (`Base.csv`, `Variant I.csv`, ... `Variant V.csv`). Version 2 has the rows in a different order, which breaks the saved predictions. `Base.csv` should have 1,000,000 rows.
2. **FiFAR** (notebooks 03, 06, 07), from the project root:
   ```sh
   curl -L -o data/FiFAR.zip https://ndownloader.figshare.com/files/52147616
   unzip data/FiFAR.zip -d data && mv data/FiFAR data/fifar && rm data/FiFAR.zip
   ```

## Running

Run the notebooks from the `notebooks/` folder, in order. Notebook 02 takes about 25 minutes (most of it is tuning) and writes the model the others use. Notebooks 03-07 take 1-10 minutes each (notebook 06 about 20 minutes the first time) and can be re-run without re-running notebook 02. Notebooks 04, 06 and 07 keep score caches in `data/cache/`, which is not committed and can be deleted at any time.
