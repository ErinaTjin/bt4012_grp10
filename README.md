# BT4012 Fraud Analytics Group Project - Group 10

Fraud detection on the [Bank Account Fraud (BAF) Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (NeurIPS 2022).

## Repository structure

```
data/                 # the BAF CSVs and FiFAR go here (not committed, see "Getting the data")
notebooks/
  01_eda.ipynb                              # exploratory data analysis
  02_feature_engineering_and_models.ipynb   # features, models, tuning, final model, test results
  03_human_ai_review.ipynb                  # FiFAR: when should an alert go to a human analyst?
  model_loading.py                          # load_model(): loads the saved models on any OS, Windows included
configs/
  final_model.json    # the frozen final model: settings, features, thresholds (written by notebook 02)
outputs/              # the latest final model and its test predictions (written by notebook 02)
  models/             #   the trained model for each seed
  predictions/        #   scores on months 6-7 of Base and all 5 variants
requirements.txt
```

## Prerequisites

- Python 3.12 (3.11 or newer is required by pandas 3)
- **macOS only:** XGBoost needs the OpenMP library. Install it with `brew install libomp`, otherwise loading the saved model fails with "libomp.dylib could not be loaded".
- **Windows only:** `joblib.load` on the saved models fails with "input stream corrupted", because the models were saved on macOS and XGBoost on Windows stores one training-only setting (the random number generator state) differently. Load them with `load_model` from `notebooks/model_loading.py` instead; the scores are the same.

## Setup

From the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code / Jupyter, select the `.venv` kernel before running a notebook. Errors such as "Missing optional dependency 'pyarrow'" or "No type extension with name arrow.py_extension_type" mean another Python is being used: switch the kernel to `.venv`.

## Getting the data

The dataset is about 1.3 GB, so it is not stored in this repository. Only the empty `data/` folder is committed.

1. Download **version 1** of the dataset from Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/versions/1. Kaggle's default download is version 2, which has the same rows sorted by month, so its row numbers do not match `outputs/predictions/`.
   - **Option A (browser):** open the version 1 link, click **Download**, then unzip the archive.
   - **Option B (Kaggle CLI):** with an [API token](https://www.kaggle.com/docs/api) set up, run (the `/1` at the end picks version 1)
     ```sh
     pip install kaggle
     kaggle datasets download sgpjesus/bank-account-fraud-dataset-neurips-2022/1 -p data --unzip
     ```
2. Put the six CSV files in the `data/` folder. **Keep the original file names from Kaggle** (capitals and spaces included), because the notebooks read them by those names.

3. Check the folder looks like this:
   ```
   data/
     .gitkeep
     Base.csv
     Variant I.csv
     Variant II.csv
     Variant III.csv
     Variant IV.csv
     Variant V.csv
   ```

**Sanity check:** `Base.csv` should have 1,000,000 rows, of which 11,029 are fraud (`fraud_bool = 1`). The EDA notebook prints both numbers in section 3.2.

**Keep the rows in Kaggle's original order** (the first rows are month 7). `outputs/predictions/` refers to rows by their row number, so a re-sorted `Base.csv` will not line up with it.

### FiFAR (for notebook 03)

The [Financial Fraud Alert Review (FiFAR)](https://springernature.figshare.com/articles/dataset/Financial_Fraud_Alert_Review_Dataset/28351172) dataset is a single zip of about 209 MB. From the project root:

```sh
curl -L -o data/FiFAR.zip https://ndownloader.figshare.com/files/52147616
unzip data/FiFAR.zip -d data && mv data/FiFAR data/fifar && rm data/FiFAR.zip
```

The folder should then contain `data/fifar/alert_data/`, `data/fifar/synthetic_experts/` and `data/fifar/testbed/`. FiFAR also includes its own copy of `Base.csv` (`data/fifar/alert_data/Base.csv`), identical to Kaggle's.

## Running the notebooks

Run the notebooks in order, from the `notebooks/` folder, since they read the data with relative paths (`../data/...`).

| Notebook | Run time | Writes |
|---|---|---|
| `01_eda.ipynb` | about 1 minute | nothing |
| `02_feature_engineering_and_models.ipynb` | about 25 minutes (most of it is tuning) | `configs/final_model.json`, `outputs/models/`, `outputs/predictions/` |
| `03_human_ai_review.ipynb` | under 1 minute | nothing (uses the saved model in `outputs/`, so notebook 02 does not need re-running) |

## Latest final model

`outputs/` always holds the **latest final model** from notebook 02, together with its predictions on the test months (6-7) of Base and all 5 variants, so you can use them without re-running the notebook. Whenever notebook 02 is changed and re-run, commit the notebook, `configs/final_model.json` and `outputs/` together, so they always match.

**Current model:** XGBoost, tuned, trained on Base months 0-3, with the threshold for each seed picked on month 5. The exact settings and thresholds are in `configs/final_model.json`.

| File | Contents |
|---|---|
| `configs/final_model.json` | Model name, settings, feature options and the threshold for each seed |
| `outputs/models/xgboost_seed0.joblib` ... `seed4.joblib` | The trained model for each seed |
| `outputs/predictions/<dataset>_test_seed0.csv` | Seed-0 scores for months 6-7 of each dataset, with columns `row_id` (row number in the original CSV), `month`, `y_true`, `score`, `customer_age` |

An application is flagged when its `score` is at least the seed's threshold. The scores are **not probabilities**: fraud cases were weighted about 26x during training, so they need calibrating before being read as probabilities.

To score other data with the saved model, run notebook 02 up to section 4.2 (which defines `make_features`, no training needed), then:

```python
import json
from model_loading import load_model   # works on Windows too, see Prerequisites

final_model = json.load(open('../configs/final_model.json'))
model = load_model('../outputs/models/xgboost_seed0.joblib')
threshold = final_model['thresholds']['0']

X = make_features(df, **final_model['features'])
scores = model.predict_proba(X)[:, 1]
flagged = scores >= threshold
```

**Latest test results** (months 6-7, mean over 5 seeds; full discussion in notebook 02, section 8):

| Dataset | Recall at 5% FPR | ROC-AUC | FPR with the frozen threshold | Predictive equality |
|---|---|---|---|---|
| Base | 0.548 | 0.894 | 5.3% | 0.31 |
| Variant I | 0.514 | 0.884 | 4.9% | 0.31 |
| Variant II | 0.551 | 0.897 | 8.4% | 0.31 |
| Variant III | 0.468 | 0.870 | 8.4% | 0.31 |
| Variant IV | 0.468 | 0.870 | 8.4% | 0.31 |
| Variant V | 0.483 | 0.873 | 8.3% | 0.31 |

- The model wrongly flags about 12.7% of genuine applicants aged 50+ vs 3.9% of those under 50 on every dataset (predictive equality 0.31, in line with the BAF paper).
- The frozen threshold flags more than 5% on Variants II-V because half of their applicants are 50+, the group the model flags more often.
- Accuracy is highest where age really predicts fraud (Base, Variant II) and lowest where it doesn't (Variants III-V), which suggests the model relies on `customer_age`.

## How models are evaluated

These rules are set at the start of notebook 02 (section 2), before any model is trained, and are the same for every model and every dataset.

| Months | Used for |
|---|---|
| 0-3 | training the models |
| 4 | comparing models and tuning (validation) |
| 5 | picking the threshold for the final model |
| 6-7 | final results, on Base and on all 5 variants |

- **Train and tune on Base only.** The final model is then scored, unchanged, on months 6-7 of Base and of every variant. Nothing is refit or re-tuned on the variants.
- **Seeds.** Every model is trained with seeds 0-4; results are reported as mean ± standard deviation.
- **Main metric: recall at 5% false positive rate (FPR).** If at most 5% of genuine applications may be flagged, what share of fraud is caught?
- **Threshold.** For the final model, the threshold that flags 5% of genuine applications is picked on month 5 and then applied unchanged to every test set.
- **Fairness.** Age groups are 50+ vs under 50. The main fairness metric is **predictive equality**: the lower group's FPR divided by the higher group's FPR (1 = genuine applicants in both groups are wrongly flagged equally often).
- **`month` is never a model feature.**
- **Once the test results (section 8) have been run, the earlier sections are not changed**, so the test results never steer the model.

Notes on the variants:
- `x1` and `x2` (only in Variants III and V) are dropped, because the Base model never sees them.
- Variant IV's test months contain exactly the same applications as Variant III's, so the two give identical results. The bias in Variants IV and V is only in months 0-5, which the Base model never trains on.

## Notebook 02: what each section does

| Section | What happens |
|---|---|
| 1. Imports | Libraries and plot settings |
| 2. Load and split | Loads Base, splits it by month, defines the metric |
| 3. Preprocessing | `-1` / negative values become NaN with a `<column>_missing` flag, `device_fraud_count` dropped, `income` rounded, categories fixed, `x1`/`x2` dropped |
| 4. Feature engineering | Tests each feature idea one at a time on month 4 (LightGBM and logistic regression, 5 seeds each), plus a drift check, then fixes the final feature set |
| 5. Model comparison | Logistic regression, LightGBM and XGBoost with their starting settings |
| 6. Tuning | Optuna, 50 trials for each of the three models; the top 3 trials of each are re-checked over 5 seeds, and the best tuned model is picked automatically |
| 7. Freeze the final model | Trains the final model for each seed, picks the thresholds on month 5, saves everything |
| 8. Score on Base and all 5 variants | Scores months 6-7 of every dataset with the frozen model and reports the metrics |

## Notebook 03: human-AI review of fraud alerts (FiFAR)

**Question:** analysts can only review part of the fraud alerts. Which alerts should the model decide by itself, and which should go to an analyst?

The frozen model from notebook 02 is reused unchanged; nothing is retrained. FiFAR's alerts cover months 3-7:

| Months | Used for |
|---|---|
| 3 | not used (the model was trained on these applications) |
| 4-6 | history: measuring each analyst, picking a threshold for alerts and the best analysts |
| 7 | test: comparing the review policies (4,457 alerts) |

| Section | What happens |
|---|---|
| 1-2. Imports, months used | Libraries, plot settings and the month split above |
| 3. Load FiFAR | Loads the alerts and analyst decisions, recovers the alert model's threshold (0.0510) and its FPR / recall per month, and links every alert to its row in `Base.csv` |
| 4. Score the alerts | Scores the alerts with the saved models (matching notebook 02's saved predictions) and compares the model with Feedzai's alert model |
| 5. Profile the analysts | Recall, FPR and age fairness of each of the 50 analysts on the history months |
| 6. Who should decide each alert? | Picks a lowest-cost threshold for alerts, then simulates 6 review policies across analyst capacity (0-100%) and cost ratios (5, 10, 20, 50), 5 runs each |
| 7. Summary | Plain-language results and recommendations |

**Cost:** cost = cost ratio × missed fraud + false alarms, where the cost ratio is how many false alarms one missed fraud is worth. Results below use a ratio of 10, per 1,000 month-7 alerts (lower is better):

| Policy | Cost | Recall | FPR | FPR 50+ / under 50 |
|---|---|---|---|---|
| AI only | 796 | 0.72 | 0.42 | 0.51 / 0.38 |
| Analysts only (random analyst, every alert) | 511 | 0.91 | 0.44 | 0.48 / 0.43 |
| 50% reviewed: random alerts | 651 | 0.82 | 0.44 | 0.49 / 0.41 |
| 50% reviewed: highest-risk alerts | 683 | 0.71 | 0.27 | 0.32 / 0.25 |
| 50% reviewed: alerts the model is least sure about | 583 | 0.85 | 0.41 | 0.48 / 0.38 |
| 50% reviewed: least-sure alerts, 10 best analysts | **510** | 0.84 | 0.30 | 0.37 / 0.27 |

- About half of all fraud is never alerted, so no review policy can catch it. Among alerts, 12% are fraud.
- On alerts, every one of the 50 analysts beats the model at the same FPR; the best catches 78% of fraud at a 1.2% FPR, where the model catches 9%.
- Sending the alerts the model is least sure about to the 10 best analysts is the cheapest capacity-limited policy at every capacity and cost ratio. At 50% capacity it matches analysts reviewing every alert.
- At a cost ratio of 50, random analysts make things worse than AI only (1,065 vs 853); only the best-analysts policy still beats it.
- Every policy wrongly flags genuine applicants aged 50+ more often. The best analysts are among the most age-biased, so the cheapest policy is not the fairest.
