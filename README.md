# BT4012 Fraud Analytics Group Project - Group 10

Fraud detection on the [Bank Account Fraud (BAF) Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (NeurIPS 2022).

## Repository structure

```
data/         # the BAF CSVs go here (not committed, see "Getting the data")
notebooks/
  01_eda.ipynb
requirements.txt
```

## Prerequisites

- Python 3.12 (3.11 or newer is required by pandas 3)

## Setup

From the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code / Jupyter, select the `.venv` kernel before running a notebook.

## Getting the data

The dataset is about 1.3 GB, so it is not stored in this repository. Only the empty `data/` folder is committed.

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
   - **Option A (browser):** click **Download**, then unzip the archive.
   - **Option B (Kaggle CLI):** with an [API token](https://www.kaggle.com/docs/api) set up, run
     ```sh
     pip install kaggle
     kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data --unzip
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

## Running the notebooks

Open `notebooks/01_eda.ipynb` and run all cells. It reads the data with relative paths (`../data/...`), so run it from the `notebooks/` folder, which is the default in VS Code and Jupyter. A full run takes about a minute, most of it loading the six CSVs.
