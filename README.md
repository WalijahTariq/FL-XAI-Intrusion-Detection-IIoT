# FL-XAI Intrusion Detection for IIoT — reproduction with CI/CD

Reproduction of *"A Federated and Explainable Machine Learning Framework for Robust
Intrusion Detection and Network Security Enhancement in Industrial Internet of Things
(IIoT) Environments"* (ICECTE 2026), on the **Edge-IIoTset** ML dataset, with a
GitHub Actions pipeline that retrains the models and republishes a results web page
on every push.

The ML code (`src/`, the notebook and `docs/`) is based on
[fahadsid1770/FL-XAI-Intrusion-Detection-IIoT](https://github.com/fahadsid1770/FL-XAI-Intrusion-Detection-IIoT).
The paper PDF is in `docs/`.

## How it works

```
you edit code  ->  git push  ->  GitHub Actions
                                   1. test    : Python tests + TypeScript type-check/build
                                   2. train   : runs scripts/run_pipeline.py on the real dataset
                                                -> results.json (fails if accuracy < 0.90)
                                   3. deploy  : builds the TypeScript page with results.json
                                                -> publishes it on GitHub Pages
```

The page shows the commit (ID, message, author, time) and every setting it was built
from, and highlights any hyperparameter that differs from the paper's Table I. So a
change you push shows up on the page once the pipeline finishes.

Pull requests only run the **test** job.

## Repository layout

| Path | What it is |
|---|---|
| `src/` | The pipeline: preprocessing, SMOTE, models, calibration, SHAP, federated simulation |
| `src/config.py` | All settings (seed, split, hyperparameters from the paper's Table I) |
| `scripts/run_pipeline.py` | Runs everything and writes `frontend/public/results.json` |
| `scripts/compress_dataset.py` | Makes `data/ML-EdgeIIoT-dataset.csv.gz` (GitHub rejects files > 100 MB) |
| `data/` | The dataset (`.csv.gz` is committed; the plain `.csv` is git-ignored) |
| `tests/` | Pytest tests (use a tiny synthetic dataset with the real 63-column schema) |
| `frontend/` | TypeScript + HTML + CSS results page (Vite) |
| `.github/workflows/ci-cd.yml` | The CI/CD pipeline |
| `FL-XAI-intrusion-detection.ipynb` | Original notebook (optional, for exploring) |
| `docs/` | Paper PDF and the original implementation checklist |

## Run locally (Windows, VS Code terminal)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

python -m pytest                      # tests (about 1-2 minutes)
python scripts/run_pipeline.py        # full pipeline on the real dataset

cd frontend
npm install
npm run dev                           # open the address it prints to see the page
```

## Verified result (real dataset)

Measured on the real `ML-EdgeIIoT-dataset.csv` before publishing this repository
(Python 3.12, 2 CPU cores; the full pipeline took about 21 minutes):

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | 0.9299 | 0.9302 |
| CatBoost | 0.9341 | 0.9349 |
| XGBoost | 0.9450 | 0.9457 |
| LGBM | 0.9456 | 0.9464 |
| Hybrid XGB-LGBM | 0.9464 | 0.9472 |
| **Hybrid XGB-LGBM (calibrated)** | **0.9461** | **0.9469** |
| Federated (4 non-IID clients) | 0.9365 | 0.9370 |

Paper (hybrid): accuracy 0.9371, F1 0.9374 — this run is about 0.9 points higher.
The live page always shows the numbers from the latest push.

## Known differences from the paper

Small differences from the paper's numbers are expected:

- **Classes.** This code keeps 13 classes. The paper's Fig. 1 shows about 8 classes
  (the largest, 49,396 rows, appears to be the four DDoS types merged), so the task here
  is not identical.
- **SMOTE before the split.** Like the paper, SMOTE is applied to the whole dataset
  before the 80/20 split, so some synthetic test rows are interpolated from training
  rows. This likely makes the scores optimistic.
- **Seed.** The paper does not state a random seed; `42` is used here.
- **Hyperparameters** are taken directly from the paper's Table I (no Optuna search is
  run in the pipeline; `src/optimization.py` is kept for reference).
- **Federated simulation.** If a client's Dirichlet shard has no rows of a class, that
  client's model gives that class probability 0 instead of crashing. With the paper's
  settings every client gets every class, so results are unchanged.

## Data notes

- Dataset: `ML-EdgeIIoT-dataset.csv` (157,800 rows, 63 columns) from the Kaggle dataset
  `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`, folder
  *Selected dataset for ML and DL*.
- **MITM** and **Fingerprinting** are dropped; **Ransomware** is recoded to **Zero_Day**.
- Identifier and free-text columns (IPs, timestamps, payloads, URLs) are dropped.
