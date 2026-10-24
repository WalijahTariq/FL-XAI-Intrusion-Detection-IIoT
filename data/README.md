# Dataset

This folder must contain **`ML-EdgeIIoT-dataset.csv.gz`** (committed to git).

1. Download `ML-EdgeIIoT-dataset.csv` (157,800 rows, 63 columns) from Kaggle:
   dataset `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`,
   folder *Edge-IIoTset dataset / Selected dataset for ML and DL*.
   Do **not** use `DNN-EdgeIIoT-dataset.csv` (2.2 million rows).
2. Put it in this folder.
3. From the repository root run: `python scripts/compress_dataset.py`
4. Commit `data/ML-EdgeIIoT-dataset.csv.gz`. The plain `.csv` is ignored by git
   because GitHub rejects files over 100 MB.
