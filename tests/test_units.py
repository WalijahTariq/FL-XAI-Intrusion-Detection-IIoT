"""Fast unit tests for individual pipeline pieces."""
from __future__ import annotations

import numpy as np
import pytest

from src import config, data_loader, federated, models, preprocessing
from tests.synthetic_data import make_synthetic_frame


def test_loader_finds_gz_copy(synthetic_csv_gz):
    # Asking for the plain .csv must fall back to the committed .csv.gz copy.
    plain = synthetic_csv_gz.with_name("ML-EdgeIIoT-dataset.csv")
    assert data_loader.resolve_dataset_path(plain) == synthetic_csv_gz
    df = data_loader.load_raw_dataset(plain)
    assert df.shape[1] == 63


def test_loader_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        data_loader.resolve_dataset_path(tmp_path / "nope.csv")


def test_preprocessing_matches_paper_rules():
    pre = preprocessing.preprocess(make_synthetic_frame())
    for dropped in config.DROPPED_CLASSES + config.ZERO_DAY_SOURCE_CLASSES:
        assert dropped not in pre.class_names
    assert config.ZERO_DAY_LABEL in pre.class_names
    assert len(pre.class_names) == 13
    assert len(pre.feature_names) == 45
    assert not set(config.DROP_COLUMNS) & set(pre.feature_names)
    # z-score scaling: every feature has mean ~0
    assert np.allclose(pre.X.mean(axis=0), 0, atol=1e-6)


def test_dirichlet_partition_uses_every_sample_once():
    y = np.repeat(np.arange(5), 40)
    parts = federated.dirichlet_partition(y, n_clients=4, alpha=0.5)
    combined = np.concatenate(parts)
    assert len(parts) == 4
    assert sorted(combined.tolist()) == list(range(len(y)))


def test_federated_client_missing_a_class_still_works():
    # Client 0 never sees class 1; aggregation must still return all 4 classes.
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 50)
    X = rng.normal(size=(200, 4)) + y[:, None]
    shards = [np.flatnonzero(y != 1), np.arange(200)]
    clients = federated.train_federated_clients(X, y, shards, learner="xgboost")
    proba = federated.federated_predict_proba(clients, X)
    assert proba.shape == (200, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(clients[0].predict_proba(X)[:, 1] == 0)
    assert set(federated.federated_predict(clients, X)) <= {0, 1, 2, 3}


def test_soft_voting_probabilities_are_averages():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    y = np.repeat(np.arange(3), 40)
    hybrid = models.build_hybrid(
        models.build_xgboost({"n_estimators": 10}), models.build_lightgbm({"n_estimators": 10})
    ).fit(X, y)
    expected = (hybrid.xgb.predict_proba(X) + hybrid.lgbm.predict_proba(X)) / 2
    assert np.allclose(hybrid.predict_proba(X), expected)
    assert np.allclose(hybrid.predict_proba(X).sum(axis=1), 1.0)
